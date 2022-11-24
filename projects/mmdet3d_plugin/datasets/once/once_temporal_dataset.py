import copy
import pickle
import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
import tempfile
from pathlib import Path
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import Box3DMode, points_cam2img
from mmdet3d.datasets.kitti_dataset import KittiDataset
from mmdet3d.core.bbox import box_np_ops
from mmcv.ops.points_in_boxes import points_in_boxes_cpu
from .once_toolkits import Octopus
from mmdet3d.datasets import build_dataset
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
import random
from mmcv.parallel import DataContainer as DC
from .once_dataset import ONCEDataset
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


def create_rotataion_4x4_matrix(pose):
    from scipy.spatial.transform import Rotation as R
  
    translation = pose[4:]
    rotation_matrix = R.from_quat(pose[:4]).as_matrix()
    matrix = np.eye(4)
    matrix[:3,:3] = rotation_matrix
    matrix[:3,3] = translation
    return matrix

@DATASETS.register_module()
class ONCETemporalDataset(ONCEDataset):

    def __init__(self,
                 *args,
                 queue_length = 3, skip_len = 0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.skip_len = skip_len

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        # import time
        # _ = time.time()
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            # print('dataloading cost: {} ms'.format(time.time()-_))
            return data

    def prepare_train_data(self, index):
        #[T, T-1]
        # index = index * self.load_interval
        idx_list = list(range(index-self.queue_length, index))
        random.shuffle(idx_list)
        idx_list = idx_list[self.skip_len:] + [index]#skip frame
        idx_list = sorted(idx_list, reverse=True)
        data_queue = []
        scene_id = None
      
        for i in idx_list:
            i = max(0,i)
            input_dict = self.get_data_info(i)
            if scene_id == None: scene_id = input_dict['seq_id']
            if input_dict is None: return None
            if scene_id == input_dict['seq_id']:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
            data_queue.insert(0, copy.deepcopy(example))
        if self.filter_empty_gt and\
            (data_queue[-1] is None or ~(data_queue[-1]['gt_labels_3d']._data != -1).any()):
            return None
        #data_queue: T-len+1, T
        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        input: queue: dict of [T-len+1, T], containing data_info
        convert sample queue into one single sample.
        calculate transformation from ego_now to image_old
        note that we dont gather gt objects of previous frames
        """
        # oldname='queue'
        # np.save('debug/debug_temporal1/'+oldname, queue)
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        if 'points' in queue[0].keys():
            points_list = [each['points'].data for each in queue]
            queue[-1]['points'] = DC(points_list, cpu_only=False, stack=False)
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            # import ipdb
            # ipdb.set_trace()
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = metas_map[i]['can_bus'][:3]
                prev_angle = metas_map[i]['can_bus'][-1]
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = metas_map[i]['can_bus'][:3]
                tmp_angle = metas_map[i]['can_bus'][-1]
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = tmp_pos
                prev_angle = tmp_angle
        # ego2global = queue[-1]['img_metas'].data['pose']
        # for i, each in enumerate(queue):
        #     metas_map[i] = each['img_metas'].data
        #     # pose: quat_x, quat_y, quat_z, quat_w, trans_x, trans_y, trans_z
        #     # matrix = metas_map[i]['pose']
            
        #     global2ego_old = np.linalg.inv(metas_map[i]['pose'])
        #     ego2img_old_rts = []
        #     for ego_old2img_old in metas_map[i]['lidar2img']:
        #         ego2img_old =ego_old2img_old @ global2ego_old @ ego2global #@pt_ego
        #         ego2img_old_rts.append(ego2img_old)
        #     metas_map[i]['lidar2img'] = ego2img_old_rts
        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        return queue



    def get_data_info(self, index):


        info = self.data_infos[index]
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
       
        input_dict = {
            'frame_id': frame_id,
            'sample_idx': frame_id,
            'seq_id': seq_id
        }
        input_dict['pose'] = create_rotataion_4x4_matrix(info['pose'])
        can_bus = np.zeros(18)
        ego_pose_r = np.array(input_dict['pose'])[:3, :3]
        ego_pose_t = np.array(input_dict['pose'])[:3, 3] 
        can_bus[:3] = ego_pose_t
        can_bus[3:7] = Quaternion(matrix=ego_pose_r)
        patch_angle = quaternion_yaw(Quaternion(matrix=ego_pose_r)) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict.update({
            'can_bus': can_bus
        })

        if 'annos' in info:
            annos = info['annos']
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['boxes_3d'],
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })
        

        # the Tr_velo_to_cam is computed for all images but not saved in .info for img1-4
        # the size of img0-2: 1280x1920; img3-4: 886x1920
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []

            for cam_name in self.cam_names:
                cam_path = os.path.join(self.data_root,'data', str(seq_id).zfill(6), cam_name, '{}.jpg'.format(frame_id))

                calib_info = info['calib'][cam_name]
                cam_2_velo = calib_info['cam_to_velo']
                lidar2cam_rt = np.linalg.inv(cam_2_velo)
                # cam_intri = np.hstack([calib_info['cam_intrinsic'], np.zeros((3, 1), dtype=np.float32)])
                viewpad = np.eye(4)
                viewpad[:calib_info['cam_intrinsic'].shape[0], :calib_info['cam_intrinsic'].shape[1]] = calib_info['cam_intrinsic']
                # lidar2img = cam_intri @ np.linalg.inv(cam_2_velo)
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                image_paths.append(cam_path)
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt)


        pts_filename = self._get_pts_filename(seq_id, frame_id)
        input_dict.update({
            'pts_filename': pts_filename,
            'img_prefix':None,
        })
        if self.modality['use_camera']:
            input_dict['img_filename'] = image_paths
            input_dict['lidar2img'] = lidar2img_rts
            input_dict['cam_intrinsic'] = image_paths
            input_dict['lidar2cam'] = lidar2img_rts        

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
      

        annos = info['annos']
        # we need other objects to avoid collision when sample
      
        gt_names = annos['name']
        gt_bboxes_3d = annos['boxes_3d'].astype(np.float32)
        gt_bboxes_3d[:,6] = - gt_bboxes_3d[:,6]

        # convert gt_bboxes_3d to velodyne coordinates
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d,
        box_dim=gt_bboxes_3d.shape[-1],
        origin=(0.5,0.5,0.5))
        gt_bboxes = annos['boxes_2d']

        # selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        # gt_bboxes = gt_bboxes[selected].astype('float32')
        # gt_names = gt_names[selected]

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names)
        return anns_results



