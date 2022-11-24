import mmcv
import numpy as np
import os
import tempfile
import torch
from mmcv.utils import print_log
from os import path as osp
# ERROR ROOT at LINE 331, AT line 236 in format_result, we adjust the worker to be really small
from mmdet.datasets import DATASETS #really fucked up for not adding '3d'
from mmdet3d.core.bbox import Box3DMode, points_cam2img
from mmdet3d.datasets.kitti_dataset import KittiDataset
from .waymo_let_metric import compute_waymo_let_metric
import copy
from mmcv.parallel import DataContainer as DC
import random

from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from .temporal_vis import save_temporal_frame
from .waymo_mv_dataset import CustomWaymoDataset


@DATASETS.register_module()
class CustomTemporalWaymoDataset(CustomWaymoDataset):

    CLASSES = ('Car', 'Pedestrian', 'Cyclist')

    def __init__(self,
                 *args,
                 queue_length = 4, skip_len = 0,
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
        index = index * self.load_interval
        idx_list = list(range(index-self.queue_length, index))
        random.shuffle(idx_list)
        idx_list = idx_list[self.skip_len:] + [index]#skip frame
        idx_list = sorted(idx_list, reverse=True)
        data_queue = []
        scene_id = None
        for i in idx_list:
            i = max(0,i)
            input_dict = self.get_data_info(i)
            if scene_id == None: scene_id = input_dict['sample_idx']//1000
            if input_dict is None: return None
            if scene_id == input_dict['sample_idx']//1000:
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
        ego2global = queue[-1]['img_metas'].data['pose']
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            global2ego_old = np.linalg.inv(metas_map[i]['pose'])
            ego2img_old_rts = []
            for ego_old2img_old in metas_map[i]['lidar2img']:
                ego2img_old =ego_old2img_old @ global2ego_old @ ego2global #@pt_ego
                ego2img_old_rts.append(ego2img_old)
            metas_map[i]['lidar2img'] = ego2img_old_rts
        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        # breakpoint()
        # save_temporal_frame(queue)
        # name = 'queue_union'
        # np.save('debug/debug_temporal1/'+name, queue)
        # breakpoint()
        return queue

    def get_data_info(self, index):
        if self.test_mode == True:
            info = self.data_infos[index]
        else: 
            info = self.data_infos_full[index]
        
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        lidar2img = P0 @ rect @ Trv2c

        # the Tr_velo_to_cam is computed for all images but not saved in .info for img1-4
        # the size of img0-2: 1280x1920; img3-4: 886x1920
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []

            # load calibration for all 5 images.
            calib_path = img_filename.replace('image_0', 'calib').replace('.png', '.txt')
            Tr_velo_to_cam_list = []
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            for line_num in range(6, 6 + self.num_views):
                trans = np.array([float(info) for info in lines[line_num].split(' ')[1:13]]).reshape(3, 4)
                trans = np.concatenate([trans, np.array([[0., 0., 0., 1.]])], axis=0).astype(np.float32)
                Tr_velo_to_cam_list.append(trans)
            assert np.allclose(Tr_velo_to_cam_list[0], info['calib']['Tr_velo_to_cam'].astype(np.float32))

            for idx_img in range(self.num_views):
                rect = info['calib']['R0_rect'].astype(np.float32)
                # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
                Trv2c = Tr_velo_to_cam_list[idx_img]
                P0 = info['calib'][f'P{idx_img}'].astype(np.float32)
                lidar2img = P0 @ rect @ Trv2c

                image_paths.append(img_filename.replace('image_0', f'image_{idx_img}'))
                lidar2img_rts.append(lidar2img)

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
        )
        if self.modality['use_camera']:
            input_dict['img_filename'] = image_paths
            input_dict['lidar2img'] = lidar2img_rts

        # There are no pose info in default waymo_*.pkl
        input_dict['pose'] = info['pose']

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        
        return input_dict

    def get_ann_info(self, index):
        # Use index to get the annos, thus the evalhook could also use this api
        if self.test_mode == True:
            info = self.data_infos[index]
        else: info = self.data_infos_full[index]
        
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)

        # convert gt_bboxes_3d to velodyne coordinates
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c))
        gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]

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