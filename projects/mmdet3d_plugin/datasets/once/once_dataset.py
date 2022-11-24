import copy
import pickle
import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
import tempfile
from pathlib import Path
from mmdet3d.core import show_result

from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import Box3DMode, points_cam2img
from mmdet3d.datasets.kitti_dataset import KittiDataset
from mmdet3d.core.bbox import box_np_ops
from mmcv.ops.points_in_boxes import points_in_boxes_cpu
from .once_toolkits import Octopus
from mmdet.datasets import build_dataset
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
import mmcv
@DATASETS.register_module()
class ONCEDataset(KittiDataset):
    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 num_views=7,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 use_point_painting=False,
                 _merge_all_iters_to_one_epoch=False,
                 load_interval=2, 
                 gt_bin = None,
                 pcd_limit_range=[-75.2, -75.2, -5.0, 75.2, 75.2, 3.0]):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range)

        self.split = split
        assert self.split in ['train', 'val', 'test', 'raw_small', 'raw_medium', 'raw_large']
        self.use_point_painting = use_point_painting
        self._merge_all_iters_to_one_epoch = _merge_all_iters_to_one_epoch
        self.load_interval = load_interval  # Once dataset annotates one frame by one unannotated
        self.root_path = data_root    
        split_dir = os.path.join(self.root_path , 'ImageSets' , (self.split + '.txt'))
        self.sample_seq_list = [x.strip() for x in open(split_dir).readlines()] if split_dir else None
        self.cam_names = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
        self.cam_tags = ['top', 'top2', 'left_back', 'left_front', 'right_front', 'right_back', 'back']
        self.toolkits = Octopus(self.root_path)


        # self.include_once_data(self.split)


    # def include_once_data(self, split):
    #     print('Loading ONCE dataset')
    #     data_infos = []

    #     for info_path in self.dataset_cfg.INFO_PATH[split]:
    #         info_path = self.root_path / info_path
    #         if not info_path.exists():
    #             continue
    #         with open(info_path, 'rb') as f:
    #             infos = pickle.load(f)
    #             data_infos.extend(infos)

    #     def check_annos(info):
    #         return 'annos' in info

    #     if self.split != 'raw':
    #         data_infos = list(filter(check_annos,data_infos))

    #     self.data_infos.extend(data_infos)

    #     print('Total samples for ONCE dataset: %d' % (len(data_infos)))

    def set_split(self, split):
        # super().__init__(
        #     dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        # )
        self.split = split

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_seq_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, sequence_id, frame_id):
        return self.toolkits.load_point_cloud(sequence_id, frame_id)

    def get_image(self, sequence_id, frame_id, cam_name):
        return self.toolkits.load_image(sequence_id, frame_id, cam_name)

    def project_lidar_to_image(self, sequence_id, frame_id):
        return self.toolkits.project_lidar_to_image(sequence_id, frame_id)

    def point_painting(self, points, info):
        semseg_dir = './' # add your own seg directory
        used_classes = [0,1,2,3,4,5]
        num_classes = len(used_classes)
        frame_id = str(info['frame_id'])
        seq_id = str(info['sequence_id'])
        painted = np.zeros((points.shape[0], num_classes)) # classes + bg
        for cam_name in self.cam_names:
            img_path = Path(semseg_dir) / Path(seq_id) / Path(cam_name) / Path(frame_id+'_label.png')
            calib_info = info['calib'][cam_name]
            cam_2_velo = calib_info['cam_to_velo']
            cam_intri = np.hstack([calib_info['cam_intrinsic'], np.zeros((3, 1), dtype=np.float32)])
            point_xyz = points[:, :3]
            points_homo = np.hstack(
                [point_xyz, np.ones(point_xyz.shape[0], dtype=np.float32).reshape((-1, 1))])
            points_lidar = np.dot(points_homo, np.linalg.inv(cam_2_velo).T)
            mask = points_lidar[:, 2] > 0
            points_lidar = points_lidar[mask]
            points_img = np.dot(points_lidar, cam_intri.T)
            points_img = points_img / points_img[:, [2]]
            uv = points_img[:, [0,1]]
            #depth = points_img[:, [2]]
            seg_map = np.array(Image.open(img_path)) # (H, W)
            H, W = seg_map.shape
            seg_feats = np.zeros((H*W, num_classes))
            seg_map = seg_map.reshape(-1)
            for cls_i in used_classes:
                seg_feats[seg_map==cls_i, cls_i] = 1
            seg_feats = seg_feats.reshape(H, W, num_classes).transpose(2, 0, 1)
            uv[:, 0] = (uv[:, 0] - W / 2) / (W / 2)
            uv[:, 1] = (uv[:, 1] - H / 2) / (H / 2)
            uv_tensor = torch.from_numpy(uv).unsqueeze(0).unsqueeze(0)  # [1,1,N,2]
            seg_feats = torch.from_numpy(seg_feats).unsqueeze(0) # [1,C,H,W]
            proj_scores = F.grid_sample(seg_feats, uv_tensor, mode='bilinear', padding_mode='zeros')  # [1, C, 1, N]
            proj_scores = proj_scores.squeeze(0).squeeze(1).transpose(0, 1).contiguous() # [N, C]
            painted[mask] = proj_scores.numpy()
        return np.concatenate([points, painted], axis=1)

    # def __len__(self):
    #     # if self._merge_all_iters_to_one_epoch:
    #     #     return len(self.data_infos) * self.total_epochs

    #     return len(self.data_infos)

    def get_data_info(self, index):
        # if self._merge_all_iters_to_one_epoch:
        #     index = index % len(self.data_infos)

        info = self.data_infos[index]
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
       
        input_dict = {
            'frame_id': frame_id,
            'sample_idx': frame_id,
            'seq_id': seq_id
        }

        if 'annos' in info:
            annos = info['annos']
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['boxes_3d'],
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })
        
        # if self.modality['use_lidar']:
        #     points = self.get_lidar(seq_id, frame_id)

        #     if self.use_point_painting:
        #         points = self.point_painting(points, info)
        #     input_dict.update({
        #         'points': points,
        #     })

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

        input_dict['pose'] = info['pose']
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
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d)
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

    def _get_pts_filename(self,seq_id, frame_id):
        seq_str=str(seq_id).zfill(6)
        bin_path = os.path.join(self.data_root, 'data' , seq_str, 'lidar_roof', '{}.bin'.format(frame_id))
        return bin_path

 

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('once_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            if 'annos' not in infos[k]:
                continue
            print('gt_database sample: %d' % (k + 1))
            info = infos[k]
            frame_id = info['frame_id']
            seq_id = info['sequence_id']
            points = self.get_lidar(seq_id, frame_id)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['boxes_3d']

            num_obj = gt_boxes.shape[0]
            point_indices = points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (frame_id, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


    def format_results(self, outputs, pklfile_prefix=None, submission_prefix_=None):

        class_names = self.CLASSES
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_3d': np.zeros((num_samples, 7))
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):

            pred_scores = box_dict['scores_3d'].numpy()
            pred_boxes = box_dict['boxes_3d'].tensor.numpy()
            pred_boxes[:,6] = - pred_boxes[:,6]
            pred_labels = box_dict['labels_3d'].numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_3d'] = pred_boxes
            return pred_dict

        annos = []
        for index, box_dict in enumerate(outputs):
            single_pred_dict = generate_single_sample_dict(box_dict['pts_bbox'])
            annos.append(single_pred_dict)


        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = os.path.join(tmp_dir.name, 'once_results')
        else:
            pklfile_prefix = os.path.join(pklfile_prefix, 'once_results')
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(annos, out)
            print(f'Result is saved to {out}.')
        return annos, pklfile_prefix


    # def format_results(self,
    #                    outputs,
    #                    pklfile_prefix=None,
    #                    submission_prefix=None,
    #                    data_format='once'):
    #     """Format the results to pkl file.

    #     Args:
    #         outputs (list[dict]): Testing results of the dataset.
    #         pklfile_prefix (str | None): The prefix of pkl files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.
    #         submission_prefix (str | None): The prefix of submitted files. It
    #             includes the file path and the prefix of filename, e.g.,
    #             "a/b/prefix". If not specified, a temp file will be created.
    #             Default: None.
    #         data_format (str | None): Output data format. Default: 'waymo'.
    #             Another supported choice is 'kitti'.

    #     Returns:
    #         tuple: (result_files, tmp_dir), result_files is a dict containing
    #             the json filepaths, tmp_dir is the temporal directory created
    #             for saving json files when jsonfile_prefix is not specified.
    #     """
    #     if pklfile_prefix is None:
    #         tmp_dir = tempfile.TemporaryDirectory()
    #         pklfile_prefix = os.path.join(tmp_dir.name, 'results')
    #     else:
    #         tmp_dir = None

    #     assert ('once' in data_format or 'kitti' in data_format), \
    #         f'invalid data_format {data_format}'
    #     print("still work before format_results ---  if not isinstance")
    #     # np.save('debug_eval/zltwaymo_eval_result_before_format_results',outputs)
    #     # print('saved!')
    #     # exit(0)
    #     if (not isinstance(outputs[0], dict)) or 'img_bbox' in outputs[00]:
    #         raise TypeError('Not supported type for reformat results.')
    #     elif 'pts_bbox' in outputs[0]:#we go this way
    #         result_files = list()
    #         for output in outputs:

    #             result_files_ = self._format_bbox(output, self.CLASSES,
    #                                                    pklfile_prefix,
    #                                                    submission_prefix)
    #             result_files.append(result_files_)
    #     else:
    #         result_files = self._format_bbox(outputs, self.CLASSES,
    #                                               pklfile_prefix,
    #                                               submission_prefix)
    #     return result_files, tmp_dir

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            # import ipdb
            # ipdb.set_trace()
            pts_path = data_info['lidar']
            file_name = os.path.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            # points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
            #                                    Coord3DMode.DEPTH)
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.LIDAR)            
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            # show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
            #                                    Box3DMode.DEPTH)
            # pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            # show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
            #                                      Box3DMode.DEPTH)
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.LIDAR)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.LIDAR)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)


    def evaluate(self, outputs,show=False,
                 out_dir=None,
                 pipeline=None,  **kwargs ):
        from .once_eval.evaluation import get_evaluation_results
        class_names = self.CLASSES

        det_annos, tmp_dir = self.format_results(outputs,kwargs['jsonfile_prefix'] )

        if show:
            # import ipdb
            # ipdb.set_trace()
            out_dir = './once_vis/'
            self.show(outputs, out_dir, show=show, pipeline=pipeline)
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.data_infos]
        print('starting to evaluate once results')
        ap_result_str, ap_dict = get_evaluation_results(eval_gt_annos, eval_det_annos, class_names)
        curr_file = os.path.join(kwargs['jsonfile_prefix'],'once_eval.txt')

        with open(curr_file, 'w') as f:
            print(ap_result_str,file=f)
            print(ap_dict,file=f)        
        
        print(ap_result_str)
        return ap_dict

def create_data_infos(dataset_cfg, save_path, workers=4):

    dataset = build_dataset(dataset_cfg)

    splits = ['train', 'val', 'test', 'raw_small', 'raw_medium', 'raw_large']
    ignore = ['test']

    print('---------------Start to generate data infos---------------')
    for split in splits:
        if split in ignore:
            continue

        filename = 'data_infos_%s.pkl' % split
        filename = save_path / Path(filename)
        dataset.set_split(split)
        data_infos = dataset.get_infos(num_workers=workers)
        with open(filename, 'wb') as f:
            pickle.dump(data_infos, f)
        print('ONCE info %s file is saved to %s' % (split, filename))

    train_filename = save_path / 'data_infos_train.pkl'
    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split('train')
    dataset.create_groundtruth_database(train_filename, split='train')
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    parser.add_argument('--runs_on', type=str, default='server', help='')
    args = parser.parse_args()

    if args.func == 'create_data_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))


        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        once_data_path = ROOT_DIR / 'data' / 'once'
        once_save_path = ROOT_DIR / 'data' / 'once'

        # if args.runs_on == 'cloud':
        #     once_data_path = Path('/cache/once/')
        #     once_save_path = Path('/cache/once/')
        #     dataset_cfg.DATA_PATH = dataset_cfg.CLOUD_DATA_PATH

        create_data_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Bus', 'Truck', 'Pedestrian', 'Bicycle'],
            data_path=once_data_path,
            save_path=once_save_path
        )