# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from collections import OrderedDict
from nuscenes.utils.geometry_utils import view_points
from pathlib import Path
import os
from mmdet3d.core.bbox import box_np_ops
# from .kitti_data_utils import get_once_image_info
from .nuscenes_converter import post_process_coords

once_categories = ('Car', 'Bus', 'Truck', 'Pedestrian', 'Cyclist')


def convert_to_kitti_info_version2(info):
    """convert kitti info v1 to v2 if possible.

    Args:
        info (dict): Info of the input kitti data.
            - image (dict): image info
            - calib (dict): calibration info
            - point_cloud (dict): point cloud info
    """
    if 'image' not in info or 'calib' not in info or 'point_cloud' not in info:
        info['image'] = {
            'image_shape': info['img_shape'],
            'image_idx': info['image_idx'],
            'image_path': info['img_path'],
        }
        info['calib'] = {
            'R0_rect': info['calib/R0_rect'],
            'Tr_velo_to_cam': info['calib/Tr_velo_to_cam'],
            'P2': info['calib/P2'],
        }
        info['point_cloud'] = {
            'velodyne_path': info['velodyne_path'],
        }


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=True,
                                num_features=4):
    for info in mmcv.track_iter_progress(infos):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if remove_outside:
            points_v = box_np_ops.remove_outside_points(
                points_v, rect, Trv2c, P2, image_info['image_shape'])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
            gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)


def get_once_image_info(root_path, split, num_worker=4, sample_seq_list=None):
        import concurrent.futures as futures
        import json
        cam_names = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']

        """
        # dataset json format
        {
            'meta_info': 
            'calib': {
                'cam01': {
                    'cam_to_velo': list
                    'cam_intrinsic': list
                    'distortion': list
                }
                ...
            }
            'frames': [
                {
                    'frame_id': timestamp,
                    'annos': {
                        'names': list
                        'boxes_3d': list of list
                        'boxes_2d': {
                            'cam01': list of list
                            ...
                        }
                    }
                    'pose': list
                },
                ...
            ]
        }
        # open pcdet format
        {
            'meta_info':
            'sequence_id': seq_idx
            'frame_id': timestamp
            'timestamp': timestamp
            'lidar': path
            'cam01': path
            ...
            'calib': {
                'cam01': {
                    'cam_to_velo': np.array
                    'cam_intrinsic': np.array
                    'distortion': np.array
                }
                ...
            }
            'pose': np.array
            'annos': {
                'name': np.array
                'boxes_3d': np.array
                'boxes_2d': {
                    'cam01': np.array
                    ....
                }
            }          
        }
        """
        def process_single_sequence(seq_idx):
            print('%s seq_idx: %s' % (split, seq_idx))
            seq_infos = []
            seq_str=str(seq_idx).zfill(6)

            seq_path = Path(root_path) / 'data' / seq_str
            json_path = seq_path / ('%s.json' % seq_str)
            with open(json_path, 'r') as f:
                info_this_seq = json.load(f)
            meta_info = info_this_seq['meta_info']
            calib = info_this_seq['calib']
            for f_idx, frame in enumerate(info_this_seq['frames']):
                frame_id = frame['frame_id']
                if f_idx == 0:
                    prev_id = None
                else:
                    prev_id = info_this_seq['frames'][f_idx-1]['frame_id']
                if f_idx == len(info_this_seq['frames'])-1:
                    next_id = None
                else:
                    next_id = info_this_seq['frames'][f_idx+1]['frame_id']
                pc_path = str(seq_path / 'lidar_roof' / ('%s.bin' % frame_id))
                pose = np.array(frame['pose'])
                frame_dict = {
                    'sequence_id': seq_idx,
                    'frame_id': frame_id,
                    'timestamp': int(frame_id),
                    'prev_id': prev_id,
                    'next_id': next_id,
                    'meta_info': meta_info,
                    'lidar': pc_path,
                    'pose': pose
                }
                calib_dict = {}
                for cam_name in cam_names:
                    cam_path = str(seq_path / cam_name / ('%s.jpg' % frame_id))
                    frame_dict.update({cam_name: cam_path})
                    calib_dict[cam_name] = {}
                    calib_dict[cam_name]['cam_to_velo'] = np.array(calib[cam_name]['cam_to_velo'])
                    calib_dict[cam_name]['cam_intrinsic'] = np.array(calib[cam_name]['cam_intrinsic'])
                    calib_dict[cam_name]['distortion'] = np.array(calib[cam_name]['distortion'])
                frame_dict.update({'calib': calib_dict})

                if 'annos' in frame:
                    # print('seq {} is with annotation'.format(seq_idx))
                    annos = frame['annos']
                    boxes_3d = np.array(annos['boxes_3d'])
                    if boxes_3d.shape[0] == 0:
                        print(frame_id)
                        continue
                    boxes_2d_dict = {}
                    for cam_name in cam_names:
                        boxes_2d_dict[cam_name] = np.array(annos['boxes_2d'][cam_name])
                    annos_dict = {
                        'name': np.array(annos['names']),
                        'boxes_3d': boxes_3d,
                        'boxes_2d': boxes_2d_dict
                    }
                    bin_path = os.path.join(root_path,'data', seq_str, 'lidar_roof', '{}.bin'.format(frame_id))
                    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
                    gt_boxes_lidar = boxes_3d
                    # corners_lidar = box_np_ops.boxes3d_to_corners3d_lidar(np.array(annos['boxes_3d']))
                    num_gt = boxes_3d.shape[0]
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    # for k in range(num_gt):
                    #     import ipdb
                    #     ipdb.set_trace()
                    #     flag = box_np_ops.points_in_rbbox(points[:, 0:3], gt_boxes_lidar[k].transpose)
                    #     num_points_in_gt[k] = flag.sum()

                    indices = box_np_ops.points_in_rbbox(points[:, :3], gt_boxes_lidar)
                    num_points_in_gt = indices.sum(0)
                    # num_ignored = len(annos['dimensions']) - num_obj
                    # num_points_in_gt = np.concatenate(
                    #     [num_points_in_gt, -np.ones([num_ignored])])
                    annos_dict['num_points_in_gt'] = num_points_in_gt.astype(np.int32)
                    frame_dict.update({'annos': annos_dict})
                seq_infos.append(frame_dict)

            return seq_infos

        sample_seq_list = sample_seq_list if sample_seq_list is not None else sample_seq_list
        with futures.ThreadPoolExecutor(num_worker) as executor:
            infos = executor.map(process_single_sequence, sample_seq_list)
        
        all_infos = []

        for info in infos:
            all_infos.extend(info)
        # for seq_idx in sample_seq_list:
        #     all_infos.append(process_single_sequence(seq_idx))

        return all_infos



def create_once_info_file(data_path,
                           pkl_prefix='once',
                           save_path=None,
                           add_raw_info=False):
    """Create info file of KITTI dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
        relative_path (bool): Whether to use relative path.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))

    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    once_infos_train = get_once_image_info(
        root_path=data_path,
        split='train',
        num_worker=4, 
        sample_seq_list=train_img_ids,
        )
    once_infos_train = [info for info in once_infos_train if 'annos' in info]
    # _calculate_num_points_in_gt(data_path, once_infos_train, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Once info train file is saved to {filename}')
    mmcv.dump(once_infos_train, filename)
    once_infos_val = get_once_image_info(
        root_path=data_path,
        split='val',
        num_worker=4, 
        sample_seq_list=val_img_ids,)
    once_infos_val = [info for info in once_infos_val if 'annos' in info]
    # _calculate_num_points_in_gt(data_path, once_infos_val, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Once info val file is saved to {filename}')
    mmcv.dump(once_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Once info trainval file is saved to {filename}')
    mmcv.dump(once_infos_train + once_infos_val, filename)

    once_infos_test = get_once_image_info(
       root_path=data_path,
        split='test',
        num_worker=4, 
        sample_seq_list=test_img_ids,)

    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'Once info test file is saved to {filename}')
    mmcv.dump(once_infos_test, filename)
    if add_raw_info:
        raw_small_img_ids = _read_imageset_file(str(imageset_folder / 'raw_small.txt'))

        raw_medium_img_ids = _read_imageset_file(str(imageset_folder / 'raw_medium.txt'))
        raw_large_img_ids = _read_imageset_file(str(imageset_folder / 'raw_large.txt'))
        
        once_infos_raw_small = get_once_image_info(
        root_path=data_path,
            split='raw_small',
            num_worker=4, 
            sample_seq_list=raw_small_img_ids,)
            
        filename = save_path / f'{pkl_prefix}_infos_raw_small.pkl'
        print(f'Once info raw small file is saved to {filename}')
        mmcv.dump(once_infos_raw_small, filename)

        once_infos_raw_medium = get_once_image_info(
        root_path=data_path,
            split='raw_medium',
            num_worker=4, 
            sample_seq_list=raw_medium_img_ids,)
            
        filename = save_path / f'{pkl_prefix}_infos_raw_medium.pkl'
        print(f'Once info raw medium file is saved to {filename}')
        mmcv.dump(once_infos_raw_medium, filename)

        once_infos_raw_large = get_once_image_info(
        root_path=data_path,
            split='raw_large',
            num_worker=4, 
            sample_seq_list=raw_large_img_ids,)
            
        filename = save_path / f'{pkl_prefix}_infos_raw_large.pkl'
        print(f'Once info raw large file is saved to {filename}')
        mmcv.dump(once_infos_raw_large, filename)


def create_waymo_info_file(data_path,
                           pkl_prefix='waymo',
                           save_path=None,
                           relative_path=True,
                           max_sweeps=5):
    """Create info file of waymo dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str | None): Path to save the info file.
        relative_path (bool): Whether to use relative path.
        max_sweeps (int): Max sweeps before the detection frame to be used.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    # val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    # test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))
    train_img_ids = [each for each in train_img_ids if each % 5 == 0]
    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    waymo_infos_train = get_waymo_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        pose=True,
        image_ids=train_img_ids,
        relative_path=relative_path,
        max_sweeps=max_sweeps)
    _calculate_num_points_in_gt(
        data_path,
        waymo_infos_train,
        relative_path,
        num_features=6,
        remove_outside=False)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Waymo info train file is saved to {filename}')
    mmcv.dump(waymo_infos_train, filename)
    #
    # waymo_infos_val = get_waymo_image_info(
    #     data_path,
    #     training=True,
    #     velodyne=True,
    #     calib=True,
    #     pose=True,
    #     image_ids=val_img_ids,
    #     relative_path=relative_path,
    #     max_sweeps=max_sweeps)
    # _calculate_num_points_in_gt(
    #     data_path,
    #     waymo_infos_val,
    #     relative_path,
    #     num_features=6,
    #     remove_outside=False)
    # filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    # print(f'Waymo info val file is saved to {filename}')
    # mmcv.dump(waymo_infos_val, filename)
    # filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    # print(f'Waymo info trainval file is saved to {filename}')
    # mmcv.dump(waymo_infos_train + waymo_infos_val, filename)
    # waymo_infos_test = get_waymo_image_info(
    #     data_path,
    #     training=False,
    #     label_info=False,
    #     velodyne=True,
    #     calib=True,
    #     pose=True,
    #     image_ids=test_img_ids,
    #     relative_path=relative_path,
    #     max_sweeps=max_sweeps)
    # filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    # print(f'Waymo info test file is saved to {filename}')
    # mmcv.dump(waymo_infos_test, filename)


def _create_reduced_point_cloud(data_path,
                                info_path,
                                save_path=None,
                                back=False,
                                num_features=4,
                                front_camera_id=2):
    """Create reduced point clouds for given info.

    Args:
        data_path (str): Path of original data.
        info_path (str): Path of data info.
        save_path (str | None): Path to save reduced point cloud data.
            Default: None.
        back (bool): Whether to flip the points to back.
        num_features (int): Number of point features. Default: 4.
        front_camera_id (int): The referenced/front camera ID. Default: 2.
    """
    kitti_infos = mmcv.load(info_path)

    for info in mmcv.track_iter_progress(kitti_infos):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']

        v_path = pc_info['velodyne_path']
        v_path = Path(data_path) / v_path
        points_v = np.fromfile(
            str(v_path), dtype=np.float32,
            count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        if front_camera_id == 2:
            P2 = calib['P2']
        else:
            P2 = calib[f'P{str(front_camera_id)}']
        Trv2c = calib['Tr_velo_to_cam']
        # first remove z < 0 points
        # keep = points_v[:, -1] > 0
        # points_v = points_v[keep]
        # then remove outside.
        if back:
            points_v[:, 0] = -points_v[:, 0]
        points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,
                                                    image_info['image_shape'])
        if save_path is None:
            save_dir = v_path.parent.parent / (v_path.parent.stem + '_reduced')
            if not save_dir.exists():
                save_dir.mkdir()
            save_filename = save_dir / v_path.name
            # save_filename = str(v_path) + '_reduced'
            if back:
                save_filename += '_back'
        else:
            save_filename = str(Path(save_path) / v_path.name)
            if back:
                save_filename += '_back'
        with open(save_filename, 'w') as f:
            points_v.tofile(f)


def create_reduced_point_cloud(data_path,
                               pkl_prefix,
                               train_info_path=None,
                               val_info_path=None,
                               test_info_path=None,
                               save_path=None,
                               with_back=False):
    """Create reduced point clouds for training/validation/testing.

    Args:
        data_path (str): Path of original data.
        pkl_prefix (str): Prefix of info files.
        train_info_path (str | None): Path of training set info.
            Default: None.
        val_info_path (str | None): Path of validation set info.
            Default: None.
        test_info_path (str | None): Path of test set info.
            Default: None.
        save_path (str | None): Path to save reduced point cloud data.
        with_back (bool): Whether to flip the points to back.
    """
    if train_info_path is None:
        train_info_path = Path(data_path) / f'{pkl_prefix}_infos_train.pkl'
    if val_info_path is None:
        val_info_path = Path(data_path) / f'{pkl_prefix}_infos_val.pkl'
    if test_info_path is None:
        test_info_path = Path(data_path) / f'{pkl_prefix}_infos_test.pkl'

    print('create reduced point cloud for training set')
    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    print('create reduced point cloud for validation set')
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    print('create reduced point cloud for testing set')
    _create_reduced_point_cloud(data_path, test_info_path, save_path)
    if with_back:
        _create_reduced_point_cloud(
            data_path, train_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, val_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, test_info_path, save_path, back=True)


def export_2d_annotation(root_path, info_path, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    # get bbox annotations for camera
    kitti_infos = mmcv.load(info_path)
    cat2Ids = [
        dict(id=kitti_categories.index(cat_name), name=cat_name)
        for cat_name in kitti_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    from os import path as osp
    for info in mmcv.track_iter_progress(kitti_infos):
        coco_infos = get_2d_boxes(info, occluded=[0, 1, 2, 3], mono3d=mono3d)
        (height, width,
         _) = mmcv.imread(osp.join(root_path,
                                   info['image']['image_path'])).shape
        coco_2d_dict['images'].append(
            dict(
                file_name=info['image']['image_path'],
                id=info['image']['image_idx'],
                Tri2v=info['calib']['Tr_imu_to_velo'],
                Trv2c=info['calib']['Tr_velo_to_cam'],
                rect=info['calib']['R0_rect'],
                cam_intrinsic=info['calib']['P2'],
                width=width,
                height=height))
        for coco_info in coco_infos:
            if coco_info is None:
                continue
            # add an empty key for coco format
            coco_info['segmentation'] = []
            coco_info['id'] = coco_ann_id
            coco_2d_dict['annotations'].append(coco_info)
            coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(info, occluded, mono3d=True):
    """Get the 2D annotation records for a given info.

    Args:
        info: Information of the given sample data.
        occluded: Integer (0, 1, 2, 3) indicating occlusion state: \
            0 = fully visible, 1 = partly occluded, 2 = largely occluded, \
            3 = unknown, -1 = DontCare
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """
    # Get calibration information
    P2 = info['calib']['P2']

    repro_recs = []
    # if no annotations in info (test dataset), then return
    if 'annos' not in info:
        return repro_recs

    # Get all the annotation with the specified visibilties.
    ann_dicts = info['annos']
    mask = [(ocld in occluded) for ocld in ann_dicts['occluded']]
    for k in ann_dicts.keys():
        ann_dicts[k] = ann_dicts[k][mask]

    # convert dict of list to list of dict
    ann_recs = []
    for i in range(len(ann_dicts['occluded'])):
        ann_rec = {}
        for k in ann_dicts.keys():
            ann_rec[k] = ann_dicts[k][i]
        ann_recs.append(ann_rec)

    for ann_idx, ann_rec in enumerate(ann_recs):
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = \
            f"{info['image']['image_idx']}.{ann_idx}"
        ann_rec['sample_data_token'] = info['image']['image_idx']
        sample_data_token = info['image']['image_idx']

        loc = ann_rec['location'][np.newaxis, :]
        dim = ann_rec['dimensions'][np.newaxis, :]
        rot = ann_rec['rotation_y'][np.newaxis, np.newaxis]
        # transform the center from [0.5, 1.0, 0.5] to [0.5, 0.5, 0.5]
        dst = np.array([0.5, 0.5, 0.5])
        src = np.array([0.5, 1.0, 0.5])
        loc = loc + dim * (dst - src)
        offset = (info['calib']['P2'][0, 3] - info['calib']['P0'][0, 3]) \
            / info['calib']['P2'][0, 0]
        loc_3d = np.copy(loc)
        loc_3d[0, 0] += offset
        gt_bbox_3d = np.concatenate([loc, dim, rot], axis=1).astype(np.float32)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box_np_ops.center_to_corner_box3d(
            gt_bbox_3d[:, :3],
            gt_bbox_3d[:, 3:6],
            gt_bbox_3d[:, 6], [0.5, 0.5, 0.5],
            axis=1)
        corners_3d = corners_3d[0].T  # (1, 8, 3) -> (3, 8)
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        camera_intrinsic = P2
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token,
                                    info['image']['image_path'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            repro_rec['bbox_cam3d'] = np.concatenate(
                [loc_3d, dim, rot],
                axis=1).astype(np.float32).squeeze().tolist()
            repro_rec['velo_cam3d'] = -1  # no velocity in KITTI

            center3d = np.array(loc).reshape([1, 3])
            center2d = box_np_ops.points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            repro_rec['attribute_name'] = -1  # no attribute in KITTI
            repro_rec['attribute_id'] = -1

        repro_recs.append(repro_rec)

    return repro_recs


def generate_record(ann_rec, x1, y1, x2, y2, sample_data_token, filename):
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    key_mapping = {
        'name': 'category_name',
        'num_points_in_gt': 'num_lidar_pts',
        'sample_annotation_token': 'sample_annotation_token',
        'sample_data_token': 'sample_data_token',
    }

    for key, value in ann_rec.items():
        if key in key_mapping.keys():
            repro_rec[key_mapping[key]] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in kitti_categories:
        return None
    cat_name = repro_rec['category_name']
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = kitti_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec

