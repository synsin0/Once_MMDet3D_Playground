
## we don't plugin this module
## the first time you eval the dataset, it create info files against tfrecord files
## and after that you get rid of using tensorflow when errors constantly occur
# Copyright (c) OpenMMLab. All rights reserved.
r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

try:
    from waymo_open_dataset import dataset_pb2 as open_dataset
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')

import mmcv
import numpy as np
import tensorflow as tf
from glob import glob
from os.path import join
from os.path import exists as osp_exists
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from time import time


class KITTI2Waymo(object):
    """KITTI predictions to Waymo converter.

    This class serves as the converter to change predictions from KITTI to
    Waymo format.

    Args:
        kitti_result_files (list[dict]): Predictions in KITTI format.
        waymo_tfrecords_dir (str): Directory to load waymo raw data.
        waymo_results_save_dir (str): Directory to save converted predictions
            in waymo format (.bin files).
        waymo_results_final_path (str): Path to save combined
            predictions in waymo format (.bin file), like 'a/b/c.bin'.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (str): Number of parallel processes.
    """

    def __init__(self,
                 kitti_result_files,
                 waymo_tfrecords_dir,
                 waymo_results_save_dir,
                 waymo_results_final_path,
                 prefix,
                 workers=64):

        self.kitti_result_files = kitti_result_files
        self.waymo_tfrecords_dir = waymo_tfrecords_dir
        self.waymo_results_save_dir = waymo_results_save_dir
        self.waymo_results_final_path = waymo_results_final_path
        self.prefix = prefix
        self.workers = int(workers)
        self.sample_index = []
        for idx, result in enumerate(kitti_result_files):
            if len(result['sample_idx']) > 0:
                sample_idx = result['sample_idx'][0]
                self.sample_index.append(f'{sample_idx:07d}')

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }

        self.T_ref_to_front_cam = np.array([[0.0, 0.0, 1.0, 0.0],
                                            [-1.0, 0.0, 0.0, 0.0],
                                            [0.0, -1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]])

        self.get_file_names()
        self.create_folder()
        tf_info_pathname = self.waymo_tfrecord_pathnames[-1].replace('.tfrecord','.pkl')
        self.first_time = (not osp_exists(join(self.waymo_tfrecords_dir, 'tf_info_all.pkl')))
        if self.first_time:
            print('it is the first time you evaluate this dataset split, we will collect info in tfrecords to speed up evaluations')
            self.gather_tfrecord_info()
        self.tf_infos = mmcv.load(join(self.waymo_tfrecords_dir, 'tf_info_all.pkl'))
        # print(self.waymo_tfrecord_pathnames)
        # print(self.name2idx)
        # print(kitti_result_files[0])
        # exit(0)
    def gather_tfrecord_info(self):
        tf_infos = {}
        for file_idx in range(len(self.waymo_tfrecord_pathnames)):
            file_pathname = self.waymo_tfrecord_pathnames[file_idx]
            file_data = tf.data.TFRecordDataset(file_pathname, compression_type='')
            ## return still got error here
            for frame_num, frame_data in enumerate(file_data):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(frame_data.numpy()))

                filename = f'{self.prefix}{file_idx:03d}{frame_num:03d}'

                for camera in frame.context.camera_calibrations:
                    # FRONT = 1, see dataset.proto for details
                    if camera.name == 1:
                        T_front_cam_to_vehicle = np.array(
                            camera.extrinsic.transform).reshape(4, 4)

                T_k2w = T_front_cam_to_vehicle @ self.T_ref_to_front_cam    # inverse(Tr_velo_to_cam)
                context_name = frame.context.name               # file name strip
                frame_timestamp_micros = frame.timestamp_micros     # timestamp

                info = {'filename': filename, 'T_k2w': T_k2w, 'context_name': context_name,\
                        'frame_timestamp_micros': frame_timestamp_micros}
                tf_infos[filename] = info
            print('done with tfrecord {}'.format(file_idx))
        mmcv.dump(tf_infos ,join(self.waymo_tfrecords_dir, 'tf_info_all.pkl'))

    def get_file_names(self):
        """Get file names of waymo raw data."""
        self.waymo_tfrecord_pathnames = sorted(
            glob(join(self.waymo_tfrecords_dir, '*.tfrecord')))
        print(len(self.waymo_tfrecord_pathnames), 'tfrecords found.')

    def create_folder(self):
        """Create folder for data conversion."""
        mmcv.mkdir_or_exist(self.waymo_results_save_dir)

    def parse_objects(self, kitti_result, T_k2w, context_name,
                      frame_timestamp_micros):
        """Parse one prediction with several instances in kitti format and
        convert them to `Object` proto.

        Args:
            kitti_result (dict): Predictions in kitti format.

                - name (np.ndarray): Class labels of predictions.
                - dimensions (np.ndarray): Height, width, length of boxes.
                - location (np.ndarray): Bottom center of boxes (x, y, z).
                - rotation_y (np.ndarray): Orientation of boxes.
                - score (np.ndarray): Scores of predictions.
            T_k2w (np.ndarray): Transformation matrix from kitti to waymo.
            context_name (str): Context name of the frame.
            frame_timestamp_micros (int): Frame timestamp.

        Returns:
            :obj:`Object`: Predictions in waymo dataset Object proto.
        """

        def parse_one_object(instance_idx):
            """Parse one instance in kitti format and convert them to `Object`
            proto.

            Args:
                instance_idx (int): Index of the instance to be converted.

            Returns:
                :obj:`Object`: Predicted instance in waymo dataset \
                    Object proto.
            """
            cls = kitti_result['name'][instance_idx]
            length = kitti_result['dimensions'][instance_idx, 0]
            height = kitti_result['dimensions'][instance_idx, 1]
            width = kitti_result['dimensions'][instance_idx, 2]
            x = kitti_result['location'][instance_idx, 0]
            y = kitti_result['location'][instance_idx, 1]
            z = kitti_result['location'][instance_idx, 2]
            rotation_y = kitti_result['rotation_y'][instance_idx]
            score = kitti_result['score'][instance_idx]

            # different conventions
            heading = -(rotation_y + np.pi / 2)
            while heading < -np.pi:
                heading += 2 * np.pi
            while heading > np.pi:
                heading -= 2 * np.pi

            box = label_pb2.Label.Box()
            box.center_x = x
            box.center_y = y
            box.center_z = z
            box.length = length
            box.width = width
            box.height = height
            box.heading = heading

            o = metrics_pb2.Object()
            o.object.box.CopyFrom(box)
            o.object.type = self.k2w_cls_map[cls]
            o.score = score

            o.context_name = context_name
            o.frame_timestamp_micros = frame_timestamp_micros

            return o
        # kitti_result['location'][instance_idx]
        kitti_result['location'][:,1] -= kitti_result['dimensions'][:,1] / 2
        center3d = kitti_result['location']#      num_q, 3
        homo = np.ones([center3d.shape[0],4])
        homo[...,:3] = center3d
        kitti_result['location'] = np.matmul(T_k2w, homo.reshape(-1,4,1)).squeeze()[:,:3]# 4,4 | num_gt,4,1  ---> num_gt,4
        objects = metrics_pb2.Objects()
        for instance_idx in range(len(kitti_result['name'])):
            o = parse_one_object(instance_idx)
            objects.objects.append(o)
        return objects


    def convert_one_pkl_style(self, i):
        _ = time()
        kitti_result = self.kitti_result_files[i]
        info = self.tf_infos.get(self.sample_index[i])
        if info == None:
            print('{} not found'.format(self.sample_index[i]))
            return

        filename = info['filename']
        T_k2w = info['T_k2w']
        context_name = info['context_name']
        frame_timestamp_micros = info['frame_timestamp_micros']
        objects = self.parse_objects(kitti_result, T_k2w, context_name, #too slow here
                                            frame_timestamp_micros)
        open(join(self.waymo_results_save_dir, f'{filename}.bin'),'wb')\
            .write(objects.SerializeToString())

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        t_st=time()
        # mmcv.track_parallel_progress(self.convert_one_pkl_style, range(len(self.sample_index)),self.workers)
                                    # parallel is not applicable since threads conflicts
        for idx in range(len(self.sample_index)):
            self.convert_one_pkl_style(idx)
        
        print('\nFinished ...')
        t_en=time()
        print('time of multi converter is {}'.format(t_en-t_st))
        # combine all files into one .bin
        pathnames = sorted(glob(join(self.waymo_results_save_dir, '*.bin')))
        combined = self.combine(pathnames)

        with open(self.waymo_results_final_path, 'wb') as f:
            f.write(combined.SerializeToString())

    def __len__(self):
        """Length of the filename list."""
        return len(self.waymo_tfrecord_pathnames)

    def transform(self, T, x, y, z):
        """Transform the coordinates with matrix T.

        Args:
            T (np.ndarray): Transformation matrix.
            x(float): Coordinate in x axis.
            y(float): Coordinate in y axis.
            z(float): Coordinate in z axis.

        Returns:
            list: Coordinates after transformation.
        """
        pt_bef = np.array([x, y, z, 1.0]).reshape(4, 1)
        pt_aft = np.matmul(T, pt_bef)
        return pt_aft[:3].flatten().tolist()

    def combine(self, pathnames):
        """Combine predictions in waymo format for each sample together.

        Args:
            pathnames (str): Paths to save predictions.

        Returns:
            :obj:`Objects`: Combined predictions in Objects proto.
        """
        combined = metrics_pb2.Objects()

        for pathname in pathnames:
            objects = metrics_pb2.Objects()
            with open(pathname, 'rb') as f:
                objects.ParseFromString(f.read())
            for o in objects.objects:
                combined.objects.append(o)

        return combined
