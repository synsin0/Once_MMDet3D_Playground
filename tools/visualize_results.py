import mmcv
from mmdet3d.apis import init_model, inference_detector, show_result_meshlab

from projects.mmdet3d_plugin.centerpoint import CenterPointONCE

config_file = '/nfs/volume-382-110/shiyining_i/BEV4D/work_dirs/centerpoint_0075voxel_second_secfpn_4x8_cyclic_80e_once/centerpoint_0075voxel_second_secfpn_4x8_cyclic_80e_once.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/nfs/volume-382-110/shiyining_i/BEV4D/work_dirs/centerpoint_0075voxel_second_secfpn_4x8_cyclic_80e_once/epoch_80.pth'
# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single sample
pcd = '/nfs/volume-382-110/shiyining_i/BEV4D/data/once/data/000149/lidar_roof/1618697290399.bin'
result, data = inference_detector(model, pcd)


# show the results
out_dir = './'
show_result_meshlab(data, result, out_dir)