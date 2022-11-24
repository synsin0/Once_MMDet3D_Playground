import numpy as np
import mmcv
import torch
import cv2
import os
import time
from mmdet3d.core.bbox import Box3DMode, points_cam2img
from mmdet3d.datasets.kitti_dataset import KittiDataset

import matplotlib.pyplot as plt
from matplotlib import patches

def save_bbox2img(img, gt_bboxes_3d, img_metas, img_filenames, dirname='debug_coord', name = None, colors = None):
    # print(img)
    ds_name = 'waymo' if len(img_metas[0]['filename'])==5 else 'nuscene'
    # for i,name in enumerate(img_metas[0]['filename']):
    #     img_out = img[0][i].permute(1,2,0).detach().cpu().numpy()
    #     img_in = cv2.imread(name)
    #     print(img_in)
    #     print(img_out.shape)
    #     cv2.imwrite('debug_forward/{}_input_vis_finalcheck_{}.png'.format(ds_name, i),img_out)
    gt_bboxes_3d = gt_bboxes_3d[0]
    reference_points = gt_bboxes_3d.gravity_center.view(1, -1, 3) # 1 num_gt, 3
    # reference_points = gt_bboxes_3d.bottom_center.view(1, -1, 3)
#     print(reference_points.size())
    # num_gt as num_query
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    # reference_points (B, num_queries, 4) ，to homogeneous coordinate
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

    B, num_query = reference_points.size()[:2]
#     print(num_query)
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)##B num_c num_q 4 ,1
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)   # B num_c num_q 4 4
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)    # B num_c num_q 4
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)   #filter out negative depth, B num_c num_q
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(  #z for depth, too shallow will cause zero division
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)    # eps controls minimum

    # ref_point_visualize = reference_points_cam.clone()
    #try to normalize to the coordinate in feature map
    if type(img_metas[0]['ori_shape']) == tuple:    
        #same size for all images, nuscene  900*1600,floor to 928*1600
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    else:
        #diff size,1280*1920 and 886*1920, waymo, get it to the normorlized point, floor 886 to 896 to meet divisor 32, 
        # which is 0.7 out of 1 against 1280, that is to say, the remaining 30% is padding
        reference_points_cam[..., 0] /= img_metas[0]['ori_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['ori_shape'][0][0]
#         print(img_metas[0]['ori_shape'])
        mask[:, 3:5, :] &= (reference_points_cam[:, 3:5, :, 1:2] < 0.7)

    # reference_points_cam = (reference_points_cam - 0.5) * 2       #0~1 to -1~1
    mask = (mask & (reference_points_cam[..., 0:1] > 0)  #we should change the criteria for waymo cam 3~4
                 & (reference_points_cam[..., 0:1] < 1)   # which is -1~0.4
                 & (reference_points_cam[..., 1:2] > 0) 
                 & (reference_points_cam[..., 1:2] < 1))  # B num_cam num_query
#     print(mask.any()==False)
    if (name == None):
        name = str(time.time())
    img_ret=[]
    for i in range(num_cam):
#         print(img.shape)
#         img_out = np.array(img[0][i].permute(1,2,0).detach().cpu())
#         h,w,_ = img_out.shape
        img_out = cv2.imread(img_filenames[i])    #'/home/zhengliangtao/'+
        h,w,_ = img_metas[0]['ori_shape'][0]
#         print(h,w)
        for j in range(num_query):
            pt = reference_points_cam[0,i,j]
#             print(pt,'  ',mask[0,i,j], ' ',(int(pt[0]*w),' ',int(pt[1]*h)))
            if mask[0,i,j] == True:
                color = [int(x) for x in colors[j]]
                cv2.circle(img_out, (int(pt[0]*w),int(pt[1]*h)), radius=5,color=color, thickness = 4)
#         print(dirname+'/{}_{}_{}.png'.format(ds_name,name, i))
#         cv2.imwrite(dirname+'/{}_{}_{}.png'.format(ds_name,name, i), img_out)
        img_ret.append(img_out)
    return img_ret

def show_camera_image(camera_image, layout):
  """Display the given camera image."""
  ax = plt.subplot(*layout)
  plt.imshow(camera_image)
  plt.grid(False)
  plt.axis('off')
  return ax

def save_temporal_frame(union, dirname='debug/debug_temporal1'):
    # union = np.load('queue_union.npy',allow_pickle=True).reshape(1)[0]
    gt = union['gt_bboxes_3d'].data
    imgs = union['img'].data
    name = str(union['img_metas'].data[1]['sample_idx']) 
    # dirname='/home/zhengliangtao/pure-detr3d/debug/debug_temporal1'
    colors = np.random.randint(256, size=(300,3))
    out1 = save_bbox2img(imgs[1:2], [gt], [union['img_metas'].data[1]], union['img_metas'].data[1]['filename'], 
                dirname= dirname, name = name, colors = colors)
    out0 = save_bbox2img(imgs[1:2], [gt], [union['img_metas'].data[0]], union['img_metas'].data[0]['filename'], 
                dirname= dirname, name = name+'_prev', colors = colors)
    plt.figure(figsize=(40, 60))    
    for i,img_out in enumerate(out0):
        show_camera_image(img_out[...,::-1], [5, 2, i*2+1])
    for i,img_out in enumerate(out1):
        show_camera_image(img_out[...,::-1], [5, 2, i*2+2])
    plt.savefig(dirname+'/{}_{}_all.png'.format('waymo',name))