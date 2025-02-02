
import torch

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import trimesh

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def depth_rendering (points,K,R,T):
    
    
    
    pc = trimesh.PointCloud(points)
    pc.show
    
    camera = o3d.camera.PinholeCameraIntrinsic()
    camera.intrinsic_matrix = K

    # Set the extrinsic parameters (rotation and translation)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = T

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    renderer = o3d.visualization.rendering.OffscreenRenderer(1920, 1080)
    renderer.scene.set_background(np.asarray([0, 0, 0, 0]))  # Set background to black
    renderer.scene.add_geometry("pcd", point_cloud,mat)
    renderer.scene.camera.set_projection(K, R,T,1920,1080)
    center = [0, 0, 1]  # look_at target
    eye = [0, 0, 0]  # camera position
    up = [0, 0, -1]  # camera orientation
    renderer.scene.camera.look_at(center, eye, up)

    depth_image = np.asarray(renderer.render_to_depth_image())
    max = depth_image.max()
    normalized_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    plt.imshow(normalized_image)
    return depth_image