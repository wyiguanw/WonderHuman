import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import torchvision

import matplotlib.pyplot as plt
import cv2

def get_camrot(campos, lookat=None, inv_camera=False):
        r""" Compute rotation part of extrinsic matrix from camera posistion and
            where it looks at.

        Args:
            - campos: Array (3, )
            - lookat: Array (3, )
            - inv_camera: Boolean

        Returns:
            - Array (3, 3)

        Reference: http://ksimek.github.io/2012/08/22/extrinsic/
        """

        if lookat is None:
            lookat = np.array([0., 0., 0.], dtype=np.float32)

        # define up, forward, and right vectors
        up = np.array([0., 1., 0.], dtype=np.float32)
        if inv_camera:
            up[1] *= -1.0
        forward = lookat - campos
        forward /= np.linalg.norm(forward)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        camrot = np.array([right, up, forward], dtype=np.float32)
        return camrot


def setup_camera(img_size,index, radius, focal,total_frames):
        x = 0.
        y = -0.25
        z = radius
        campos = np.array([x, y, z], dtype='float32')
        #index = 25
        angle = 2 * np.pi / total_frames * index
        axis = np.array([0., 1., 0.], dtype=np.float32)
        Rrel = cv2.Rodrigues(angle * axis)[0]
        campos = campos @ Rrel
        camrot = get_camrot(campos, 
                            lookat=np.array([0, y, 0.]),
                            inv_camera=False)



        return camrot




height=512

view_index = 0
view_index2 = -0
idx = 50
CAM_PARAMS = {
            'radius': 5.0, 'focal': 1250
        }

R = setup_camera(img_size = height,index=view_index, total_frames=100,
                                 **CAM_PARAMS)



R2 = setup_camera(img_size = height,index=view_index2, total_frames=100,
                                 **CAM_PARAMS)


rotation_matrix = torch.from_numpy(R).to('cuda')
rotation_matrix2 = torch.from_numpy(R2).to('cuda')

image_path = f'/GaussianAvatar_normal/output_youtube6/youtube6_stage1/normal/{idx}.jpg'
image = Image.open(image_path)

rgb_image = Image.open(image_path).convert('RGB')
mask = np.array( rgb_image.copy())
mask[mask !=1 ] = 0
mask = 1 - mask
mask = torch.from_numpy(mask).to('cuda')

transform = transforms.Compose([
    transforms.ToTensor()  # Convert to tensor of shape (C, H, W) and range [0, 1]
])

normal_image = transform(rgb_image).permute(1, 2, 0).to('cuda')  # Shape: (H, W, 3)

mask = normal_image.clone()
mask[mask !=1 ] = 0
mask = 1 - mask

# Normalize the normal image to the range [-1, 1]
normal_image = normal_image * 2 - 1

normals_flat = normal_image.view(-1, 3)
normals_view_flat = torch.matmul(normals_flat, rotation_matrix.t())
H, W, _ = normal_image.shape
normals_view = normals_view_flat.view(H, W, 3)

normals_rgb = (normals_view + 1) / 2

normals_rgb = normals_rgb * mask + (1 - mask) * 255

normals_rgb = normals_rgb.permute(2,0,1)

torchvision.utils.save_image(normals_rgb, f'/GaussianAvatar_normal/output_youtube6/youtube6_stage1/normal_test/back_{idx}_{view_index}.jpg')
normal_image2 = normals_rgb.permute(1, 2, 0) # Shape: (H, W, 3)

# Normalize the normal image to the range [-1, 1]
normal_image2 = normal_image2 * 2 - 1

normals_flat2 = normal_image2.view(-1, 3)
normals_view_flat2 = torch.matmul(normals_flat2, rotation_matrix2.t())
H, W, _ = normal_image2.shape
normals_view2 = normals_view_flat2.view(H, W, 3)

normals_rgb2 = (normals_view2 + 1) / 2

normals_rgb2 = normals_rgb2.permute(2,0,1)

torchvision.utils.save_image(normals_rgb2, f'GaussianAvatar_normal/output_youtube6/youtube6_stage1/normal_test/back_2_{idx}_{view_index2}.jpg')

print(normals_view.shape)  # Should be (H, W, 3)

