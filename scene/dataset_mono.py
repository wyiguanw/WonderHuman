import os
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join
from PIL import Image
import math
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
import cv2
from pathlib import Path
import random 


def _update_extrinsics(
        extrinsics, 
        angle, 
        trans=None, 
        rotate_axis='y'):
    r""" Uptate camera extrinsics when rotating it around a standard axis.

    Args:
        - extrinsics: Array (3, 3)
        - angle: Float
        - trans: Array (3, )
        - rotate_axis: String

    Returns:
        - Array (3, 3)
    """
    E = extrinsics
    inv_E = np.linalg.inv(E)

    camrot = inv_E[:3, :3]
    campos = inv_E[:3, 3]
    if trans is not None:
        campos -= trans

    rot_y_axis = camrot.T[1, 1]
    if rot_y_axis < 0.:
        angle = -angle
    
    rotate_coord = {
        'x': 0, 'y': 1, 'z':2
    }
    grot_vec = np.array([0., 0., 0.])
    grot_vec[rotate_coord[rotate_axis]] = angle
    grot_mtx = cv2.Rodrigues(grot_vec)[0].astype('float32')

    rot_campos = grot_mtx.dot(campos) 
    rot_camrot = grot_mtx.dot(camrot)
    if trans is not None:
        rot_campos += trans
    
    new_E = np.identity(4)
    new_E[:3, :3] = rot_camrot.T
    new_E[:3, 3] = -rot_camrot.T.dot(rot_campos)

    return new_E

def rotate_camera_by_frame_idx(
        extrinsics, 
        frame_idx, 
        trans=None,
        rotate_axis='y',
        period=196,
        inv_angle=False):
    r""" Get camera extrinsics based on frame index and rotation period.

    Args:
        - extrinsics: Array (3, 3)
        - frame_idx: Integer
        - trans: Array (3, )
        - rotate_axis: String
        - period: Integer
        - inv_angle: Boolean (clockwise/counterclockwise)

    Returns:
        - Array (3, 3)
    """

    angle = 2 * np.pi * (frame_idx / period)
    if inv_angle:
        angle = -angle
    return _update_extrinsics(
                extrinsics, angle, trans, rotate_axis)

# we reguire the data path as follows:
# data_path
#   train
#       -images 
#       -masks 
#       -cam_parms 
#       -smpl_parms.npy
#   test
#       -images
#       -masks 
#       -cam_parms 
#       -smpl_parms.npy
#each have the sanme name 
#and the smpl_parms like {beta:N 10;  trans,N 3; body_pose: N 165 or 72} 


class MonoDataset_train(Dataset):
    @torch.no_grad()
    def __init__(self, dataset_parms,
                 device = torch.device('cuda:0')):
        super(MonoDataset_train, self).__init__()

        self.dataset_parms = dataset_parms

        self.data_folder = join(dataset_parms.source_path, 'train')
        self.device = device
        self.gender = self.dataset_parms.smpl_gender

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0

        self.no_mask = bool(self.dataset_parms.no_mask)

                
        self.src_type = dataset_parms.src_type

        if dataset_parms.train_stage == 1 or self.src_type in ["zju_mocap","zju_mocap_1","monocap"]:
            print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))
        else:
            print('loading smpl data ', join(self.data_folder, 'smpl_parms_pred.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms_pred.pth'))
        # print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
        # self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))

        self.data_length = len(os.listdir(join(self.data_folder, 'images')))
        
        self.name_list = []
        for index, img in enumerate(sorted(os.listdir(join(self.data_folder, 'images')))):
            base_name = img.split('.')[0]
            self.name_list.append((index, base_name))
        
        self.image_fix = os.listdir(join(self.data_folder, 'images'))[0].split('.')[-1]
        
        if not dataset_parms.no_mask:
            self.mask_fix = os.listdir(join(self.data_folder, 'masks'))[0].split('.')[-1]

        print("total pose length", self.data_length )


        if dataset_parms.smpl_type == 'smplx':

            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]

            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
            if not torch.is_tensor(self.smpl_data['body_pose']):
                self.pose_data = torch.from_numpy(self.pose_data)
            if not torch.is_tensor(self.smpl_data['trans']):
                self.transl_data = torch.from_numpy(self.transl_data)
        else:
            self.pose_data = self.smpl_data['body_pose'][:self.data_length]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            # self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
            if not torch.is_tensor(self.smpl_data['body_pose']):
                self.pose_data = torch.from_numpy(self.pose_data)
            if not torch.is_tensor(self.smpl_data['trans']):
                self.transl_data = torch.from_numpy(self.transl_data)

        if dataset_parms.cam_static:
            cam_path = join(self.data_folder, 'cam_parms.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)

        
    def __len__(self):
        return self.data_length

    def __getitem__(self, index, ignore_list=None):
        return self.getitem(index, ignore_list)

    @torch.no_grad()
    def getitem(self, index, ignore_list):
        pose_idx, name_idx = self.name_list[index]

        image_path = join(self.data_folder, 'images' ,name_idx + '.' + self.image_fix)

        normal_path = join(self.data_folder, 'normal', 'front', name_idx + '.jpg')
        
        cam_path = join(self.data_folder, 'cam_parms', name_idx + '.npz')

        if not self.dataset_parms.no_mask:
            mask_path = join(self.data_folder, 'masks', name_idx + '.' + self.mask_fix)

        if self.dataset_parms.train_stage in [2,3]:

            inp_posmap_path = self.data_folder + '/inp_map/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_parms.inp_posmap_size), str(pose_idx).zfill(8))
            inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_parms.inp_posmap_size)]

        if not self.dataset_parms.cam_static:
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            T = np.array(extr_npy[:3, 3], np.float32)
            intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        
        else:
            R = self.R
            T = self.T
            intrinsic = self.intrinsic

        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]

        image = Image.open(image_path)
       
        width, height = image.size
        # width = int( width * 0.5)
        # height = int(height * 0.5)
        # image = image.resize((width,height))

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        if not self.dataset_parms.no_mask:
            #mask = Image.open(mask_path)
            mask = np.array(Image.open(mask_path))

            if len(mask.shape) <3:
                mask = mask[...,None]

            mask[mask < 128] = 0
            mask[mask >= 128] = 1
            color_img = image * mask + (1 - mask) * 255
            image = Image.fromarray(np.array(color_img, dtype=np.byte), "RGB")

            mask_crop = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, binary_mask = cv2.threshold(mask_crop, 127, 255, cv2.THRESH_BINARY)

            # Find contours in the mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)

            # Assuming there is only one contour (the main object), find its bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            crop_inf = np.zeros((4))
            # Calculate the crop region around the center
            #border = 60
            crop_size = max(w,h)# +border
            

            crop_inf[0] = max(0, center_x - crop_size // 2)
            crop_inf[1] = max(0, center_y - crop_size // 2)
            crop_inf[2] = min(image.size[0], crop_inf[0] + crop_size)
            crop_inf[3] = min(image.size[1], crop_inf[1] + crop_size)

            crop_inf = torch.from_numpy(crop_inf)
        
        data_item = dict()
        if self.dataset_parms.train_stage in [2,3]:
            data_item['inp_pos_map'] = inp_posmap.transpose(2,0,1)


        resized_image = torch.from_numpy(np.array(image)) / 255.0
        if len(resized_image.shape) == 3:
            resized_image =  resized_image.permute(2, 0, 1)
        else:
            resized_image =  resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

        original_image = resized_image.clamp(0.0, 1.0)

        # pose_data = self.pose_data[pose_idx]
        # transl_data = self.transl_data[pose_idx]
        
        normal = Image.open(normal_path)
        resized_normal = torch.from_numpy(np.array(normal)) / 255.0
        resized_normal =  resized_normal.permute(2, 0, 1)
        original_normal = resized_normal.clamp(0.0, 1.0)

        data_item['crop_inf'] = crop_inf
        data_item['data_type'] = "GT"
        data_item['original_image'] = original_image
        data_item['original_normal'] = original_normal
        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx
        data_item['src_type'] = self.src_type
        data_item['view_index'] = -1
        # data_item['pose_data'] = pose_data
        # data_item['transl_data'] = transl_data
        if self.dataset_parms.smpl_type == 'smplx':
            rest_pose = self.rest_pose_data[pose_idx]
            data_item['rest_pose'] = rest_pose
  
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=intrinsic, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        
        return data_item

class MonoDataset_test(Dataset):
    @torch.no_grad()
    def __init__(self, dataset_parms,
                 device = torch.device('cuda:0')):
        super(MonoDataset_test, self).__init__()

        self.dataset_parms = dataset_parms

        self.data_folder = join(dataset_parms.source_path, 'train')
        self.device = device
        self.gender = self.dataset_parms.smpl_gender

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0

        self.no_mask = bool(self.dataset_parms.no_mask)

        if dataset_parms.train_stage == 1:
            print('loading smpl data ', join(self.data_folder, 'smpl_parms_pred.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms_pred.pth'))
        else:
            print('loading smpl data ', join(self.data_folder, 'smpl_parms_pred.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms_pred.pth'))

        self.data_length = len(os.listdir(join(self.data_folder, 'images')))
        
        self.name_list = []
        for index, img in enumerate(sorted(os.listdir(join(self.data_folder, 'images')))):
            base_name = img.split('.')[0]
            self.name_list.append((index, base_name))
        
        self.image_fix = os.listdir(join(self.data_folder, 'images'))[0].split('.')[-1]
        
        if not dataset_parms.no_mask:
            self.mask_fix = os.listdir(join(self.data_folder, 'masks'))[0].split('.')[-1]

        print("total pose length", self.data_length )


        if dataset_parms.smpl_type == 'smplx':

            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
        else:
            self.pose_data = self.smpl_data['body_pose'][:self.data_length]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            # self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]

        if dataset_parms.cam_static:
            cam_path = join(self.data_folder, 'cam_parms_test.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        

        
    def __len__(self):
        return self.data_length

    def __getitem__(self, index, ignore_list=None):
        return self.getitem(index, ignore_list)

    @torch.no_grad()
    def getitem(self, index, ignore_list):
        pose_idx, name_idx = self.name_list[index]

        image_path = join(self.data_folder, 'images' ,name_idx + '.' + self.image_fix)
        
        cam_path = join(self.data_folder, 'cam_parms', name_idx + '.npz')

        if not self.dataset_parms.no_mask:
            mask_path = join(self.data_folder, 'masks', name_idx + '.' + self.mask_fix)
        if self.dataset_parms.train_stage in [2,3]:

            inp_posmap_path = self.data_folder + '/inp_map/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_parms.inp_posmap_size), str(pose_idx).zfill(8))
            inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_parms.inp_posmap_size)]

        if not self.dataset_parms.cam_static:
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            T = np.array(extr_npy[:3, 3], np.float32)
            intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        
        else:
            R = self.R
            T = self.T
            intrinsic = self.intrinsic

        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]

        pose_data = self.pose_data[pose_idx]
        transl_data = self.transl_data[pose_idx]

        image = Image.open(image_path)
        width, height = image.size

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        if not self.dataset_parms.no_mask:
            mask = np.array(Image.open(mask_path))

            if len(mask.shape) <3:
                mask = mask[...,None]

            mask[mask < 128] = 0
            mask[mask >= 128] = 1
            # color_img = image * mask 
            color_img = image * mask + (1 - mask) * 255
            image = Image.fromarray(np.array(color_img, dtype=np.byte), "RGB")

    
        data_item = dict()

        # data_item['vtransf'] = vtransf
        # data_item['query_pos_map'] = query_posmap.transpose(2,0,1)
        if self.dataset_parms.train_stage in [2,3]:
            data_item['inp_pos_map'] = inp_posmap.transpose(2,0,1)


        resized_image = torch.from_numpy(np.array(image)) / 255.0
        if len(resized_image.shape) == 3:
            resized_image =  resized_image.permute(2, 0, 1)
        else:
            resized_image =  resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

        original_image = resized_image.clamp(0.0, 1.0)

        data_item['data_type']= "test"
        data_item['original_image'] = original_image
        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx
        data_item['pose_data'] = pose_data
        data_item['transl_data'] = transl_data
        if self.dataset_parms.smpl_type == 'smplx':
            rest_pose = self.rest_pose_data[pose_idx]
            data_item['rest_pose'] = rest_pose
  
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=intrinsic, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        
        return data_item

class MonoDataset_novel_pose(Dataset):
    @torch.no_grad()
    def __init__(self, dataset_parms,
                 device = torch.device('cuda:0')):
        super(MonoDataset_novel_pose, self).__init__()


        # self.dataset_parms = dataset_parms
        # self.data_folder = join(dataset_parms.source_path,'test')

        self.dataset_parms = dataset_parms
        self.data_folder = dataset_parms.test_folder
        self.device = device
        self.gender = self.dataset_parms.smpl_gender

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0

        self.no_mask = bool(self.dataset_parms.no_mask)

        print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
        self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))

        self.data_length = self.smpl_data['body_pose'].shape[0]
        print("total pose length", self.data_length )

        if dataset_parms.smpl_type == 'smplx':

            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
        else:
            self.pose_data = self.smpl_data['body_pose']
            self.transl_data = self.smpl_data['trans']
            # self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
        for bones in self.pose_data:
            bones[59:68] = 0.
        print('novel pose shape', self.pose_data.shape)
        print('novel pose shape', self.transl_data.shape)
        if dataset_parms.cam_static:
            cam_path = join(self.data_folder, 'cam_parms.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        

        
    def __len__(self):
        return self.data_length

    def __getitem__(self, index, ignore_list=None):
        return self.getitem(index, ignore_list)

    @torch.no_grad()
    def getitem(self, index, ignore_list):

        pose_idx  =  index
        if self.dataset_parms.train_stage in [2,3]:
            inp_posmap_path = self.data_folder + '/inp_map/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_parms.inp_posmap_size), str(pose_idx).zfill(8))
            inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_parms.inp_posmap_size)]

        R = self.R
        T = self.T
        intrinsic = self.intrinsic

        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]

        pose_data = self.pose_data[pose_idx]
        transl_data = self.transl_data[pose_idx]


        width, height = 1284 , 940

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)


        data_item = dict()
        if self.dataset_parms.train_stage in [2,3]:
            data_item['inp_pos_map'] = inp_posmap.transpose(2,0,1)

        data_item['data_type']= "new"
        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx
        data_item['pose_data'] = pose_data
        data_item['transl_data'] = transl_data
        if self.dataset_parms.smpl_type == 'smplx':
            rest_pose = self.rest_pose_data[pose_idx]
            data_item['rest_pose'] = rest_pose
  
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=intrinsic, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        
        return data_item



NUM_FRAMES_observ = 50
class MonoDataset_novel_view(Dataset):
    # these code derive from humannerf(https://github.com/chungyiweng/humannerf), to keep the same view point
    ROT_CAM_PARAMS = {
        'zju_mocap': {'rotate_axis': 'z', 'inv_angle': True},
        'zju_mocap_1': {'rotate_axis': 'z', 'inv_angle': False},
        'monocap': {'rotate_axis': 'y', 'inv_angle': False},
        'wild': {'rotate_axis': 'y', 'inv_angle': False}
    }
    @torch.no_grad()
    def __init__(self, dataset_parms, train = False, device = torch.device('cuda:0')):
        super(MonoDataset_novel_view, self).__init__()

        self.dataset_parms = dataset_parms

        self.data_folder = join(dataset_parms.source_path, 'train')
        self.device = device
        self.gender = self.dataset_parms.smpl_gender

        self.train = train

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0

        self.no_mask = bool(self.dataset_parms.no_mask)

        self.src_type = dataset_parms.src_type

        if dataset_parms.train_stage == 1 or self.src_type in ["zju_mocap","zju_mocap_1","monocap"]:    # == "zju_mocap":
            print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))
        else:
            print('loading smpl data ', join(self.data_folder, 'smpl_parms_pred.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms_pred.pth'))
            # print('loading smpl data ', join(self.data_folder, 'smpl_parms_pred.pth'))
            # self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms_pred.pth'))

        self.data_length = len(os.listdir(join(self.data_folder, 'images')))
        if self.train:
            self.train_length = NUM_FRAMES_observ
        
        self.name_list = []
        for index, img in enumerate(sorted(os.listdir(join(self.data_folder, 'images')))):
            base_name = img.split('.')[0]
            self.name_list.append((index, base_name))
        
        self.image_fix = os.listdir(join(self.data_folder, 'images'))[0].split('.')[-1]
        
        if not dataset_parms.no_mask:
            self.mask_fix = os.listdir(join(self.data_folder, 'masks'))[0].split('.')[-1]

        print("total pose length", self.data_length )


        if dataset_parms.smpl_type == 'smplx':

            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
        else:
            self.pose_data = self.smpl_data['body_pose'][:self.data_length]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            # self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]

        if dataset_parms.cam_static:
            cam_path = join(self.data_folder, 'cam_parms.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            self.extr_npy = extr_npy
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        
        self.src_type = dataset_parms.src_type
        
    def __len__(self):
        if self.train:
            return self.train_length
        return self.data_length

    def __getitem__(self, index,):
        return self.getitem(index)
    

    def get_freeview_camera(self, frame_idx, total_frames, trans):
        E = rotate_camera_by_frame_idx(
                extrinsics= self.extr_npy, 
                frame_idx=frame_idx,
                period=total_frames,
                trans=trans,
                **self.ROT_CAM_PARAMS[self.src_type])
        return E

    @torch.no_grad()
    def getitem(self, index,):

        if self.train is not True:
            pose_idx = 0 #index  #self.fix_posindexe_idx
            _, name_idx = self.name_list[pose_idx]
            view_index = index
            #print(view_index)
        else:           
            pose_idx = np.random.randint(0,self.data_length)
            _, name_idx = self.name_list[pose_idx]
            view_index = np.random.randint(25,76)
            #view_index = random.choice([43,50,57])
            #view_index = np.random.randint(0,100)



        image_path = join(self.data_folder, 'images' ,name_idx + '.' + self.image_fix)
        if self.src_type in ['zju_mocap','monocap','zju_mocap_1']:
            cond_image_path = image_path
        else:
            cond_image_path = join(self.data_folder, 'cropped_images' ,name_idx + '.' + self.image_fix)

        

        if self.dataset_parms.train_stage in [2,3]:

            inp_posmap_path = self.data_folder + '/inp_map/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_parms.inp_posmap_size), str(pose_idx).zfill(8))
            inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_parms.inp_posmap_size)]

        Th = self.smpl_data['trans'][pose_idx].numpy()
        extr_npy =  self.get_freeview_camera(view_index, 100, Th)

        R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array([extr_npy[:3, 3]], np.float32)

        intrinsic = self.intrinsic

        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]

        pose_data = self.pose_data[pose_idx]
        transl_data = self.transl_data[pose_idx]

        image = Image.open(image_path)
        width, height = image.size

        azimuth = -360.0 * (view_index / 100.)
        if azimuth < -180.:
            azimuth +=360.


        if not self.dataset_parms.no_mask:
            mask_path = join(self.data_folder, 'masks', name_idx + '.' + self.mask_fix)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Find contours in the mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)

            # Assuming there is only one contour (the main object), find its bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            crop_inf = np.zeros((4))
            # Calculate the crop region around the center
            border = 60
            crop_size = max(w,h) +border
            

            crop_inf[0] = max(0, center_x - crop_size // 2)
            crop_inf[1] = max(0, center_y - crop_size // 2)
            crop_inf[2] = min(image.size[0], crop_inf[0] + crop_size)
            crop_inf[3] = min(image.size[1], crop_inf[1] + crop_size)

            crop_inf = torch.from_numpy(crop_inf)

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        data_item = dict()
        if self.dataset_parms.train_stage in [2,3]:
            data_item['inp_pos_map'] = inp_posmap.transpose(2,0,1)


        # data_item['K']= intrinsic
        # data_item['R']= self.R
        # data_item['T']= self.T
        if view_index >0:
            normal_path = join(self.data_folder, 'normal', 'back', name_idx + '.jpg')
            normal = Image.open(normal_path)
            resized_normal = torch.from_numpy(np.array(normal)) / 255.0
            resized_normal =  resized_normal.permute(2, 0, 1)
            original_normal = resized_normal.clamp(0.0, 1.0)
            data_item['original_normal'] = original_normal

        if self.src_type == 'zju_mocap':
            data_item['data_type']= "observ_zju"
        else:          
            data_item['data_type']= "observ"

        data_item['src_type'] = self.src_type
        
        data_item['view_index'] = view_index
        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx
        data_item['pose_data'] = pose_data
        data_item['transl_data'] = transl_data
        data_item['crop_inf'] = crop_inf
        data_item['polar']= 0.
        data_item['azimuth']= azimuth
        data_item['radius']= 0.
        if self.train:
            data_item['cond_image_path']= cond_image_path
        if self.dataset_parms.smpl_type == 'smplx':
            rest_pose = self.rest_pose_data[pose_idx]
            data_item['rest_pose'] = rest_pose
  
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=intrinsic, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        
        return data_item

            # inp_posmap_path = join(self.data_folder, 'query_posemap_{}_cano_smpl.npz'.format(self.dataset_parms.inp_posmap_size))
            # inp_posmap =np.load(inp_posmap_path)['posmap' + str(self.dataset_parms.inp_posmap_size)]

def get_camrot(campos, lookat=None,up = None, inv_camera=False):
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
        if up is None:
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
        if index == 100:
            campos = np.array([0., -4.75, 0.], dtype='float32')
            camrot = get_camrot(campos, 
                            lookat=np.array([0, y, 0.]),
                            up = np.array([0., 0., -1.]),
                            inv_camera=True)
        else:
            campos = np.array([x, y, z], dtype='float32')
            #index = 25
            angle = 2 * np.pi / total_frames * index
            axis = np.array([0., 1., 0.], dtype=np.float32)
            Rrel = cv2.Rodrigues(angle * axis)[0]
            campos = campos @ Rrel
            camrot = get_camrot(campos, 
                            lookat=np.array([0, y, 0.]),
                            inv_camera=True)
                            

        E = np.eye(4, dtype='float32')
        E[:3, :3] = camrot
        E[:3, 3] = -camrot.dot(campos)

        K = np.eye(3, dtype='float32')
        K[0, 0] = focal
        K[1, 1] = focal
        K[:2, 2] = img_size / 2.

        return K, E



NUM_FRAMES_tpose = 75
class MonoDataset_tpose_novel_view(Dataset):
    # these code derive from humannerf(https://github.com/chungyiweng/humannerf), to keep the same view point
    ROT_CAM_PARAMS = {
        'zju_mocap': {'rotate_axis': 'z', 'inv_angle': True},
        'wild': {'rotate_axis': 'y', 'inv_angle': False},
        'monocap': {'rotate_axis': 'y', 'inv_angle': False},
    }
    CAM_PARAMS = {
            'radius': 5.0, 'focal': 1250
        }

    @torch.no_grad()
    def __init__(self, dataset_parms, train = False,device = torch.device('cuda:0')):
        super(MonoDataset_tpose_novel_view, self).__init__()

        self.dataset_parms = dataset_parms
        self.train = train

        self.data_folder = join(dataset_parms.source_path, 'train')
        self.device = device
        self.gender = self.dataset_parms.smpl_gender

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0

        self.no_mask = bool(self.dataset_parms.no_mask)

        self.src_type = dataset_parms.src_type

        if dataset_parms.train_stage == 1 or self.src_type== "zju_mocap":
            print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))
        else:
            print('loading smpl data ', join(self.data_folder, 'smpl_parms_pred.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms_pred.pth'))

        self.data_length = len(os.listdir(join(self.data_folder, 'images')))
        if self.train:
            self.train_length =NUM_FRAMES_tpose
        else:
            self.train_length = 101
        self.name_list = []
        for index, img in enumerate(sorted(os.listdir(join(self.data_folder, 'images')))):
            base_name = img.split('.')[0]
            self.name_list.append((index, base_name))
        
        self.image_fix = os.listdir(join(self.data_folder, 'images'))[0].split('.')[-1]
        
        if not dataset_parms.no_mask:
            self.mask_fix = os.listdir(join(self.data_folder, 'masks'))[0].split('.')[-1]

        print("total pose length", self.data_length )


        if dataset_parms.smpl_type == 'smplx':

            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
        else:
            self.pose_data = self.smpl_data['body_pose'][:self.data_length]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            # self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]

        if dataset_parms.cam_static:
            cam_path = join(self.data_folder, 'cam_parms.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            extr_npy[2,3] = 5
            intr_npy = cam_npy['intrinsic']
            self.extr_npy = extr_npy
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        
        if self.train:
            self.folder_path = Path(dataset_parms.stage1_out_path).parts
            self.cond_image_path = join(self.folder_path[0],self.folder_path[1],"00000.png")
            print(f"loading image from {self.cond_image_path}")

        
    def __len__(self):
        return self.train_length

    def __getitem__(self, index,):
        return self.getitem(index)
    
    @torch.no_grad()
    def getitem(self, index,):
        #self.fix_posindexe_idx

        if self.train is not True:
            view_index = index
            pose_idx = 0
        # elif index % 3 ==0:
        #     left_indx = np.random.randint(20,30)
        #     right_indx = np.random.randint(70,80)
        #     index = np.random.choice([left_indx,right_indx], 1,
        #       p=[0.5,0.5]).item()
        else:
            pose_idx = np.random.randint(0,self.data_length)
            view_index = np.random.randint(25,76)
            if view_index > 75:
               view_index = 100 # 76 is from top 
            #view_index = random.choice([43,50,57])
            #view_index = np.random.randint(0,101)

        if self.dataset_parms.train_stage in [2,3]:
            # inp_posmap_path = join(self.data_folder, 'query_posemap_{}_cano_smpl.npz'.format(self.dataset_parms.inp_posmap_size))
            # inp_posmap =np.load(inp_posmap_path)['posmap' + str(self.dataset_parms.inp_posmap_size)]
            inp_posmap_path = self.data_folder + '/inp_map/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_parms.inp_posmap_size), str(pose_idx).zfill(8))
            inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_parms.inp_posmap_size)]


        leg_angle = 30
        pose_data = torch.zeros_like(self.pose_data[pose_idx])

        #print(pose_data.shape)
        # pose_data[5] =  leg_angle / 180 * math.pi
        # pose_data[8] = -leg_angle / 180 * math.pi
        #pose_data = self.pose_data[pose_idx]


        transl_data = torch.zeros((3))

       

        width = height = 512



        K, E = setup_camera(img_size = height,index=view_index, total_frames=100,
                                 **self.CAM_PARAMS)

        R = E[:3, :3]
        T = E[:3, 3]


        # azimuth = 360.0 * (view_index / self.data_length)
        # if azimuth > 180.:
        #     azimuth -=360.

        azimuth = -360.0 * (view_index / 100.)
        if azimuth < -180.:
            azimuth += 360.

 

        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        data_item = dict()
        if self.dataset_parms.train_stage in [2,3]:
            data_item['inp_pos_map'] = inp_posmap.transpose(2,0,1)
        
        data_item['K']= K
        data_item['R']= R
        data_item['T']= T

        if self.dataset_parms.train_stage == 3 and self.train is True:
            normal_path = join(self.folder_path[0],self.folder_path[1],"normal",f"{view_index}.jpg")
            normal = Image.open(normal_path)
            resized_normal = torch.from_numpy(np.array(normal)) / 255.0
            resized_normal =  resized_normal.permute(2, 0, 1)
            original_normal = resized_normal.clamp(0.0, 1.0)
            data_item['original_normal'] = original_normal


        data_item['src_type'] = self.src_type
        data_item['data_type']= "tpose"
        data_item['polar']= 0.
        data_item['azimuth']= azimuth
        data_item['radius']= 0.
        data_item['view_index'] = view_index
        if view_index ==100:
            data_item['polar']= -90.
            data_item['azimuth']= 0.
            data_item['radius']= 0.            

        if self.train:
            data_item['cond_image_path']= self.cond_image_path
        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx

        data_item['pose_data'] = pose_data
        data_item['transl_data'] = transl_data
        if self.dataset_parms.smpl_type == 'smplx':
            rest_pose = self.rest_pose_data[pose_idx]
            data_item['rest_pose'] = rest_pose
  
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=K, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        
        return data_item

            # inp_posmap_path = join(self.data_folder, 'query_posemap_{}_cano_smpl.npz'.format(self.dataset_parms.inp_posmap_size))
            # inp_posmap =np.load(inp_posmap_path)['posmap' + str(self.dataset_parms.inp_posmap_size)]
