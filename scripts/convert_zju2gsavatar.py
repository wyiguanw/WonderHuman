
import os
import torch
import shutil
import numpy as np
import numpy as np
from os.path import join
from PIL import Image
import cv2
import pickle

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()
    
def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat

def get_transform_params(smpl, params):
    """ obtain the transformation parameters for linear blend skinning
    """
    v_template = np.array(smpl['v_template'])

    # add shape blend shapes
    shapedirs = np.array(smpl['shapedirs'])
    betas = params['shapes']
    n = betas.shape[-1]
    m = shapedirs.shape[-1]
    n = min(m, n)
    v_shaped = v_template + np.sum(shapedirs[..., :n] * betas[None][..., :n], axis=2)

    # obtain the joints
    joints = smpl['J_regressor'].dot(v_shaped)

    return joints


def load_smpl_param(path, data_list, return_thata=True):
    #smpl_params = dict(np.load(str(path)))
    #smpl_params_right = dict(np.load(str(path_right)))

    R_list =[]
    T_list = []
    pose_list = []

    smpl_path = "GaussianAvatar/assets/smpl_files/smpl/SMPL_NEUTRAL.pkl"
    smpl = read_pickle(smpl_path)

    # smpl_betas_path = path.replace("smpl_params", "poses_optimized.npz")
    # smpl_params_betas = dict(np.load(str(smpl_betas_path)))

    for i in range(len(os.listdir(path))):
        smpl_params_path = join(path,f'{i+1}.npy')
        smpl_params = np.load(str(smpl_params_path),allow_pickle=True).item()
        
        Rh = smpl_params['Rh'][0]
        R_list.append(Rh)
        joints = np.expand_dims( get_transform_params(smpl,smpl_params),axis=0)
        Th = smpl_params['Th'][0].copy()
        # T[0] = -T[0]
        j0 = joints[:, 0, :]
        rot = batch_rodrigues(np.expand_dims(Rh.copy(),axis=0))
        Tnew = Th - j0 + np.einsum('bij,bj->bi', rot, j0)
        # T_list.append(T)
        T_list.append(Tnew)
        pose_list.append(smpl_params['poses'][0])


    theta = np.zeros((len(data_list), 72)).astype(np.float32)
    trans  = np.zeros((len(data_list), 3)).astype(np.float32)
    iter = 0
    for idx in data_list:
        theta[iter, :3] = R_list[idx].astype(np.float32)
        theta[iter, 3:] = pose_list[idx][3:].astype(np.float32)
        trans[iter, :] = T_list[idx].astype(np.float32)
        #print(theta[iter])

        iter +=1

    return {
        "beta": torch.from_numpy(smpl_params["shapes"]),
        #"beta": torch.from_numpy(smpl_params_betas["betas"].reshape(1,10).astype(np.float32)),
        "body_pose": torch.from_numpy(theta),
        "trans": torch.from_numpy(trans),
    }


# snap male 3 casual
# train_list = [0, 455, 4]
# test_list = [456, 675, 4]

# snap female 3 casual
# train_list = [0:445:4]
# test_list = 446:647:4]
snap = True

data_folder = 'GaussianAvatar/data'
subject = 'monocap_vlad'
view = 65
start = 0
end_train = 100
end_test = 150
interval = 1

all_image_path = join(data_folder, subject, 'images')
all_mask_apth = join(data_folder, subject, 'masks')



if snap:
    train_split_name = sorted(os.listdir(all_image_path))[start:end_train:interval]
    test_split_name = sorted(os.listdir(all_image_path))[end_train+1:end_test:interval]
    scene_length = len(os.listdir(all_image_path))
    train_list = list(range(scene_length))[start:end_train:interval]
    test_list = list(range(scene_length))[end_train+1:end_test:interval]

# the rule to split data is derived from InstantAvatar
else:
    scene_length = len(os.listdir(all_image_path))
    print('len:', scene_length)
    num_val = scene_length // 5
    length = int(1 / (num_val) * scene_length)
    offset = length // 2
    val_list = list(range(scene_length))[offset::length]
    train_list = list(set(range(scene_length)) - set(val_list))
    test_list = val_list[:len(val_list) // 2]
    val_list = val_list[len(val_list) // 2:]

    train_split_name = []
    test_split_name = []
    for idx,idx_name in enumerate(sorted(os.listdir(all_image_path))):
        if idx in train_list:
            train_split_name.append(idx_name)
        if idx in test_list:
            test_split_name.append(idx_name)


data_path = join(data_folder, subject)

out_path = join(data_path, 'train')
out_image_path =join(out_path, 'images')
out_mask_path =join(out_path, 'masks')

os.makedirs(out_image_path, exist_ok=True)
os.makedirs(out_mask_path, exist_ok=True)

test_path = join(data_path, 'test')
test_image_path =join(test_path, 'images')
test_mask_path =join(test_path, 'masks')

os.makedirs(test_image_path, exist_ok=True)
os.makedirs(test_mask_path, exist_ok=True)

# load camera

# camera_right = np.load(join("/GaussianAvatar/data/zju_377", "cameras.npz"))
# intrinsic_right = np.array(camera_right["intrinsic"])
# extrinsin_right = np.array(camera_right["extrinsic"])


camera = np.load(join(data_path, "annots.npy"),allow_pickle=True).item()
cam = camera['cams']
intrinsic = np.array(cam['K'][view])
R = np.array(cam['R'][view])
T = np.array(cam['T'][view]).reshape(3,) / 1000.

extrinsic = np.eye(4)
extrinsic[:3, :3] = R
extrinsic[:3, 3] = T

cam_all = {} 

cam_all['intrinsic'] = intrinsic
cam_all['extrinsic'] = extrinsic
np.savez(join(out_path, 'cam_parms.npz'), **cam_all)
np.savez(join(test_path, 'cam_parms.npz'), **cam_all)

train_smpl_params = load_smpl_param(join(data_path, "smpl_params"), train_list)

torch.save(train_smpl_params ,join(out_path, 'smpl_parms.pth'))

test_smpl_params = load_smpl_param(join(data_path, "smpl_params"), test_list)

torch.save(test_smpl_params ,join(test_path, 'smpl_parms.pth'))

assert len(os.listdir(all_image_path)) == len(os.listdir(all_mask_apth))



# List all files in the PNG directory
png_files = os.listdir(all_mask_apth)

# Iterate over each PNG file and convert it to JPG format
for png_file in png_files:
    if png_file.endswith('.png'):
        # Open the PNG image
        png_path = os.path.join(all_mask_apth, png_file)
        png_image = Image.open(png_path)
        
        # Remove the file extension (.png) and save as JPG (overwrite PNG)
        jpg_path = os.path.splitext(png_path)[0] + '.jpg'
        png_image.convert('RGB').save(jpg_path, 'JPEG')
        os.remove(png_path)


train_sum_dict = {}
for image_name in train_split_name:
    shutil.copy(join(all_image_path, image_name), join(out_image_path, image_name))
    shutil.copy(join(all_mask_apth, image_name), join(out_mask_path, image_name))

test_sum_dict = {}
for image_name in test_split_name:
    shutil.copy(join(all_image_path, image_name), join(test_image_path, image_name))
    shutil.copy(join(all_mask_apth, image_name), join(test_mask_path, image_name))
