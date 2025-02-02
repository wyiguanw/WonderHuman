import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from utils.general_utils import safe_state, to_cuda
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args, NetworkParams, OptimizationParams
from model.avatar_model import  AvatarModel
import torch.nn.functional as F
from utils.image_utils import depth_rendering
import cv2
import numpy as np
import trimesh


def render_sets(model, net, opt, epoch:int,pose:str,stage:int):
    with torch.no_grad():
        avatarmodel = AvatarModel(model, net, opt, train=False)
        avatarmodel.training_setup()

        if stage ==1:
            avatarmodel.stage1_load(epoch)
        else:
            avatarmodel.stage2_load(epoch)
        
        
        # novel_pose_dataset = avatarmodel.getNovelposeDataset()
 
        # novel_pose_dataset = avatarmodel.getNovelviewDataset(tpose=tpose)
        # novel_pose_dataset.data_length = 100

        
        
        #train_loader = avatarmodel.getTrainDataloader()
        if pose == 'tpose':
            render_path = os.path.join(avatarmodel.model_path, 'novel_tpose_pose', "ours_tpose_{}_0".format(epoch))
            novel_pose_dataset = avatarmodel.getNovelviewDataset(tpose=True)
            novel_pose_dataset.data_length = 100

        elif pose == 'observ':
            render_path = os.path.join(avatarmodel.model_path, 'novel_tpose_pose', "ours_observ_{}_0".format(epoch))
            novel_pose_dataset = avatarmodel.getNovelviewDataset()
            novel_pose_dataset.data_length = 100
        elif pose == 'test':
            render_path = os.path.join(avatarmodel.model_path, 'novel_tpose_pose', "ours_test_{}_14".format(epoch))
            novel_pose_dataset = avatarmodel.getTestDataset()
        else:
            novel_pose_dataset = avatarmodel.getNovelposeDataset()
            render_path = os.path.join(avatarmodel.model_path, 'novel_tpose_pose', "ours_novel_pose_{}_2".format(epoch))
        makedirs(render_path, exist_ok=True)

        novel_pose_loader = torch.utils.data.DataLoader(novel_pose_dataset,
                                            batch_size = 1,
                                            shuffle = False,
                                            num_workers = 4,)

        # depth_path = os.path.join(avatarmodel.model_path, 'novel_tpose_pose', 'masks')
        # makedirs(depth_path, exist_ok=True)

        print(render_path)
        for idx, batch_data in enumerate(tqdm(novel_pose_loader, desc="Rendering progress")):
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))

            if model.train_stage ==1:
                image, normal, points, _, _, _ = avatarmodel.train_stage1(batch_data, 59600)
                torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(normal, os.path.join(render_path,"norm_"+ '{0:05d}'.format(idx) + ".png"))
            
            else:
                image,normal, points,colors= avatarmodel.render_free_stage2(batch_data, 59400)#image, _, _, _,= avatarmodel.train_stage3(batch_data, 59600) #avatarmodel.render_free_stage2(batch_data, 59400)
                torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(normal, os.path.join(render_path,"norm_"+ '{0:05d}'.format(idx) + ".png"))
            # points = points[0].cpu().detach().numpy()[::25]
            #colors = colors.cpu().detach().numpy()#[::25]
            # scene = trimesh.Scene()
            # transf = np.eye(4,4)
            # for i , p in enumerate( points):
            #     transf[0][3] = p[0] 
            #     transf[1][3] = p[1]
            #     transf[2][3] = p[2]
            #     dot = trimesh.creation.uv_sphere(radius=0.007,count=[8,8],transform=transf)
            #     dot.visual.face_colors = colors[i]
            #     scene.add_geometry(dot)
            # scene.show()
            # rotation_matrix = np.array([
            #         [-1,  0,  0],
            #         [ 0,  1,  0],
            #         [ 0,  0, -1]
            #     ], dtype=np.float32)
            # rotated_points = np.dot(points, rotation_matrix.T)
            # visul_mesh = trimesh.PointCloud(points)#.show()

            # #vertex_normals = visul_mesh.vertex_normals 

            # normals = visul_mesh.kdtree.query(visul_mesh.vertices, k=4)[1]
            # vec = np.diff(points[normals], axis=1).mean(axis=1)
            # normals = np.cross(vec[:, 0], vec[:, 1])
            # normals /= np.linalg.norm(normals, axis=1).reshape(-1, 1)

            # vertex_colors = (normals + 1.0 ) * 0.5 * 255.0

            # visul_mesh.visual.vertex_colors = vertex_colors.astype(np.uint8)

            # visul_mesh.show()

            # break


            # if batch_data['data_type'][0]== 'observ':
            #     crop_inf = batch_data['crop_inf'][0].int()
            #     image = image.squeeze().permute(1,2,0)
            #     cropped_image = image[crop_inf[1]:crop_inf[3], crop_inf[0]:crop_inf[2]].permute(2,0,1).unsqueeze(dim=0)
            #     cropped_image = F.interpolate(
            #         cropped_image, (512, 512), mode="bilinear", align_corners=True
            #     )
                
            #     torchvision.utils.save_image(cropped_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            # else:
            #     torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            # torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            # torchvision.utils.save_image(normal, os.path.join(render_path,"norm_"+ '{0:05d}'.format(idx) + ".png"))
        # if pose == 'tpose':
        #     points = points[0].cpu().detach().numpy()[::50]# + np.array([0, 0.25, 0])
        #     visul_mesh = trimesh.PointCloud(points)
        #     visul_mesh.export(os.path.join(avatarmodel.model_path,"pcd_tpose.ply"))


            # colors = colors.cpu().detach().numpy()[::25]
            # scene = trimesh.Scene()
            # transf = np.eye(4,4)

            # for i , p in enumerate( points):
            #     transf[0][3] = p[0] 
            #     transf[1][3] = p[1]
            #     transf[2][3] = p[2]
            #     dot = trimesh.creation.uv_sphere(radius=0.007,count=[8,8],transform=transf)
            #     dot.visual.face_colors = colors[i]
            #     scene.add_geometry(dot)
            # scene.show()
            # visul_mesh = trimesh.PointCloud(points,colors=colors).show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    network = NetworkParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--pose", default=None, type=str)
    parser.add_argument("--stage", default=1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), network.extract(args), op.extract(args), args.epoch, args.pose, args.stage,)