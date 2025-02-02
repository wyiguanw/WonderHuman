import os
import torch
import lpips
import torchvision
import open3d as o3d
import sys
import uuid
from tqdm import tqdm
import torch.nn.functional as F
from utils.loss_utils import l1_loss_w, ssim
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, OptimizationParams, NetworkParams
from model.avatar_model import AvatarModel
from utils.general_utils import to_cuda, adjust_loss_weights

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *



guidance = {
    # "grad_normalize": True,
    # 'grad_normalize_scale': 0.9,
    'pretrained_model_name_or_path': "./pretrained/zero123/zero123-xl.ckpt", #"./pretrained/zero123/105000.ckpt"
    'pretrained_config': "./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml",
    'vram_O': True,
    # 'cond_image_path': "./exps/inb/inb_377/0000.png",
    'cond_elevation_deg': 0,
    'cond_azimuth_deg': 0,
    'cond_camera_distance': 0,
    'guidance_scale': 3.0,
    'min_step_percent': 0.02,
    # min_step_percent: [0, 0.4, 0.02, 2000]  # (start_iter, start_val, end_val, end_iter)
    'max_step_percent': 0.5
}


def train(model, net, opt, saving_epochs, checkpoint_epochs):
    tb_writer = prepare_output_and_logger(model)
    #if model
    avatarmodel = AvatarModel(model, net, opt, train=True)
    
    loss_fn_vgg = lpips.LPIPS(net='alex').cuda()
    train_loader = avatarmodel.getTrainDataloader()
    if avatarmodel.model_parms.train_stage ==3:
            guidance_model = threestudio.find("zero123-guidance")(guidance)
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    first_iter = 0
    epoch_start = 0
    data_length = len(train_loader)
    avatarmodel.training_setup()

    if checkpoint_epochs:
        avatarmodel.load(checkpoint_epochs[0])
        epoch_start += checkpoint_epochs[0]
        first_iter += epoch_start * data_length

    if model.train_stage ==3 :
        #avatarmodel.stage2_load(200)
        avatarmodel.stage_load(model.stage1_out_path)
    elif model.train_stage == 2:
        avatarmodel.stage_load(model.stage1_out_path)
    
    progress_bar = tqdm(range(first_iter, data_length * opt.epochs), desc="Training progress")
    ema_loss_for_log = 0.0


    print(opt.epochs)
    for epoch in range(epoch_start + 1, opt.epochs + 1):

        if model.train_stage ==1:
            avatarmodel.net.train()
            avatarmodel.pose.train()
            avatarmodel.transl.train()
        else:
            avatarmodel.net.train()
            avatarmodel.pose.eval()
            avatarmodel.transl.eval()
            avatarmodel.pose_encoder.train()
            if epoch > opt.epochs:
                avatarmodel.geo_feature.requires_grad = False
                print(avatarmodel.geo_feature.requires_grad)
        
        iter_start.record()

        wdecay_rgl = adjust_loss_weights(opt.lambda_rgl, epoch, mode='decay', start=epoch_start, every=20)

        AG = 8

        for num, batch_data in enumerate(train_loader):
            
            first_iter += 1
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))
            
            if model.train_stage ==1:
                gt_image = batch_data['original_image']


                gt_normal = batch_data['original_normal']
                image, normal, points, offset_loss, geo_loss, scale_loss = avatarmodel.train_stage1(batch_data, first_iter)
                
                if batch_data['src_type'][0]== 'wild':
                        
                    crop_inf = batch_data['crop_inf'][0].int()
                    pred_normal = normal[0].permute(1,2,0)
                    cropped_normal = pred_normal[crop_inf[1]:crop_inf[3], crop_inf[0]:crop_inf[2]].permute(2,0,1).unsqueeze(dim=0)
                    normal = F.interpolate(
                                cropped_normal, (512, 512), mode="bilinear", align_corners=True
                            )

                    # gt_image = gt_image[0].permute(1,2,0)
                    # cropped_gt_image = gt_image[crop_inf[1]:crop_inf[3], crop_inf[0]:crop_inf[2]].permute(2,0,1).unsqueeze(dim=0)
                    # gt_image = F.interpolate(
                    #             cropped_gt_image, (512, 512), mode="bilinear", align_corners=True
                    #         )
                    
                    # pred_image = image[0].permute(1,2,0)
                    # cropped_pred_image = pred_image[crop_inf[1]:crop_inf[3], crop_inf[0]:crop_inf[2]].permute(2,0,1).unsqueeze(dim=0)
                    # image = F.interpolate(
                    #             cropped_pred_image, (512, 512), mode="bilinear", align_corners=True
                    #         )

                        
                        
                        
                # torchvision.utils.save_image(normal, os.path.join(model.model_path, 'log', 'normal'+ ".png"))
                # torchvision.utils.save_image(gt_normal, os.path.join(model.model_path, 'log', 'norm_gt'+ ".png"))
                # torchvision.utils.save_image(image, os.path.join(model.model_path, 'log', 'image'+ ".png"))
                scale_loss = opt.lambda_scale  * scale_loss
                offset_loss = wdecay_rgl * offset_loss
                
                Ll1 = (1.0 - opt.lambda_dssim) * l1_loss_w(image, gt_image)
                ssim_loss = opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 

                Ll1_normal = (1.0 - opt.lambda_dssim) * l1_loss_w(normal, gt_normal)
                ssim_normal = opt.lambda_dssim * (1.0 - ssim(normal, gt_normal)) 

                loss = scale_loss + offset_loss + geo_loss + Ll1 + ssim_loss + Ll1_normal + ssim_normal

                if epoch > opt.lpips_start_iter:
                        vgg_loss = opt.lambda_lpips * loss_fn_vgg((image-0.5)*2, (gt_image- 0.5)*2).mean()
                        loss = loss + vgg_loss

                        
            elif model.train_stage ==2:
                gt_image = batch_data['original_image']
                image, points, pose_loss, offset_loss, = avatarmodel.train_stage2(batch_data, first_iter)
                
                offset_loss = wdecay_rgl * offset_loss
                
                Ll1 = (1.0 - opt.lambda_dssim) * l1_loss_w(image, gt_image)
                ssim_loss = opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 

                loss =  offset_loss + Ll1 + ssim_loss + pose_loss * 10

                if epoch > opt.lpips_start_iter:
                        vgg_loss = opt.lambda_lpips * loss_fn_vgg((image-0.5)*2, (gt_image- 0.5)*2).mean()
                        loss = loss + vgg_loss

            elif model.train_stage ==3:
                image,normal, points,  offset_loss, pose_loss,scale_loss= avatarmodel.train_stage3(batch_data, first_iter)
                
                flag = False
                if batch_data['data_type'][0] in ["tpose", "observ", "observ_zju"]:
                #if batch_data['data_type'][0] == "tpose" or batch_data['data_type'][0] == "observ":
                    #torchvision.utils.save_image(image, os.path.join(model.model_path, 'log', 'tpose'+ ".png"))
                    if epoch < opt.epochs*0.3:
                         lambda_sds = 0.3
                    elif epoch< opt.epochs* 0.6:
                        lambda_sds = 0.15
                    elif epoch< opt.epochs* 0.9:
                        lambda_sds = 0.1
                    else: 
                        lambda_sds = 0.0005
                        flag = True
                    #lambda_sds = 0.3
                    # if batch_data['data_type'][0]== 'tpose':
                    #     torchvision.utils.save_image(image, os.path.join(model.model_path, 'log', 'top_img'+ ".png"))
                    #     torchvision.utils.save_image(normal, os.path.join(model.model_path, 'log', 'top_normal'+ ".png"))

                    # if batch_data['data_type'][0] == "tpose" :
                    #     if batch_data['view_index'][0] in range(0,25) or batch_data['view_index'][0] in range(75,100) :
                    #         lambda_sds = 0.000001

                    #lambda_sds = 0.3
                    if batch_data['data_type'][0]== 'observ':
                        crop_inf = batch_data['crop_inf'][0].int()
                        image = image.squeeze().permute(1,2,0)
                        cropped_image = image[crop_inf[1]:crop_inf[3], crop_inf[0]:crop_inf[2]].permute(2,0,1).unsqueeze(dim=0)
                        image = F.interpolate(
                            cropped_image, (512, 512), mode="bilinear", align_corners=True
                        )
                        #lambda_sds = 0.1
                    polar = batch_data['polar'][0].to(guidance_model.device)
                    azimuth = batch_data['azimuth'][0].to(guidance_model.device)
                    radius = batch_data['radius'][0].to(guidance_model.device)
                    cond_image_path = batch_data['cond_image_path'][0]
                    rgb = image.permute(0,2,3,1).to(guidance_model.device)

                    offset_loss = wdecay_rgl * offset_loss
                   #torchvision.utils.save_image(image, os.path.join(model.model_path, 'log', 'test'+ ".png"))
                    
                    guidance_out = guidance_model(
                                                    rgb,
                                                    polar,
                                                    azimuth,
                                                    radius,
                                                    cond_image_path,
                                                    rgb_as_latents=False,              
                                                )
                # claforte: TODO: rename the loss_terms keys
                    sds_loss = guidance_out["loss_sds"].to(image.device)* lambda_sds
                    loss = sds_loss + offset_loss + pose_loss * 10
                
                elif batch_data['data_type'][0] == "GT":
                    gt_image = batch_data['original_image']
                    offset_loss = wdecay_rgl * offset_loss
                
                    Ll1 = (1.0 - opt.lambda_dssim) * l1_loss_w(image, gt_image)
                    ssim_loss = opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 

                    loss =  offset_loss + Ll1 + ssim_loss + pose_loss * 10

                    if epoch > opt.lpips_start_iter:
                        vgg_loss = opt.lambda_lpips * loss_fn_vgg((image-0.5)*2, (gt_image- 0.5)*2).mean()
                        loss = loss + vgg_loss

                scale_loss = opt.lambda_scale  * scale_loss


                gt_normal = batch_data['original_normal']  
                if batch_data['src_type'][0]== 'wild' and batch_data['data_type'][0] in ['observ','GT']:
                            
                        crop_inf = batch_data['crop_inf'][0].int()
                        pred_normal = normal[0].permute(1,2,0)
                        cropped_normal = pred_normal[crop_inf[1]:crop_inf[3], crop_inf[0]:crop_inf[2]].permute(2,0,1).unsqueeze(dim=0)
                        normal = F.interpolate(
                                    cropped_normal, (512, 512), mode="bilinear", align_corners=True
                                )
                
       
                Ll1_normal = (1.0 - opt.lambda_dssim) * l1_loss_w(normal, gt_normal)
                ssim_normal = opt.lambda_dssim * (1.0 - ssim(normal, gt_normal)) 
           
                if batch_data['data_type'][0] in ['observ','observ_zju']:
                     norm_weight = 0.0000001
                else:
                    norm_weight =2
                    # if flag==True:
                    #      norm_weight = 1
                #norm_weight = 0.0000001
                norm_loss = (Ll1_normal + ssim_normal) * norm_weight
                loss = loss + norm_loss +scale_loss
            avatarmodel.zero_grad(epoch)

            loss.backward(retain_graph=True)
            iter_end.record()

            avatarmodel.step(epoch)

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if first_iter % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)

                if (first_iter-1) % opt.log_iter == 0:
                    save_poitns = points.clone().detach().cpu().numpy()
                    for i in range(save_poitns.shape[0]):
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(save_poitns[i])
                        o3d.io.write_point_cloud(os.path.join(model.model_path, 'log',"pred_%d.ply" % i) , pcd)

                    torchvision.utils.save_image(image, os.path.join(model.model_path, 'log', '{0:05d}_pred'.format(first_iter) + ".png"))
                    torchvision.utils.save_image(normal, os.path.join(model.model_path, 'log', '{0:05d}_norm_pred'.format(first_iter) + ".png"))
                    if 'gt_image' in locals():
                        torchvision.utils.save_image(gt_image, os.path.join(model.model_path, 'log', '{0:05d}_gt'.format(first_iter) + ".png"))
                    if 'normal_orig' in locals():
                        torchvision.utils.save_image(normal_orig, os.path.join(model.model_path, 'log', '{0:05d}_norm_orig'.format(first_iter) + ".png"))
                    if 'gt_normal' in locals():
                        torchvision.utils.save_image(gt_normal, os.path.join(model.model_path, 'log', '{0:05d}_norm_gt'.format(first_iter) + ".png"))
                    
            if tb_writer:
                if 'Ll1' in locals():
                    tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), first_iter)
                if 'sds_loss' in locals():
                    tb_writer.add_scalar('train_loss_patches/sds_loss', sds_loss.item(), first_iter)
                if 'norm_loss' in locals():
                    tb_writer.add_scalar('train_loss_patches/norm_loss', norm_loss.item(), first_iter)
                tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), first_iter)
                if model.train_stage ==1:
                    tb_writer.add_scalar('train_loss_patches/scale_loss', scale_loss.item(), first_iter)
                tb_writer.add_scalar('train_loss_patches/offset_loss', offset_loss.item(), first_iter)
                # tb_writer.add_scalar('train_loss_patches/aiap_loss', aiap_loss.item(), first_iter)
                tb_writer.add_scalar('iter_time', iter_start.elapsed_time(iter_end), first_iter)
                if model.train_stage ==1:
                    tb_writer.add_scalar('train_loss_patches/geo_loss', geo_loss.item(), first_iter)
                else:
                    tb_writer.add_scalar('train_loss_patches/pose_loss', pose_loss.item(), first_iter)
                if epoch > opt.lpips_start_iter and 'vgg_loss' in locals():
                    tb_writer.add_scalar('train_loss_patches/vgg_loss', vgg_loss.item(), first_iter)

        if ((epoch > saving_epochs[0]) and epoch % model.save_epoch == 0) or epoch == saving_epochs[1]:
            print("\n[Epoch {}] Saving Model".format(epoch))
            avatarmodel.save(epoch)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    os.makedirs(os.path.join(args.model_path, 'log'), exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    np = NetworkParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[10])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_epochs.append(args.epochs)
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train(lp.extract(args), np.extract(args), op.extract(args), args.save_epochs, args.checkpoint_epochs)

    print("\nTraining complete.")
