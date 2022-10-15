from torch import save
from model import MRCF_test as MRCF
from model import LTE
from utils import flow_to_color
from dataset import dataloader
from utils import calc_psnr_and_ssim_cuda, bgr2ycbcr

import os
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import visdom
from imageio import imread, imsave, get_writer
from PIL import Image
import cv2
# from ptflops import get_model_complexity_info
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

def foveated_metric(LR, LR_fv, HR, mn, hw, crop, kernel_size, stride_size, eval_mode=False):
    m, n = mn
    h, w = hw
    crop_h, crop_w = crop
    HR_fold = F.unfold(HR.unsqueeze(0), kernel_size=(kernel_size, kernel_size), stride=stride_size)  # [N, 3*11*11, Hr*Wr]
    LR_fv_fold = F.unfold(LR_fv.unsqueeze(0), kernel_size=(kernel_size, kernel_size), stride=stride_size)  # [N, 3*11*11, Hr*Wr]
    B, C, N = HR_fold.size()
    HR_fold = HR_fold.permute(0, 2, 1).view(B*N , 3, kernel_size, kernel_size)        # [N, 3*11*11, Hr*Wr]
    LR_fv_fold = LR_fv_fold.permute(0, 2, 1).view(B*N , 3, kernel_size, kernel_size)
    Hr = (h - kernel_size) // stride_size + 1
    Wr = (w - kernel_size) // stride_size + 1

    B, C, H, W = HR_fold.size()
    mask = torch.ones((B, 1, H, W)).float()
    psnr_score, ssim_score = calc_psnr_and_ssim_cuda(HR_fold, LR_fv_fold, mask, is_tensor=False, batch_avg=True)
    psnr_score = psnr_score.view(Hr, Wr)
    ssim_score = ssim_score.view(Hr, Wr)

    # psnr_y_idx = (torch.argmax(psnr_score) // Wr) * stride_size
    # psnr_x_idx = (torch.argmax(psnr_score) %  Wr) * stride_size
    # ssim_y_idx = (torch.argmax(ssim_score) // Wr) * stride_size
    # ssim_x_idx = (torch.argmax(ssim_score) %  Wr) * stride_size
    if not eval_mode:
        HR[:, m:m+crop_h, n]          = torch.tensor([0., 0., 255.]).unsqueeze(1).repeat((1,crop_h))
        HR[:, m:m+crop_h, n+crop_w-1] = torch.tensor([0., 0., 255.]).unsqueeze(1).repeat((1,crop_h))
        HR[:, m,          n:n+crop_w] = torch.tensor([0., 0., 255.]).unsqueeze(1).repeat((1,crop_w))
        HR[:, m+crop_h-1, n:n+crop_w] = torch.tensor([0., 0., 255.]).unsqueeze(1).repeat((1,crop_w))

        LR_fv[:, m:m+crop_h, n]          = torch.tensor([0., 0., 255.]).unsqueeze(1).repeat((1,crop_h))
        LR_fv[:, m:m+crop_h, n+crop_w-1] = torch.tensor([0., 0., 255.]).unsqueeze(1).repeat((1,crop_h))
        LR_fv[:, m,          n:n+crop_w] = torch.tensor([0., 0., 255.]).unsqueeze(1).repeat((1,crop_w))
        LR_fv[:, m+crop_h-1, n:n+crop_w] = torch.tensor([0., 0., 255.]).unsqueeze(1).repeat((1,crop_w))

    psnr_min = psnr_score.min()
    psnr_max = psnr_score.max()
    ssim_min = ssim_score.min()
    ssim_max = ssim_score.max()
    # psnr_score = (psnr_score - psnr_min) / (psnr_max - psnr_min)
    # ssim_score = (ssim_score - ssim_min) / (ssim_max - ssim_min)
    psnr_score = psnr_score / 100
    ssim_score = (ssim_score.clip(0, 1) - 0.7) / 0.3

    # psnr_score_discrete = torch.zeros_like(psnr_score)
    # ssim_score_discrete = torch.zeros_like(ssim_score)

    # psnr_score_discrete[psnr_score <= 1.0] = 1.0
    # psnr_score_discrete[psnr_score <= 0.9] = 0.9
    # psnr_score_discrete[psnr_score <= 0.8] = 0.8
    # psnr_score_discrete[psnr_score <= 0.7] = 0.7
    # psnr_score_discrete[psnr_score <= 0.6] = 0.6
    # psnr_score_discrete[psnr_score <= 0.5] = 0.5
    # psnr_score_discrete[psnr_score <= 0.4] = 0.4
    # psnr_score_discrete[psnr_score <= 0.3] = 0.3
    # psnr_score_discrete[psnr_score <= 0.2] = 0.2
    # psnr_score_discrete[psnr_score <= 0.1] = 0.1

    # ssim_score_discrete[ssim_score <= 1.0] = 1.0
    # ssim_score_discrete[ssim_score <= 0.9] = 0.9
    # ssim_score_discrete[ssim_score <= 0.8] = 0.8
    # ssim_score_discrete[ssim_score <= 0.7] = 0.7
    # ssim_score_discrete[ssim_score <= 0.6] = 0.6
    # ssim_score_discrete[ssim_score <= 0.5] = 0.5
    # ssim_score_discrete[ssim_score <= 0.4] = 0.4
    # ssim_score_discrete[ssim_score <= 0.3] = 0.3
    # ssim_score_discrete[ssim_score <= 0.2] = 0.2
    # ssim_score_discrete[ssim_score <= 0.1] = 0.1

    # self.viz.viz.image(HR.cpu().numpy(), win='{}'.format('HR'), opts=dict(title='{}, Image size : {}'.format('HR', HR.size())))
    # self.viz.viz.image(LR.cpu().numpy(), win='{}'.format('LR'), opts=dict(title='{}, Image size : {}'.format('LR', LR.size())))
    # self.viz.viz.image(LR_fv.cpu().numpy(), win='{}'.format('FV'), opts=dict(title='{}, Image size : {}'.format('FV', LR_fv.size())))
    # self.viz.viz.image(psnr_score.cpu().numpy(), win='{}'.format('PSNR_score'), opts=dict(title='{}, Image size : {}'.format('PSNR_score', psnr_score.size())))
    # self.viz.viz.image(ssim_score.cpu().numpy(), win='{}'.format('SSIM_score'), opts=dict(title='{}, Image size : {}'.format('SSIM_score', ssim_score.size())))
    # self.viz.viz.image(psnr_score_discrete.cpu().numpy(), win='{}'.format('PSNR_score_discrete'), opts=dict(title='{}, Image size : {}'.format('PSNR_score_discrete', psnr_score_discrete.size())))
    # self.viz.viz.image(ssim_score_discrete.cpu().numpy(), win='{}'.format('SSIM_score_discrete'), opts=dict(title='{}, Image size : {}'.format('SSIM_score_discrete', ssim_score_discrete.size())))

    return psnr_score, ssim_score, (psnr_min, psnr_max), (ssim_min, ssim_max)

def rgb2yuv(rgb, y_only=True):
    # rgb_ = rgb.permute(0,2,3,1)
    # A = torch.tensor([[0.299, -0.14714119,0.61497538],
                    #   [0.587, -0.28886916, -0.51496512],
                    #   [0.114, 0.43601035, -0.10001026]])
    # yuv = torch.tensordot(rgb_,A,1).transpose(0,2)
    r = rgb[:, 0, :, :]
    g = rgb[:, 1, :, :]
    b = rgb[:, 2, :, :]

    y =  0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b
    v =  0.615 * r - 0.515 * g - 0.100 * b
    yuv = torch.stack([y,u,v], dim=1)
    if y_only:
        return y.unsqueeze(1)
    else:
        return yuv

def yuv2rgb(yuv):
    y = yuv[:, 0, :, :]
    u = yuv[:, 1, :, :]
    v = yuv[:, 2, :, :]

    r = y + 1.14 * v  # coefficient for g is 0
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u  # coefficient for b is 0
    rgb = torch.stack([r,g,b], 1)

    return rgb

if __name__ == '__main__':
    
    device = torch.device('cuda')
    viz = visdom.Visdom(server='140.113.212.214', port=8803, env='Gen_video')
    # fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    # out = cv2.VideoWriter('test_arcane_simple.mp4', fourcc, 30.0, (1080,  1920))

    dataset_name = 'REDS'
    # dataset_name = 'old_tree'
    # dataset_name = 'arcane'
    regional_dcn = False
    eval_mode = True
    model_code = 15
    model_epoch = 99
    y_only = False
    hr_dcn = True
    offset_prop = True
    split_ratio = 3
    sigma = 50
    dcn_size = 720
    model_name = 'FVSR_x8_simple_v{}_hrdcn_{}_offsetprop_{}_fnet{}'.format(model_code, 'y' if hr_dcn else 'n',
                                                                                       'y' if offset_prop else 'n', 
                                                                                       '_{}outof4'.format(4-split_ratio) if model_code == 18 else '')
    print('Current model name: {}, Epoch: {}'.format(model_name, model_epoch))
    video_num = [ 0, 11, 15, 20]
    # video_num = [ 0, 1, 6, 17]
    if eval_mode:
        fv_st_idx = [0, 0, 0, 0]
    else:
        fv_st_idx = [66, 30, 31,  0]
    # fv_st_idx = [100, 100, 100, 100]
    video_set = 'train'
    # video_set = 'val'
    model_path = 'train/REDS/{}/model/'.format(model_name)
    model_saves = os.listdir(model_path)
    model_save = [v for v in model_saves if '{:05d}'.format(model_epoch) in v]
    assert len(model_save) == 1
    model_save = model_save[0]
    model_name += '_gaussian'
    if eval_mode:
        save_dir = 'test_png/eval_video/'
    else:
        save_dir = 'test_png/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, model_name)):
        os.makedirs(os.path.join(save_dir, model_name))
    
    # model = MRCF.MRCF_CRA_x8(mid_channels=64, num_blocks=1, spynet_pretrained='pretrained_models/spynet_20210409-c6c1bd09.pth', device=device).to(device)
    if model_code == 13:
        model = MRCF.MRCF_simple_v13(mid_channels=32, y_only=y_only, hr_dcn=hr_dcn, offset_prop=offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
        # model = MRCF.MRCF_simple_v13_nodcn(mid_channels=32, y_only=y_only, hr_dcn=hr_dcn, offset_prop=offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
    elif model_code == 15:
        model = MRCF.MRCF_simple_v15(mid_channels=32, y_only=y_only, hr_dcn=hr_dcn, offset_prop=offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
    elif model_code == 18:
        model = MRCF.MRCF_simple_v18(mid_channels=32, y_only=y_only, hr_dcn=hr_dcn, offset_prop=offset_prop, split_ratio=split_ratio, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
        # model = MRCF.MRCF_simple_v18_cra(mid_channels=32, y_only=y_only, hr_dcn=hr_dcn, offset_prop=offset_prop, split_ratio=split_ratio, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
    elif model_code == 0:
        model = MRCF.MRCF_simple_v0(mid_channels=32, y_only=y_only, hr_dcn=hr_dcn, offset_prop=offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)

    model_state_dict = model.state_dict()
    model_state_dict_save = {k.replace('basic_', 'basic_module.'):v for k,v in torch.load(os.path.join(model_path, model_save)).items() if k.replace('basic_', 'basic_module.') in model_state_dict}
    # model_state_dict_save = {k.replace('module.',''):v for k,v in torch.load(model_path).items() if k.replace('module.','') in model_state_dict}
    # for k in model_state_dict.keys():
    #     print(k)
    # print('-----')
    # for k in model_state_dict_save.keys():
    #     print(k)
    # print(model_save)
    # for k,v in torch.load(model_path).items():
        # print(k)
    model_state_dict.update(model_state_dict_save)
    model.load_state_dict(model_state_dict, strict=True)

    psnr_whole_list = []
    ssim_whole_list = []
    psnr_outskirt_list = []
    ssim_outskirt_list = []
    psnr_past_list = []
    ssim_past_list = []
    psnr_fovea_list = []
    ssim_fovea_list = []

    for v_idx, v in enumerate(video_num):
        if dataset_name == 'REDS':
            GT_img_dir = '/DATA/REDS_sharp/{}/{}/{}_sharp/{:03d}/'.format(video_set, video_set, video_set, v)
            LR_img_dir = '/DATA/REDS_sharp_BI_x8/{}/{}/{}_sharp/{:03d}/'.format(video_set, video_set, video_set, v)
        else:
            GT_img_dir = '{}_x1'.format(dataset_name)
            LR_img_dir = '{}_x8'.format(dataset_name)
        print('Data location: {}'.format(GT_img_dir))

        lr_frames = []
        hr_frames = []
        GT_imgs = []
        LR_imgs = []
        LRSR_imgs = []
        GT_files = os.listdir(GT_img_dir)
        LR_files = os.listdir(LR_img_dir)
        GT_files = sorted(GT_files)
        LR_files = sorted(LR_files)
        for file in GT_files:
            img = cv2.imread(os.path.join(GT_img_dir, file))
            GT_imgs.append(img[:1072, :1920, :])
            # img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            # hr_frames.append(img)
        H_, W_, _ = GT_imgs[0].shape
        for file in LR_files:
            img = cv2.imread(os.path.join(LR_img_dir, file))
            LR_imgs.append(img[:134, :240, :])
            LRSR_imgs.append(np.array(PIL.Image.fromarray(img).resize((W_, H_), PIL.Image.BICUBIC)))
            # img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            # lr_frames.append(img)

        gen_frames = []
        gt_frames = []
        lr_frames = []
        psnr_score_list = []
        ssim_score_list = []
        psnr_score_bicubic_list = []
        ssim_score_bicubic_list = []
        # fv_size = 144
        fv_size = 96
        dx = 0
        dy = 0
        psnr_min = 1000
        psnr_max = 0
        ssim_min = 1000
        ssim_max = 0

        if dataset_name == 'arcane':
            st_x = 760
            st_y = 300
            ed_x = 1160
            ed_y = 500
        else:
            st_x = 360 + dx
            st_y = 300 + dy
            ed_x = 720 + dx
            ed_y = 500 + dy

        cur_x = ed_x
        cur_y = ed_y
        step_x = 20
        step_y = 0
        n_frames = 100

        bd_length = 10
        rg_w = dcn_size
        rg_h = dcn_size

        GT_imgs = GT_imgs[:n_frames]
        LR_imgs = LR_imgs[:n_frames]
        LRSR_imgs = LRSR_imgs[:n_frames]
        # with get_writer(os.path.join(save_dir,'test_{}_{}_{:03}_{}_bicubic.gif'.format(model_name, dataset_name, model_epoch, int(y_only))), mode="I", fps=7) as writer:
        #     for n in range(n_frames):
        #         writer.append_data(LRSR_imgs[n][:,:,::-1])
        # with get_writer(os.path.join(save_dir,'test_{}_{}_{:03}_{}_gt.gif'.format(model_name, dataset_name, model_epoch, int(y_only))), mode="I", fps=7) as writer:
        #     for n in range(n_frames):
        #         writer.append_data(GT_imgs[n][:,:,::-1])

        GT_imgs = np.stack(GT_imgs, axis=0)
        LR_imgs = np.stack(LR_imgs, axis=0)
        LRSR_imgs = np.stack(LRSR_imgs, axis=0)
        GT_imgs = GT_imgs.astype(np.float32) / 255.
        LR_imgs = LR_imgs.astype(np.float32) / 255.
        LRSR_imgs = LRSR_imgs.astype(np.float32) / 255.
        GT_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(GT_imgs, (0, 3, 1, 2)))).float()
        LR_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(LR_imgs, (0, 3, 1, 2)))).float()
        LRSR_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(LRSR_imgs, (0, 3, 1, 2)))).float()
        N, C, H, W = GT_imgs.size()

        kernel = np.array([ [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1] ], dtype=np.float32)
        kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)) # size: (1, 1, 3, 3)
        mk_list = []
        mk_one = torch.ones((1, 1, H, W)).to(device)
        x_array = sigma * np.random.randn(N) + (W / 2)
        y_array = sigma * np.random.randn(N) + (H / 2)
        white_paper = np.ones((H,W,3), np.uint8) * 255
        traj_list = []

        model.eval()
        with torch.no_grad():
            for n in range(N):
                print(n, '\r', end='')
                lr = LR_imgs[n:n+1].unsqueeze(0).to(device)
                lrsr = LRSR_imgs[n:n+1].unsqueeze(0).to(device).clone()
                gt = GT_imgs[n:n+1].unsqueeze(0).to(device).clone()
                fv = torch.zeros_like(gt).to(device)
                mk = torch.zeros((1, 1, 1, H, W)).to(device)
                fg = torch.zeros((1, 1, 1, H, W)).to(device)
                
                #### Raster scan
                # N_H = H // fv_size
                # N_W = W // fv_size
                # SP_H = H / N_H
                # SP_W = W / N_W
                # fv_sp = []
                # x_i = n % N_W
                # y_i = (n // N_W) % N_H
                # cur_y = int((1+y_i)*SP_H - (SP_H + fv_size)//2)
                # cur_x = int((1+x_i)*SP_W - (SP_W + fv_size)//2)
                
                #### Gaussian span
                cur_y = int(y_array[n]) - fv_size//2
                cur_x = int(x_array[n]) - fv_size//2
                traj_list.append((cur_y, cur_x))

                if n >= fv_st_idx[v_idx]:
                    fv[:, :, :, cur_y:cur_y+fv_size, cur_x:cur_x+fv_size] = gt[:, :, :, cur_y:cur_y+fv_size, cur_x:cur_x+fv_size]
                    mk[:, :, :, cur_y:cur_y+fv_size, cur_x:cur_x+fv_size] = 1

                mk_fv = mk.clone()
                mk_fv[:, :, :, cur_y:cur_y+fv_size, cur_x:cur_x+fv_size] = 1
                mk_out = mk_fv.clone().squeeze(0)
                for _ in range(10):
                    mk_out = torch.clamp(F.conv2d(mk_out, kernel_tensor.to(mk_out.device), padding=(1, 1)), 0, 1)
                mk_out = torch.logical_and(torch.logical_not(mk), mk_out)

                st_rg_x = max(cur_x+(fv_size//2)-(rg_w//2), 0)
                ed_rg_x = min(cur_x+(fv_size//2)+(rg_w//2), 1920)
                st_rg_y = max(cur_y+(fv_size//2)-(rg_h//2), 0)
                ed_rg_y = min(cur_y+(fv_size//2)+(rg_h//2), 1080)
                if regional_dcn:
                    fg[:, :, :, st_rg_y:ed_rg_y, st_rg_x:ed_rg_x] = 1
                else:
                    fg = torch.ones((1, 1, 1, H, W)).to(device)

                sr = model(lrs=lr, fvs=fv, mks=mk, fgs=fg)
                psnr, ssim = calc_psnr_and_ssim_cuda(sr.squeeze(0), gt.squeeze(0), mk_one)
                psnr_whole_list.append(psnr)
                ssim_whole_list.append(ssim)
                psnr, ssim = calc_psnr_and_ssim_cuda(sr.squeeze(0), gt.squeeze(0), mk_fv)
                psnr_fovea_list.append(psnr)
                ssim_fovea_list.append(ssim)
                psnr, ssim = calc_psnr_and_ssim_cuda(sr.squeeze(0), gt.squeeze(0), mk_out)
                psnr_outskirt_list.append(psnr)
                ssim_outskirt_list.append(ssim)
                if n > 0:
                    psnr, ssim = calc_psnr_and_ssim_cuda(sr.squeeze(0), gt.squeeze(0), mk_past)
                    psnr_past_list.append(psnr)
                    ssim_past_list.append(ssim)
                
                mk_list.append(mk_out.squeeze(0))
                if len(mk_list) > 3:
                    mk_list.pop(0)
                mk_past = torch.sum(torch.cat(mk_list, dim=1), dim=1, keepdim=True).clip(0, 1)

                psnr_score, ssim_score, psnr, ssim = foveated_metric(lr[0,0], sr[0,0], gt[0,0].clone(), (cur_y, cur_x), (H, W), (fv_size, fv_size), kernel_size=10, stride_size=5, eval_mode=eval_mode)
                psnr_score_list.append((psnr_score.unsqueeze(2).repeat(1, 1, 3) * 255).round().cpu().detach().numpy().astype(np.uint8))
                ssim_score_list.append((ssim_score.unsqueeze(2).repeat(1, 1, 3) * 255).round().cpu().detach().numpy().astype(np.uint8))
                psnr_score, ssim_score, psnr, ssim = foveated_metric(lr[0,0], lrsr[0,0], gt[0,0], (cur_y, cur_x), (H, W), (fv_size, fv_size), kernel_size=10, stride_size=5, eval_mode=eval_mode)
                if psnr[0] < psnr_min:
                    psnr_min = psnr[0]
                if psnr[1] > psnr_max:
                    psnr_max = psnr[1]
                if ssim[0] < ssim_min:
                    ssim_min = ssim[0]
                if ssim[1] > ssim_max:
                    ssim_max = ssim[1]
                psnr_score_bicubic_list.append((psnr_score.unsqueeze(2).repeat(1, 1, 3) * 255).round().cpu().detach().numpy().astype(np.uint8))
                ssim_score_bicubic_list.append((ssim_score.unsqueeze(2).repeat(1, 1, 3) * 255).round().cpu().detach().numpy().astype(np.uint8))

                if y_only:
                    B, N, C, H, W = lrsr.size()
                    lrsr = lrsr.view(B*N, C, H, W)
                    B, N, C, H, W = sr.size()
                    sr = sr.view(B*N, C, H, W)
                    lrsr = rgb2yuv(lrsr, y_only=False)
                    sr = yuv2rgb(torch.cat((sr[:,0:1,:,:], lrsr[:,1:3,:,:]), dim=1))
                
                sr = (sr * 255.).clip(0., 255.)
                lr = (lrsr * 255.).clip(0., 255.)
                gt = (gt * 255.).clip(0., 255.)
                sr = np.transpose(sr.squeeze().clone().detach().round().cpu().numpy().astype(np.uint8), (1, 2, 0))
                lr = np.transpose(lr.squeeze().clone().detach().round().cpu().numpy().astype(np.uint8), (1, 2, 0))
                gt = np.transpose(gt.squeeze().clone().detach().round().cpu().numpy().astype(np.uint8), (1, 2, 0))
                sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
                H, W, C = sr.shape
                # lr = cv2.resize(cv2.cvtColor(lr, cv2.COLOR_RGB2BGR), (W, H), cv2.INTER_CUBIC)
                lr = cv2.cvtColor(lr, cv2.COLOR_RGB2BGR)
                gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
                
                sr_copy = sr.copy()
                # cv2.rectangle(sr, (cur_x, cur_y), (cur_x+fv_size, cur_y+fv_size), (51, 51, 255), 3)
                # cv2.rectangle(sr, (st_rg_x, st_rg_y), (ed_rg_x, ed_rg_y), (255, 51, 51), 3)
                # cv2.line(sr, (cur_x+fv_size//2-5, cur_y+fv_size//2), (cur_x+fv_size//2+5, cur_y+fv_size//2), (51, 51, 255), 3)
                # cv2.line(sr, (cur_x+fv_size//2, cur_y+fv_size//2-5), (cur_x+fv_size//2, cur_y+fv_size//2+5), (51, 51, 255), 3)

                # sr[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x, :] = sr_copy[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x, :]
                # sr[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x-1, :] = sr_copy[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x-1, :]
                # sr[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+1, :] = sr_copy[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+1, :]
                # sr[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x-2, :] = sr_copy[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x-2, :]
                # sr[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+2, :] = sr_copy[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+2, :]

                # sr[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+fv_size, :] = sr_copy[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+fv_size, :]
                # sr[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+fv_size-1, :] = sr_copy[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+fv_size-1, :]
                # sr[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+fv_size+1, :] = sr_copy[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+fv_size+1, :]
                # sr[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+fv_size-2, :] = sr_copy[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+fv_size-2, :]
                # sr[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+fv_size+2, :] = sr_copy[cur_y+bd_length:cur_y+fv_size-bd_length, cur_x+fv_size+2, :]
                
                # sr[cur_y, cur_x+bd_length:cur_x+fv_size-bd_length, :] = sr_copy[cur_y, cur_x+bd_length:cur_x+fv_size-bd_length, :]
                # sr[cur_y-1, cur_x+bd_length:cur_x+fv_size-bd_length, :] = sr_copy[cur_y-1, cur_x+bd_length:cur_x+fv_size-bd_length, :]
                # sr[cur_y+1, cur_x+bd_length:cur_x+fv_size-bd_length, :] = sr_copy[cur_y+1, cur_x+bd_length:cur_x+fv_size-bd_length, :]
                # sr[cur_y-2, cur_x+bd_length:cur_x+fv_size-bd_length, :] = sr_copy[cur_y-2, cur_x+bd_length:cur_x+fv_size-bd_length, :]
                # sr[cur_y+2, cur_x+bd_length:cur_x+fv_size-bd_length, :] = sr_copy[cur_y+2, cur_x+bd_length:cur_x+fv_size-bd_length, :]
                
                # sr[cur_y+fv_size, cur_x+bd_length:cur_x+fv_size-bd_length, :] = sr_copy[cur_y+fv_size, cur_x+bd_length:cur_x+fv_size-bd_length, :]
                # sr[cur_y+fv_size-1, cur_x+bd_length:cur_x+fv_size-bd_length, :] = sr_copy[cur_y+fv_size-1, cur_x+bd_length:cur_x+fv_size-bd_length, :]
                # sr[cur_y+fv_size+1, cur_x+bd_length:cur_x+fv_size-bd_length, :] = sr_copy[cur_y+fv_size+1, cur_x+bd_length:cur_x+fv_size-bd_length, :]
                # sr[cur_y+fv_size-2, cur_x+bd_length:cur_x+fv_size-bd_length, :] = sr_copy[cur_y+fv_size-2, cur_x+bd_length:cur_x+fv_size-bd_length, :]
                # sr[cur_y+fv_size+2, cur_x+bd_length:cur_x+fv_size-bd_length, :] = sr_copy[cur_y+fv_size+2, cur_x+bd_length:cur_x+fv_size-bd_length, :]

                # sr[st_rg_y+bd_length:ed_rg_y-bd_length, st_rg_x-2:st_rg_x+3, :] = sr_copy[st_rg_y+bd_length:ed_rg_y-bd_length, st_rg_x-2:st_rg_x+3, :]
                # sr[st_rg_y+bd_length:ed_rg_y-bd_length, ed_rg_x-2:ed_rg_x+3, :] = sr_copy[st_rg_y+bd_length:ed_rg_y-bd_length, ed_rg_x-2:ed_rg_x+3, :]
                # sr[st_rg_y-2:st_rg_y+3, st_rg_x+bd_length:ed_rg_x-bd_length, :] = sr_copy[st_rg_y-2:st_rg_y+3, st_rg_x+bd_length:ed_rg_x-bd_length, :]
                # sr[ed_rg_y-2:ed_rg_y+3, st_rg_x+bd_length:ed_rg_x-bd_length, :] = sr_copy[ed_rg_y-2:ed_rg_y+3, st_rg_x+bd_length:ed_rg_x-bd_length, :]
                # cv2.rectangle(sr, (0, 100), (0+fv_size, 100+fv_size), (51, 51, 255), 3)
                # sr = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)
                gen_frames.append(sr.copy())
                lr_frames.append(lr.copy())
                gt_frames.append(gt.copy())
                cur_x += step_x
                cur_y += step_y
                # viz.image(sr.transpose(2, 0, 1), win='{}'.format('sr'), opts=dict(title='{}, Image size : {}'.format('sr', sr.shape)))
                # viz.image(lr.transpose(2, 0, 1), win='{}'.format('lr'), opts=dict(title='{}, Image size : {}'.format('lr', lr.shape)))
                # viz.image(gt.transpose(2, 0, 1), win='{}'.format('gt'), opts=dict(title='{}, Image size : {}'.format('gt', gt.shape)))

                if cur_x >= ed_x and cur_y <= st_y:
                    step_x = 0
                    step_y = 20
                elif cur_x >= ed_x and cur_y >= ed_y:
                    step_x = -20
                    step_y = 0
                elif cur_x <= st_x and cur_y >= ed_y:
                    step_x = 0
                    step_y = -20
                elif cur_x <= st_x and cur_y <= st_y:
                    step_x = 20
                    step_y = 0

        # for (idy, idx) in traj_list:
            # white_paper = cv2.circle(white_paper, (idx,idy), radius=10, color=(0, 0, 255), thickness=-1)

        model.clear_states()
        if dataset_name == 'REDS':
            #### Reconstructed results
            if not os.path.exists(os.path.join(save_dir, model_name, str(video_num[v_idx]), 'results')):
                os.makedirs(os.path.join(save_dir, model_name, str(video_num[v_idx]), 'results'))
            for i in range(len(gen_frames)):
                cv2.imwrite(os.path.join(save_dir, model_name, str(video_num[v_idx]), 'results', '{:03d}.png'.format(i)), gen_frames[i][:,:,::-1])
            with get_writer(os.path.join(save_dir, model_name, str(video_num[v_idx]), 'results', 'results.gif'), mode="I", fps=7) as writer:
                for n in range(len(gen_frames)):
                    writer.append_data(gen_frames[n])
            if not os.path.exists(os.path.join(save_dir, model_name, str(video_num[v_idx]), 'psnr')):
                os.makedirs(os.path.join(save_dir, model_name, str(video_num[v_idx]), 'psnr'))
            for i in range(len(gen_frames)):
                cv2.imwrite(os.path.join(save_dir, model_name, str(video_num[v_idx]), 'psnr', '{:03d}.png'.format(i)), psnr_score_list[i])
            if not os.path.exists(os.path.join(save_dir, model_name, str(video_num[v_idx]), 'ssim')):
                os.makedirs(os.path.join(save_dir, model_name, str(video_num[v_idx]), 'ssim'))
            for i in range(len(gen_frames)):
                cv2.imwrite(os.path.join(save_dir, model_name, str(video_num[v_idx]), 'ssim', '{:03d}.png'.format(i)), ssim_score_list[i])
            # cv2.imwrite(os.path.join(save_dir, model_name, str(video_num[v_idx]), 'traj.png'), white_paper)
            #### Bicubic upsample results
            if not os.path.exists(os.path.join(save_dir, 'Bicubic', str(video_num[v_idx]))):
                os.makedirs(os.path.join(save_dir, 'Bicubic', str(video_num[v_idx]), 'results'))
                os.makedirs(os.path.join(save_dir, 'Bicubic', str(video_num[v_idx]), 'psnr'))
                os.makedirs(os.path.join(save_dir, 'Bicubic', str(video_num[v_idx]), 'ssim'))
                for i in range(len(lr_frames)):
                    cv2.imwrite(os.path.join(save_dir, 'Bicubic', str(video_num[v_idx]), 'results', '{:03d}.png'.format(i)), lr_frames[i][:,:,::-1])
                for i in range(len(lr_frames)):
                    cv2.imwrite(os.path.join(save_dir, 'Bicubic', str(video_num[v_idx]), 'psnr', '{:03d}.png'.format(i)), psnr_score_bicubic_list[i])
                for i in range(len(lr_frames)):
                    cv2.imwrite(os.path.join(save_dir, 'Bicubic', str(video_num[v_idx]), 'ssim', '{:03d}.png'.format(i)), ssim_score_bicubic_list[i])
            #### GroundTruth
            if not os.path.exists(os.path.join(save_dir, 'GroundTruth', str(video_num[v_idx]))):
                os.makedirs(os.path.join(save_dir, 'GroundTruth', str(video_num[v_idx])))
                for i in range(len(lr_frames)):
                    cv2.imwrite(os.path.join(save_dir, 'GroundTruth', str(video_num[v_idx]), '{:03d}.png'.format(i)), gt_frames[i][:,:,::-1])
        else:
            if not os.path.exists(os.path.join(save_dir, model_name, dataset_name, 'results')):
                os.makedirs(os.path.join(save_dir, model_name, dataset_name, 'results'))
            for i in range(len(gen_frames)):
                cv2.imwrite(os.path.join(save_dir, model_name, dataset_name, 'results', '{:03d}.png'.format(i)), gen_frames[i][:,:,::-1])
            if not os.path.exists(os.path.join(save_dir, model_name, dataset_name, 'psnr')):
                os.makedirs(os.path.join(save_dir, model_name, dataset_name, 'psnr'))
            for i in range(len(gen_frames)):
                cv2.imwrite(os.path.join(save_dir, model_name, dataset_name, 'psnr', '{:03d}.png'.format(i)), psnr_score_list[i])
            if not os.path.exists(os.path.join(save_dir, model_name, dataset_name, 'ssim')):
                os.makedirs(os.path.join(save_dir, model_name, dataset_name, 'ssim'))
            for i in range(len(gen_frames)):
                cv2.imwrite(os.path.join(save_dir, model_name, dataset_name, 'ssim', '{:03d}.png'.format(i)), ssim_score_list[i])
            # cv2.imwrite(os.path.join(save_dir, model_name, dataset_name, 'traj.png'), white_paper)
            break

        print('PSNR_MIN: {}, PSNR_MAX: {}'.format(psnr_min, psnr_max))
        print('SSIM_MIN: {}, SSIM_MAX: {}'.format(ssim_min, ssim_max))
        # with get_writer(os.path.join(save_dir,'test_{}_{}_{:03}_{}.gif'.format(model_name, dataset_name, model_epoch, int(y_only))), mode="I", fps=7) as writer:
        #     for n in range(n_frames):
        #         # out.write(gen_frames[n][:,:,::-1])
        #         writer.append_data(gen_frames[n])
        
        # with get_writer('test_output_hr.gif', mode="I", fps=10) as writer:
        #     for n in range(n_frames):
        #         writer.append_data(hr_frames[n])

        # with get_writer('test_output_lr.gif', mode="I", fps=10) as writer:
        #     for n in range(n_frames):
        #         writer.append_data(lr_frames[n])

    print('PSNR_W: {}, SSIM_W: {}'.format(sum(psnr_whole_list)/len(psnr_whole_list), sum(ssim_whole_list)/len(ssim_whole_list)))
    print('PSNR_F: {}, SSIM_F: {}'.format(sum(psnr_fovea_list)/len(psnr_fovea_list), sum(ssim_fovea_list)/len(ssim_fovea_list)))
    print('PSNR_P: {}, SSIM_P: {}'.format(sum(psnr_past_list)/len(psnr_past_list), sum(ssim_past_list)/len(ssim_past_list)))
    print('PSNR_O: {}, SSIM_O: {}'.format(sum(psnr_outskirt_list)/len(psnr_outskirt_list), sum(ssim_outskirt_list)/len(ssim_outskirt_list)))
