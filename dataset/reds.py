
import os
import random
import pickle
import logging
import numpy as np
import PIL
import pdb
import math

import torch
import torch.utils.data as data
import torchvision.transforms.functional as F

logger = logging.getLogger('base')

def fovea_generator(GT_imgs, method='Rscan', step=0.1, FV_HW=(32, 32)):

    len_sp = len(GT_imgs)
    if torch.is_tensor(GT_imgs):
        _, GT_H, GT_W = GT_imgs[0].size()
    else:
        GT_H, GT_W, _ = GT_imgs[0].shape
    
    FV_H, FV_W = FV_HW
    if method == 'Cscan' or method == 'Zscan':
        SP = 0.1
        CP = 0.5
        EP = 0.9
    else:
        SP = 0.1
        CP = 0.5
        EP = 0.9
    
    #### shift according to center point
    CP_H = (GT_H*CP - FV_H//2)/GT_H
    CP_W = (GT_W*CP - FV_W//2)/GT_W
    EP_H = (GT_H*EP - FV_H)/GT_H
    EP_W = (GT_W*EP - FV_W)/GT_W
    
    #### finetune step size
    if method == 'Cscan' or method == 'Zscan':
        if SP + math.ceil(math.sqrt(len_sp)) * step > EP_H or SP + math.ceil(math.sqrt(len_sp)) * step > EP_W:
            step = min((EP_H - SP) / math.ceil(math.sqrt(len_sp)), (EP_W - SP) / math.ceil(math.sqrt(len_sp)))
        SP = int(SP * 100)
        step = int(step * 100)
        EP = int(SP + math.ceil(math.sqrt(len_sp) - 1) * step)
    elif method == 'Hscan':
        if SP + len_sp * step > EP_W:
            step = (EP_W - SP) / len_sp
        SP = int(SP * 100)
        step = int(step * 100)
        EP = int((SP + len_sp * step))
    elif method == 'Vscan':
        if SP + len_sp * step > EP_H:
            step = (EP_H - SP) / len_sp
        SP = int(SP * 100)
        step = int(step * 100)
        EP = int((SP + len_sp * step))
    else:
        if SP + len_sp * step > EP_H or SP + len_sp * step > EP_W:
            step = min((EP_H - SP) / len_sp, (EP_W - SP) / len_sp)
        SP = int(SP * 100)
        step = int(step * 100)
        EP = int((SP + len_sp * step))

    #### fovea scan simulation
    if method == 'Hscan':
        fv_sp = [[int(CP_H * GT_H), int((v / 100) * GT_W)] for v in [*range(SP, EP, step)]]
    elif method == 'Vscan':
        fv_sp = [[int((v / 100) * GT_H), int(CP_W * GT_W)] for v in [*range(SP, EP, step)]]
    elif method == 'DRscan': # Not done
        fv_sp = [[int((v / 100) * GT_H), int((v / 100) * GT_W)] for v in [*range(SP, EP, step)]]
    elif method == 'DLscan': # Not done
        fv_sp = [[int((v / 100) * GT_H), int((v / 100) * GT_W)] for v in [*range(SP, EP, step)]]
    elif method == 'Cscan':
        fv_sp = []
        v, h = (SP, SP)
        v_step, h_step = (step, step)
        for t in range(len_sp):
            fv_sp.append([int((v / 100) * GT_H), int((h / 100) * GT_W)])
            if h == EP and h_step > 0:
                h_step = -h_step
                v += v_step
            elif h == SP and h_step < 0:
                h_step = -h_step
                v += v_step
            else:
                h += h_step
    elif method == 'Zscan':
        fv_sp = []
        v, h = (SP, SP)
        v_step, h_step = (step, step)
        for t in range(len_sp):
            fv_sp.append([int((v / 100) * GT_H), int((h / 100) * GT_W)])
            if h == EP and v_step < 0:
                v_step = -v_step
                v += v_step
                h_step = -abs(h_step)
            elif v == SP and h_step > 0:
                h += h_step
                h_step = -h_step
                v_step = abs(v_step)
            elif v == EP and h_step < 0:
                h_step = -h_step
                h += h_step
                v_step = -abs(v_step)
            elif h == SP and v_step > 0:
                v += v_step
                v_step = -v_step
                h_step = abs(h_step)
            else:
                h += h_step
                v += v_step
    elif method == 'Rscan':
        sigma = 0.05
        rand_h = np.random.normal(CP_H, sigma, len_sp).clip(0, EP_H)
        rand_w = np.random.normal(CP_W, sigma, len_sp).clip(0, EP_W)
        fv_sp = [[int(rh * GT_H), int(rw * GT_W)] for rh, rw in zip(rand_h, rand_w)]
    elif method == 'Nanascan': # Not done
        # SP_H = 0
        # EP_H = (GT_H - FV_H - 1) / GT_H
        # Q1_H = (0.25 - (FV_H / GT_H)/2) if (0.25 - (FV_H / GT_H)/2) > 0 else SP_H
        # Q2_H = (0.50 - (FV_H / GT_H)/2)
        # Q3_H = (0.75 - (FV_H / GT_H)/2) if (0.75 + (FV_H / GT_H)/2) < 1 else EP_H
        # T1_H = (0.33 - (FV_H / GT_H)/2) if (0.33 - (FV_H / GT_H)/2) > 0 else SP_H
        # T2_H = (0.66 - (FV_H / GT_H)/2) if (0.66 + (FV_H / GT_H)/2) < 1 else EP_H

        ratio_H = FV_H / GT_H
        SP_H = 0 + (ratio_H / 2)
        EP_H = 1 - (ratio_H / 2)
        Q1_H = SP_H + ((EP_H - SP_H) * 0.25)
        Q2_H = SP_H + ((EP_H - SP_H) * 0.50)
        Q3_H = SP_H + ((EP_H - SP_H) * 0.75)
        T1_H = SP_H + ((EP_H - SP_H) * 0.33)
        T2_H = SP_H + ((EP_H - SP_H) * 0.66)

        ratio_W = FV_W / GT_W
        SP_W = 0 + (ratio_W / 2)
        EP_W = 1 - (ratio_W / 2)
        Q1_W = SP_W + ((EP_W - SP_W) * 0.25)
        Q2_W = SP_W + ((EP_W - SP_W) * 0.50)
        Q3_W = SP_W + ((EP_W - SP_W) * 0.75)
        T1_W = SP_W + ((EP_W - SP_W) * 0.33)
        T2_W = SP_W + ((EP_W - SP_W) * 0.66)

        locs = [[SP_H, SP_W], [SP_H, T1_W], [SP_H, T2_W], [SP_H, EP_W],
                [T1_H, SP_W], [T1_H, T1_W], [T1_H, T2_W], [T1_H, EP_W],
                [T2_H, SP_W], [T2_H, T1_W], [T2_H, T2_W], [T2_H, EP_W], 
                [EP_H, SP_W], [EP_H, T1_W], [EP_H, T2_W], [EP_H, EP_W]]
        locs = [(y - (ratio_H / 2), x - (ratio_H / 2)) for y, x in locs]
        # locs = [[Q1_H, T1_W], [Q1_H, T2_W], [Q2_H, Q1_W], [Q2_H, Q2_W], [Q2_H, Q3_W], [Q3_H, T1_W], [Q3_H, T2_W]]
        # locs = [[Q2_H, Q2_W]]

        fv_sp = random.choices(locs, k=len_sp)
        fv_sp = [[min(int(v[0] * GT_H), GT_H-FV_H), min(int(v[1] * GT_W), GT_W-FV_W)] for v in fv_sp]
        random.shuffle(fv_sp)
    elif method == 'Evenscan': # Not done
        idx = 20
        N_H = GT_H // FV_H
        N_W = GT_W // FV_W
        SP_H = GT_H / N_H
        SP_W = GT_W / N_W
        fv_sp = []
        for i in range(idx, idx + len_sp):
            x_i = i % N_W
            y_i = (i // N_W) % N_H
            fv_sp.append([int((1+y_i)*SP_H - (SP_H + FV_H)/2), int((1+x_i)*SP_W - (SP_W + FV_W)/2)])
    elif method == 'DemoHscan': # Not done
        SP_H = 0
        EP_H = 1
        SP_W = 0
        EP_W = 1
        fv_sp = []
        direction = -1
        scan_step = 8
        accm_step = GT_W - scan_step
        for _ in range(len_sp):
            fv_sp.append([0, accm_step])
            accm_step += direction * scan_step
            if accm_step < 0:
                direction *= -1
                accm_step += direction * scan_step
            elif accm_step >= GT_W:
                direction *= -1
                accm_step += direction * scan_step
    else:
        fv_sp = [[int((v / 100) * GT_H), int((v / 100) * GT_W)] for v in [*range(SP, EP, step)]]

    fv_sp = torch.tensor(fv_sp)
    if torch.is_tensor(GT_imgs):
        FV_imgs = []
        Ref_sps = []
        for t in range(len(GT_imgs)):
            #### With padding ####
            Ref_sp = torch.zeros_like(GT_imgs[t])
            if method == 'DemoHscan':
                Ref_sp[:, fv_sp[t][0]:, fv_sp[t][1]:] = 1
            else:
                Ref_sp[:, fv_sp[t][0]:fv_sp[t][0] + FV_H, fv_sp[t][1]:fv_sp[t][1] + FV_W] = 1
            Ref = GT_imgs[t] * Ref_sp
            FV_imgs.append(Ref)
            Ref_sps.append(Ref_sp)
            #### Without padding ####
            # FV_imgs.append(GT_imgs[t][:, fv_sp[t][0]:fv_sp[t][0] + FV_H, fv_sp[t][1]:fv_sp[t][1] + FV_W].clone())
        # FV_imgs = torch.stack(FV_imgs, dim=0)
    else:
        FV_imgs = []
        Ref_sps = []
        for t in range(len(GT_imgs)):
            #### With padding ####
            # Ref_sp = np.zeros_like(GT_imgs[t])
            H, W, C = GT_imgs[t].shape
            Ref_sp = np.zeros((H, W, 1))
            if method == 'DemoHscan':
                Ref_sp[fv_sp[t][0]:, fv_sp[t][1]:, :] = 1
            else:
                Ref_sp[fv_sp[t][0]:fv_sp[t][0] + FV_H, fv_sp[t][1]:fv_sp[t][1] + FV_W, :] = 1
            Ref = GT_imgs[t] * Ref_sp
            FV_imgs.append(Ref)
            Ref_sps.append(Ref_sp)
            #### Without padding ####
            # FV_imgs.append(GT_imgs[t][fv_sp[t][0]:fv_sp[t][0] + FV_H, fv_sp[t][1]:fv_sp[t][1] + FV_W, :].copy())
        # FV_imgs = np.stack(FV_imgs, dim=0)

    return FV_imgs, Ref_sps, fv_sp

class TrainSet(data.Dataset):
    '''
    Reading the training Vimeo dataset
    key example: train/00001/0001/im1.png
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution frames
    support reading N HR frames, N = 3, 5, 7
    '''
    def __init__(self, args):
        super(TrainSet, self).__init__()
        self.args = args

        scale = self.args.scale
        if scale == 8:
            LR_root = args.dataset_dir.replace('_sharp', '_sharp_BI_x8')
        elif scale == 4:
            LR_root = args.dataset_dir.replace('_sharp', '_sharp_BI')
        self.GT_dir_list = sorted([os.path.join(args.dataset_dir, 'train/train/train_sharp', name) for name in 
            os.listdir(os.path.join(args.dataset_dir, 'train/train/train_sharp')) if name not in ['000', '011', '015', '020']]) + \
        sorted([os.path.join(args.dataset_dir, 'val/val/val_sharp', name) for name in 
            os.listdir(os.path.join(args.dataset_dir, 'val/val/val_sharp')) if name not in ['000', '001', '006', '017']])
        self.LR_dir_list = sorted([os.path.join(LR_root, 'train/train/train_sharp', name) for name in 
            os.listdir(os.path.join(LR_root, 'train/train/train_sharp')) if name not in ['000', '011', '015', '020']]) + \
        sorted([os.path.join(LR_root, 'val/val/val_sharp', name) for name in 
            os.listdir(os.path.join(LR_root, 'val/val/val_sharp')) if name not in ['000', '001', '006', '017']])
        N_frames = self.args.N_frames
        self.GT_imgfiles = []
        self.LR_imgfiles = []
        for idx in range(len(self.GT_dir_list)):
            GT_imgfiles_cur = sorted(os.listdir(self.GT_dir_list[idx]))
            for img_idx in range(0, len(GT_imgfiles_cur) - N_frames + 1):
                self.GT_imgfiles.append([os.path.join(self.GT_dir_list[idx], img_f) for img_f in GT_imgfiles_cur[img_idx:img_idx + N_frames]])
        for idx in range(len(self.LR_dir_list)):
            LR_imgfiles_cur = sorted(os.listdir(self.LR_dir_list[idx]))
            for img_idx in range(0, len(LR_imgfiles_cur) - N_frames + 1):
                self.LR_imgfiles.append([os.path.join(self.LR_dir_list[idx], img_f) for img_f in LR_imgfiles_cur[img_idx:img_idx + N_frames]])

    def __getitem__(self, index):
        #### Configs
        scale = self.args.scale
        GT_size = self.args.GT_size
        LR_size = GT_size // scale
        FV_size = self.args.FV_size
        
        ### GT 
        GT_imgfiles = self.GT_imgfiles[index]
        GT_imgs = [np.array(PIL.Image.open(img)) for img in GT_imgfiles]

        #### Bicubic downsampling
        ### LR and LR_sr
        H_, W_, _ = GT_imgs[0].shape
        LR_imgfiles = self.LR_imgfiles[index]
        LR_imgs = [np.array(PIL.Image.open(img)) for img in LR_imgfiles]
        LR_sr_imgs = [np.array(PIL.Image.fromarray(img).resize((W_, H_), PIL.Image.BICUBIC)) for img in LR_imgs]

        ### Random cropping
        H, W, C = LR_imgs[0].shape
        rnd_h = random.randint(0, max(0, H - LR_size))
        rnd_w = random.randint(0, max(0, W - LR_size))
        LR_imgs = [v[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :] for v in LR_imgs]

        rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
        GT_imgs = [v[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :] for v in GT_imgs]
        LR_sr_imgs = [v[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :] for v in LR_sr_imgs]

        ### Ref(FV) and Ref_sr
        Ref, Ref_sp, _ = fovea_generator(GT_imgs, method='Nanascan', FV_HW=(FV_size, FV_size))

        #### Stacking
        GT_imgs = np.stack(GT_imgs, axis=0)
        LR_imgs = np.stack(LR_imgs, axis=0)
        LR_sr_imgs = np.stack(LR_sr_imgs, axis=0)
        Ref = np.stack(Ref, axis=0)
        Ref_sp = np.stack(Ref_sp, axis=0)

        #### Scaling
        GT_imgs = GT_imgs.astype(np.float32) / 255.
        LR_imgs = LR_imgs.astype(np.float32) / 255.
        LR_sr_imgs = LR_sr_imgs.astype(np.float32) / 255.
        Ref = Ref.astype(np.float32) / 255.
        Ref_sp = Ref_sp.astype(np.bool_)

        GT_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(GT_imgs, (0, 3, 1, 2)))).float()
        LR_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(LR_imgs, (0, 3, 1, 2)))).float()
        LR_sr_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(LR_sr_imgs, (0, 3, 1, 2)))).float()
        Ref = torch.from_numpy(np.ascontiguousarray(np.transpose(Ref, (0, 3, 1, 2)))).float()
        Ref_sp = torch.from_numpy(np.ascontiguousarray(np.transpose(Ref_sp, (0, 3, 1, 2))))

        if torch.rand(1) < 0.5:
            GT_imgs = F.hflip(GT_imgs)
            LR_imgs = F.hflip(LR_imgs)
            LR_sr_imgs = F.hflip(LR_sr_imgs)
            Ref = F.hflip(Ref)
            Ref_sp = F.hflip(Ref_sp)

        if torch.rand(1) < 0.5:
            GT_imgs = F.vflip(GT_imgs)
            LR_imgs = F.vflip(LR_imgs)
            LR_sr_imgs = F.vflip(LR_sr_imgs)
            Ref = F.vflip(Ref)
            Ref_sp = F.vflip(Ref_sp)

        return {'LR': LR_imgs, 
                'LR_sr': LR_sr_imgs,
                'HR': GT_imgs,
                'Ref': Ref,
                'Ref_sp': Ref_sp}

    def __len__(self):
        return len(self.GT_imgfiles)

class EvalSet(data.Dataset):
    '''
    Reading the training Vimeo dataset
    key example: train/00001/0001/im1.png
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution frames
    support reading N HR frames, N = 3, 5, 7
    '''
    def __init__(self, args):
        super(EvalSet, self).__init__()
        self.args = args

        scale = self.args.scale
        if scale == 8:
            LR_root = args.dataset_dir.replace('_sharp', '_sharp_BI_x8')
        elif scale == 4:
            LR_root = args.dataset_dir.replace('_sharp', '_sharp_BI')
        self.GT_dir_list = sorted([os.path.join(args.dataset_dir, 'val/val/val_sharp', name) for name in
            ['000', '001', '006', '017']])
        self.LR_dir_list = sorted([os.path.join(LR_root, 'val/val/val_sharp', name) for name in 
            ['000', '001', '006', '017']])
        # self.GT_dir_list = sorted([os.path.join(args.dataset_dir, 'train/train/train_sharp', name) for name in
            # ['000', '011', '015', '020']])
        # self.LR_dir_list = sorted([os.path.join(LR_root, 'train/train/train_sharp', name) for name in 
            # ['000', '011', '015', '020']])
        N_frames = self.args.N_frames

        self.GT_imgfiles = []
        self.LR_imgfiles = []
        for idx in range(len(self.GT_dir_list)):
            GT_imgfiles_cur = sorted(os.listdir(self.GT_dir_list[idx]))
            for img_idx in range(0, len(GT_imgfiles_cur) - N_frames + 1):
                self.GT_imgfiles.append([os.path.join(self.GT_dir_list[idx], img_f) for img_f in GT_imgfiles_cur[img_idx:img_idx + N_frames]])
        for idx in range(len(self.LR_dir_list)):
            LR_imgfiles_cur = sorted(os.listdir(self.LR_dir_list[idx]))
            for img_idx in range(0, len(LR_imgfiles_cur) - N_frames + 1):
                self.LR_imgfiles.append([os.path.join(self.LR_dir_list[idx], img_f) for img_f in LR_imgfiles_cur[img_idx:img_idx + N_frames]])

    def __getitem__(self, index):
        #### Configs
        scale = self.args.scale
        GT_size = self.args.GT_size
        LR_size = GT_size // scale
        FV_size = self.args.FV_size

        ### GT 
        GT_imgfiles = self.GT_imgfiles[index]
        GT_imgs = [np.array(PIL.Image.open(img)) for img in GT_imgfiles]

        #### Bicubic downsampling
        ### LR and LR_sr
        H_, W_, _ = GT_imgs[0].shape
        LR_imgfiles = self.LR_imgfiles[index]
        LR_imgs = [np.array(PIL.Image.open(img)) for img in LR_imgfiles]
        # LR_imgs = [np.array(PIL.Image.fromarray(img).resize((W_ // 8, H_ // 8), PIL.Image.BILINEAR)) for img in GT_imgs]
        LR_sr_imgs = [np.array(PIL.Image.fromarray(img).resize((W_, H_), PIL.Image.BICUBIC)) for img in LR_imgs]
        
        ### Ref(FV) and Ref_sr
        Ref, Ref_sp, fv_sp = fovea_generator(GT_imgs, method='Evenscan', FV_HW=(FV_size, FV_size))

        #### Stacking
        GT_imgs = np.stack(GT_imgs, axis=0)
        LR_imgs = np.stack(LR_imgs, axis=0)
        LR_sr_imgs = np.stack(LR_sr_imgs, axis=0)
        Ref = np.stack(Ref, axis=0)
        Ref_sp = np.stack(Ref_sp, axis=0)

        #### Scaling
        GT_imgs = GT_imgs.astype(np.float32) / 255.
        LR_imgs = LR_imgs.astype(np.float32) / 255.
        LR_sr_imgs = LR_sr_imgs.astype(np.float32) / 255.
        Ref = Ref.astype(np.float32) / 255.
        Ref_sp = Ref_sp.astype(np.bool_)

        GT_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(GT_imgs, (0, 3, 1, 2)))).float()
        LR_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(LR_imgs, (0, 3, 1, 2)))).float()
        LR_sr_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(LR_sr_imgs, (0, 3, 1, 2)))).float()
        Ref = torch.from_numpy(np.ascontiguousarray(np.transpose(Ref, (0, 3, 1, 2)))).float()
        Ref_sp = torch.from_numpy(np.ascontiguousarray(np.transpose(Ref_sp, (0, 3, 1, 2))))

        return {'LR': LR_imgs, 
                'LR_sr': LR_sr_imgs,
                'HR': GT_imgs,
                'Ref': Ref,
                'Ref_sp': Ref_sp,
                'FV_sp': fv_sp}
                
    def __len__(self):
        return len(self.GT_imgfiles)

class TestSet(data.Dataset):
    '''
    Reading the training Vimeo dataset
    key example: train/00001/0001/im1.png
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution frames
    support reading N HR frames, N = 3, 5, 7
    '''
    def __init__(self, args):
        super(TestSet, self).__init__()
        self.args = args
        scale = self.args.scale
        if scale == 8:
            LR_root = args.dataset_dir.replace('_sharp', '_sharp_BI_x8')
        elif scale == 4:
            LR_root = args.dataset_dir.replace('_sharp', '_sharp_BI')

        self.GT_dir_list = sorted([os.path.join(args.dataset_dir, 'train/train/train_sharp', name) for name in
            ['000', '011', '015', '020']])
        self.LR_dir_list = sorted([os.path.join(LR_root, 'train/train/train_sharp', name) for name in 
            ['000', '011', '015', '020']])

        N_frames = self.args.N_frames

        self.GT_imgfiles = []
        self.LR_imgfiles = []
        for idx in range(len(self.GT_dir_list)):
            GT_imgfiles_cur = sorted(os.listdir(self.GT_dir_list[idx]))
            for img_idx in range(0, len(GT_imgfiles_cur) - N_frames + 1):
                self.GT_imgfiles.append([os.path.join(self.GT_dir_list[idx], img_f) for img_f in GT_imgfiles_cur[img_idx:img_idx + N_frames]])
        for idx in range(len(self.LR_dir_list)):
            LR_imgfiles_cur = sorted(os.listdir(self.LR_dir_list[idx]))
            for img_idx in range(0, len(LR_imgfiles_cur) - N_frames + 1):
                self.LR_imgfiles.append([os.path.join(self.LR_dir_list[idx], img_f) for img_f in LR_imgfiles_cur[img_idx:img_idx + N_frames]])

    def __getitem__(self, index):
        #### Configs
        scale = self.args.scale
        GT_size = self.args.GT_size
        LR_size = GT_size // scale
        FV_size = self.args.FV_size

        ### GT 
        GT_imgfiles = self.GT_imgfiles[index]
        GT_imgs = [np.array(PIL.Image.open(img)) for img in GT_imgfiles]

        #### Bicubic downsampling
        ### LR and LR_sr
        H_, W_, _ = GT_imgs[0].shape
        LR_imgfiles = self.LR_imgfiles[index]
        LR_imgs = [np.array(PIL.Image.open(img)) for img in LR_imgfiles]
        LR_sr_imgs = [np.array(PIL.Image.fromarray(img).resize((W_, H_), PIL.Image.BICUBIC)) for img in LR_imgs]
        
        ### Ref(FV) and Ref_sr
        Ref, Ref_sp, fv_sp = fovea_generator(GT_imgs, method='Evenscan', FV_HW=(FV_size, FV_size))

        #### Stacking
        GT_imgs = np.stack(GT_imgs, axis=0)
        LR_imgs = np.stack(LR_imgs, axis=0)
        LR_sr_imgs = np.stack(LR_sr_imgs, axis=0)
        Ref = np.stack(Ref, axis=0)
        Ref_sp = np.stack(Ref_sp, axis=0)

        #### Scaling
        GT_imgs = GT_imgs.astype(np.float32) / 255.
        LR_imgs = LR_imgs.astype(np.float32) / 255.
        LR_sr_imgs = LR_sr_imgs.astype(np.float32) / 255.
        Ref = Ref.astype(np.float32) / 255.
        Ref_sp = Ref_sp.astype(np.bool_)

        GT_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(GT_imgs, (0, 3, 1, 2)))).float()
        LR_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(LR_imgs, (0, 3, 1, 2)))).float()
        LR_sr_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(LR_sr_imgs, (0, 3, 1, 2)))).float()
        Ref = torch.from_numpy(np.ascontiguousarray(np.transpose(Ref, (0, 3, 1, 2)))).float()
        Ref_sp = torch.from_numpy(np.ascontiguousarray(np.transpose(Ref_sp, (0, 3, 1, 2))))

        return {'LR': LR_imgs, 
                'LR_sr': LR_sr_imgs,
                'HR': GT_imgs,
                'Ref': Ref,
                'Ref_sp': Ref_sp,
                'FV_sp': fv_sp}
                
    def __len__(self):
        return len(self.GT_imgfiles)