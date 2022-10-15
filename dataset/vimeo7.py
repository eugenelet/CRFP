
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
import torch.nn.functional as nnF
from torchvision.transforms import Compose, ToTensor

logger = logging.getLogger('base')

def gaussian_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code
    Args:
        x (Tensor, [C, T, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    if scale == 2:
        h = gkern(13, 0.8)  # 13 and 0.8 for x2
    elif scale == 3:
        h = gkern(13, 1.2)  # 13 and 1.2 for x3
    elif scale == 4:
        h = gkern(13, 1.6)  # 13 and 1.6 for x4
    else:
        print('Invalid upscaling factor: {} (Must be one of 2, 3, 4)'.format(R))
        exit(1)

    C, T, H, W = x.size()
    x = x.contiguous().view(-1, 1, H, W) # depth convolution (channel-wise convolution)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0

    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)

    x = nnF.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], mode='reflect')
    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = nnF.conv2d(x, gaussian_filter, stride=scale)
    # please keep the operation same as training.
    # if  downsample to 32 on training time, use the below code.
    x = x[:, :, 2:-2, 2:-2]
    # if downsample to 28 on training time, use the below code.
    #x = x[:,:,scale:-scale,scale:-scale]
    x = x.view(C, T, x.size(2), x.size(3)).permute(1, 0 ,2, 3)
    return x

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
        SP_H = 0
        EP_H = (GT_H - FV_H - 1) / GT_H
        Q1_H = (0.25 - (FV_H / GT_H)/2) if (0.25 - (FV_H / GT_H)/2) >  0 else SP_H
        Q2_H = (0.50 - (FV_H / GT_H)/2)
        Q3_H = (0.75 - (FV_H / GT_H)/2) if (0.75 + (FV_H / GT_H)/2) <= 1 else EP_H
        T1_H = (0.33 - (FV_H / GT_H)/2) if (0.33 - (FV_H / GT_H)/2) >  0 else SP_H
        T2_H = (0.66 - (FV_H / GT_H)/2) if (0.66 + (FV_H / GT_H)/2) <= 1 else EP_H
        SP_W = 0
        EP_W = (GT_W - FV_W - 1) / GT_W
        Q1_W = (0.25 - (FV_W / GT_W)/2) if (0.25 - (FV_W / GT_W)/2) >  0 else SP_W
        Q2_W = (0.50 - (FV_W / GT_W)/2)
        Q3_W = (0.75 - (FV_W / GT_W)/2) if (0.75 + (FV_W / GT_W)/2) <= 1 else EP_W
        T1_W = (0.33 - (FV_W / GT_W)/2) if (0.33 - (FV_W / GT_W)/2) >  0 else SP_W
        T2_W = (0.66 - (FV_W / GT_W)/2) if (0.66 + (FV_W / GT_W)/2) <= 1 else EP_W

        fv_sp = [[Q1_H, T1_W], [Q1_H, T2_W], [Q2_H, Q1_W], [Q2_H, Q2_W], [Q2_H, Q3_W], [Q3_H, T1_W], [Q3_H, T2_W]]
        fv_sp = [[int(v[0] * GT_H), int(v[1] * GT_W)] for v in fv_sp]
        random.shuffle(fv_sp)
    else:
        fv_sp = [[int((v / 100) * GT_H), int((v / 100) * GT_W)] for v in [*range(SP, EP, step)]]

    fv_sp = torch.tensor(fv_sp)
    if torch.is_tensor(GT_imgs):
        FV_imgs = []
        Ref_sps = []
        for t in range(len(GT_imgs)):
            #### With padding ####
            Ref_sp = torch.zeros_like(GT_imgs[t])
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
            Ref_sp[fv_sp[t][0]:fv_sp[t][0] + FV_H, fv_sp[t][1]:fv_sp[t][1] + FV_W, :] = 1
            Ref = GT_imgs[t] * Ref_sp
            FV_imgs.append(Ref)
            Ref_sps.append(Ref_sp)
            #### Without padding ####
            # FV_imgs.append(GT_imgs[t][fv_sp[t][0]:fv_sp[t][0] + FV_H, fv_sp[t][1]:fv_sp[t][1] + FV_W, :].copy())
        # FV_imgs = np.stack(FV_imgs, dim=0)

    return FV_imgs, Ref_sps, fv_sp
    # return FV_imgs, fv_sp

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
        
        self.LR_dir_list = []
        self.GT_dir_list = []
        GT_list = open(os.path.join(args.dataset_dir, 'sep_trainlist.txt'), 'r')
        LR_root = args.dataset_dir.replace('90k', '90k_BD')
        for line in GT_list.readlines():
            self.GT_dir_list.append(os.path.join(args.dataset_dir, 'sequences', line.strip()))
            self.LR_dir_list.append(os.path.join(LR_root, 'sequences', line.strip()))
        
        self.transform = Compose([ToTensor()])

    def __getitem__(self, index):
        #### Configs
        scale = self.args.scale
        GT_size = self.args.GT_size
        LR_size = GT_size // scale
        FV_size = self.args.FV_size

        #### Bicubic downsampling
        ### GT 
        GT_imgfiles = sorted(os.listdir(self.GT_dir_list[index]))
        GT_imgs = [np.array(PIL.Image.open(os.path.join(self.GT_dir_list[index], img))) for img in GT_imgfiles]

        ### LR and LR_sr
        H, W, C = GT_imgs[0].shape
        LR_imgs = [np.array(PIL.Image.fromarray(img).resize((W // scale, H // scale), PIL.Image.BICUBIC)) for img in GT_imgs]

        #### Random cropping
        H, W, C = LR_imgs[0].shape
        rnd_h = random.randint(0, max(0, H - LR_size))
        rnd_w = random.randint(0, max(0, W - LR_size))
        LR_imgs = [v[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :] for v in LR_imgs]

        rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
        GT_imgs = [v[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :] for v in GT_imgs]
        
        ### Ref(FV) and Ref_sr
        Ref, Ref_sp, _ = fovea_generator(GT_imgs, method='Nanascan', FV_HW=(FV_size, FV_size))

        #### Stacking
        GT_imgs = np.stack(GT_imgs, axis=0)
        LR_imgs = np.stack(LR_imgs, axis=0)
        Ref = np.stack(Ref, axis=0)
        Ref_sp = np.stack(Ref_sp, axis=0)

        GT_imgs = GT_imgs.astype(np.float32) / 255.
        LR_imgs = LR_imgs.astype(np.float32) / 255.
        Ref = Ref.astype(np.float32) / 255.
        Ref_sp = Ref_sp.astype(np.bool_)

        GT_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(GT_imgs, (0, 3, 1, 2)))).float()
        LR_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(LR_imgs, (0, 3, 1, 2)))).float()
        Ref = torch.from_numpy(np.ascontiguousarray(np.transpose(Ref, (0, 3, 1, 2)))).float()
        Ref_sp = torch.from_numpy(np.ascontiguousarray(np.transpose(Ref_sp, (0, 3, 1, 2)))).float()

        if torch.rand(1) < 0.5:
            GT_imgs = F.hflip(GT_imgs)
            LR_imgs = F.hflip(LR_imgs)
            Ref = F.hflip(Ref)
            Ref_sp = F.hflip(Ref_sp)

        if torch.rand(1) < 0.5:
            GT_imgs = F.vflip(GT_imgs)
            LR_imgs = F.vflip(LR_imgs)
            Ref = F.vflip(Ref)
            Ref_sp = F.vflip(Ref_sp)

        return {'LR': LR_imgs, 
                'HR': GT_imgs,
                'Ref': Ref,
                'Ref_sp': Ref_sp}

    def __len__(self):
        return len(self.GT_dir_list)

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
        
        self.LR_dir_list = []
        self.GT_dir_list = []
        GT_list = open(os.path.join(args.dataset_dir, 'sep_testlist.txt'), 'r')
        LR_root = args.dataset_dir.replace('90k', '90k_BD')
        for line in GT_list.readlines():
            self.GT_dir_list.append(os.path.join(args.dataset_dir, 'sequences', line.strip()))
            self.LR_dir_list.append(os.path.join(LR_root, 'sequences', line.strip()))

        self.transform = Compose([ToTensor()])

    def __getitem__(self, index):
        #### Configs
        scale = self.args.scale
        GT_size = self.args.GT_size
        LR_size = GT_size // scale
        FV_size = self.args.FV_size

        #### Bicubic downsampling
        ### GT
        GT_imgfiles = sorted(os.listdir(self.GT_dir_list[index]))
        GT_imgs = [np.array(PIL.Image.open(os.path.join(self.GT_dir_list[index], img))) for img in GT_imgfiles]

        ### LR and LR_sr
        H, W, C = GT_imgs[0].shape
        LR_imgs = [np.array(PIL.Image.fromarray(img).resize((W // scale, H // scale), PIL.Image.BICUBIC)) for img in GT_imgs]

        ### Ref(FV) and Ref_sr
        Ref, Ref_sp, _ = fovea_generator(GT_imgs, method='Nanascan', FV_HW=(FV_size, FV_size))

        #### Stacking
        GT_imgs = np.stack(GT_imgs, axis=0)
        LR_imgs = np.stack(LR_imgs, axis=0)
        Ref = np.stack(Ref, axis=0)
        Ref_sp = np.stack(Ref_sp, axis=0)

        #### Scaling
        GT_imgs = GT_imgs.astype(np.float32) / 255.
        LR_imgs = LR_imgs.astype(np.float32) / 255.
        Ref = Ref.astype(np.float32) / 255.
        Ref_sp = Ref_sp.astype(np.bool_)

        GT_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(GT_imgs, (0, 3, 1, 2)))).float()
        LR_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(LR_imgs, (0, 3, 1, 2)))).float()
        Ref = torch.from_numpy(np.ascontiguousarray(np.transpose(Ref, (0, 3, 1, 2)))).float()
        Ref_sp = torch.from_numpy(np.ascontiguousarray(np.transpose(Ref_sp, (0, 3, 1, 2))))

        return {'LR': LR_imgs, 
                'HR': GT_imgs,
                'Ref': Ref,
                'Ref_sp': Ref_sp}
                
    def __len__(self):
        return len(self.GT_dir_list)

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
        
        self.LR_dir_list = []
        self.GT_dir_list = []
        GT_list = open(os.path.join(args.dataset_dir, 'slow_testset.txt'), 'r')
        LR_root = args.dataset_dir.replace('90k', '90k_LR')
        for line in GT_list.readlines():
            self.GT_dir_list.append(os.path.join(args.dataset_dir, 'sequences', line.strip()))
            self.LR_dir_list.append(os.path.join(LR_root, 'sequences', line.strip()))

    def __getitem__(self, index):
        #### Configs
        scale = self.args.scale

        #### Bicubic downsampling
        ### GT 
        GT_imgfiles = sorted(os.listdir(self.GT_dir_list[index]))
        GT_imgs = [np.array(PIL.Image.open(os.path.join(self.GT_dir_list[index], img))) for img in GT_imgfiles]
        GT_H, GT_W = GT_imgs[0].shape[:2]
        LR_H, LR_W = GT_H // scale, GT_W // scale
        FV_size = self.args.FV_size
        ### LR and LR_sr
        LR_imgs = [np.array(PIL.Image.fromarray(img).resize((LR_W, LR_H), PIL.Image.BICUBIC)) for img in GT_imgs]

        ### Ref(FV) and Ref_sr
        Ref, Ref_sp, fv_sp = fovea_generator(GT_imgs, method='Hscan', step=0.2, FV_HW=(FV_size, FV_size))
        
        #### Stacking
        GT_imgs = np.stack(GT_imgs, axis=0)
        LR_imgs = np.stack(LR_imgs, axis=0)
        Ref = np.stack(Ref, axis=0)
        Ref_sp = np.stack(Ref_sp, axis=0)

        #### Scaling
        GT_imgs = GT_imgs.astype(np.float32) / 255.
        LR_imgs = LR_imgs.astype(np.float32) / 255.
        Ref = Ref.astype(np.float32) / 255.
        Ref_sp = Ref_sp.astype(np.bool_)

        GT_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(GT_imgs, (0, 3, 1, 2)))).float()
        LR_imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(LR_imgs, (0, 3, 1, 2)))).float()
        Ref     = torch.from_numpy(np.ascontiguousarray(np.transpose(Ref, (0, 3, 1, 2)))).float()
        Ref_sp = torch.from_numpy(np.ascontiguousarray(np.transpose(Ref_sp, (0, 3, 1, 2))))

        return {'LR': LR_imgs, 
                'HR': GT_imgs,
                'Ref': Ref, 
                'Ref_sp': Ref_sp,
                'FV_sp': fv_sp}

    def __len__(self):
        return len(self.GT_dir_list)