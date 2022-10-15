from utils import calc_psnr_and_ssim_cuda, bgr2ycbcr

import os
import numpy as np
from imageio import imread, imsave, get_writer
from PIL import Image
import random
import time
from math import cos, pi
from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
import visdom

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

def get_position_from_periods(iteration, cumulative_periods):
    """Get the position from a period list.
    It will return the index of the right-closest number in the period list.
    For example, the cumulative_periods = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 3.
    Args:
        iteration (int): Current iteration.
        cumulative_periods (list[int]): Cumulative period list.
    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_periods):
        if iteration < period:
            return i
    raise ValueError(f'Current iteration {iteration} exceeds '
                     f'cumulative_periods {cumulative_periods}')


def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.
    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.
    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out

class Visdom_exe(object):
    def __init__(self, port='8907', env='main'):
        self.port = port
        self.env = env
        self.viz = visdom.Visdom(server='140.113.212.214', port=port, env=env)

    def plot_metric(self, loss=[], psnr=[], ssim=[], psnr_cuda=[], ssim_cuda=[], psnr_y_cuda=[], ssim_y_cuda=[], phase='train'):
        if len(loss) != 0:
            self.viz.line(X=[*range(len(loss))], Y=loss, win='{}_LOSS'.format(phase), opts={'title':'{}_LOSS'.format(phase)})
        if len(psnr) != 0:
            self.viz.line(X=[*range(len(psnr))], Y=psnr, win='{}_PSNR'.format(phase), opts={'title':'{}_PSNR'.format(phase)})
        if len(ssim) != 0:
            self.viz.line(X=[*range(len(ssim))], Y=ssim, win='{}_SSIM'.format(phase), opts={'title':'{}_SSIM'.format(phase)})
        if len(psnr_cuda) != 0:
            self.viz.line(X=[*range(len(psnr_cuda))], Y=psnr_cuda, win='{}_PSNR_cuda'.format(phase), opts={'title':'{}_PSNR_cuda'.format(phase)})
        if len(ssim_cuda) != 0:
            self.viz.line(X=[*range(len(ssim_cuda))], Y=ssim_cuda, win='{}_SSIM_cuda'.format(phase), opts={'title':'{}_SSIM_cuda'.format(phase)})
        if len(psnr_y_cuda) != 0:
            self.viz.line(X=[*range(len(psnr_y_cuda))], Y=psnr_y_cuda, win='{}_PSNR_y_cuda'.format(phase), opts={'title':'{}_PSNR_y_cuda'.format(phase)})
        if len(ssim_y_cuda) != 0:
            self.viz.line(X=[*range(len(ssim_y_cuda))], Y=ssim_y_cuda, win='{}_SSIM_y_cuda'.format(phase), opts={'title':'{}_SSIM_y_cuda'.format(phase)})
        
class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all):
        self.viz = Visdom_exe(args.visdom_port, args.visdom_view)
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.loss_all = loss_all
        # self.device = torch.device('cpu') if args.cpu else torch.device('cuda:{}'.format(args.gpu_id))
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')

        self.cur_clip = 0

        #### CosineAnnealing settings ####
        self.cur_iter = 0
        self.by_epoch = False
        self.periods = [600000]
        self.min_lr = 1e-7
        self.restart_weights = [1]
        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        #### CosineAnnealing settings ####
            
        self.params = [
            {"params": [p for n, p in (self.model.named_parameters() if 
             args.num_gpu==1 else self.model.module.named_parameters()) 
             if ('spynet' not in n)],
             "lr": args.lr_rate
            },
            {"params": [p for n, p in (self.model.named_parameters() if 
             args.num_gpu==1 else self.model.module.named_parameters()) 
             if ('spynet' in n)],
             "lr": args.lr_rate_flow
            }
        ]

        #### TTSR settings ####
        # self.optimizer = optim.Adam(self.params, lr=args.lr_rate, betas=(args.beta1, args.beta2), eps=args.eps)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)

        #### BasicVSR settings ####
        self.optimizer = optim.Adam(self.params, lr=args.lr_rate, betas=(args.beta1, args.beta2), eps=args.eps)

        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0

        self.max_y_psnr = 0.
        self.max_y_psnr_epoch = 0
        self.max_y_ssim = 0.
        self.max_y_ssim_epoch = 0
        
        self.train_loss_list = []
        self.train_psnr_list = []
        self.train_ssim_list = []
        self.train_psnr_cuda_list = []
        self.train_ssim_cuda_list = []
        self.train_psnr_y_cuda_list = []
        self.train_ssim_y_cuda_list = []

        self.eval_loss_list = []
        self.eval_psnr_list = []
        self.eval_ssim_list = []
        self.eval_psnr_cuda_list = []
        self.eval_ssim_cuda_list = []
        self.eval_psnr_y_cuda_list = []
        self.eval_ssim_y_cuda_list = []

        self.test_psnr = []
        self.test_ssim = []
        self.test_lr = []
        self.test_hr = []
        self.test_sr = []

        self.print_network(self.model)

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)

            model_state_dict = self.model.state_dict()
            for k in model_state_dict.keys():
                print(k)
            print('-----')
            model_state_dict_save = {k.replace('basic_', 'basic_module.'):v for k,v in torch.load(model_path).items() if k.replace('basic_', 'basic_module.') in model_state_dict}
            # model_state_dict_save = {k.replace('basic_','basic_module.'):v for k,v in torch.load(model_path).items()}
            # model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location=self.device)['state_dict'].items()}
            for k in model_state_dict_save.keys():
                print(k)
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict, strict=True)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train_basicvsr(self, current_epoch=0):
        self.model.train()
        loss_list = []
        psnr_cuda_list = []
        ssim_cuda_list = []
        psnr_y_cuda_list = []
        ssim_y_cuda_list = []
        loop = tqdm(enumerate(self.dataloader['train']), total=len(self.dataloader['train']), leave=False)
        for i_batch, sample_batched in loop:
            sample_batched = self.prepare(sample_batched)
            lr = sample_batched['LR']
            lr_sr = sample_batched['LR_sr']
            hr = sample_batched['HR']
            ref = sample_batched['Ref']
            # ref_sr = sample_batched['Ref_sr']
            ref_sp = sample_batched['Ref_sp']

            if self.cur_iter < 5000:
                for k, v in self.model.named_parameters():
                    if 'spynet' in k:
                        v.requires_grad_(False)
            elif self.cur_iter == 5000:
                #### train all the parameters
                self.model.requires_grad_(True)
            
            self.before_train_iter()

            sr = self.model(lrs=lr, fvs=ref, mks=ref_sp)
            B, N, C, H, W = sr.size()
            sr = sr.view(B*N, C, H, W)
            B, N, C, H, W = hr.size()
            hr = hr.view(B*N, C, H, W)

            if self.args.y_only:
                B, N, C, H, W = lr_sr.size()
                lr_sr = lr_sr.view(B*N, C, H, W)
                lr_sr = rgb2yuv(lr_sr, y_only=False)
                sr = yuv2rgb(torch.cat((sr[:,0:1,:,:], lr_sr[:,1:3,:,:]), dim=1))

            rec_loss = self.args.rec_w * self.loss_all['cb_loss'](sr, hr)
            loss = rec_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.cpu().item())
            
            ### calculate psnr and ssim
            #### RGB domain ####
            # B, N, C, H, W = sr.size()
            mk_ = torch.ones((B*N, 1, H, W)).to(sr.device)
            psnr, ssim = calc_psnr_and_ssim_cuda(sr.detach(), hr.detach(), mk_)
            psnr_cuda_list.append(psnr.cpu().item())
            ssim_cuda_list.append(ssim.cpu().item())

            # #### YCbCr domain ####
            # B, N, C, H, W = sr.size()
            psnr, ssim = calc_psnr_and_ssim_cuda(bgr2ycbcr(sr.permute(0, 2, 3, 1).detach(), y_only=True), \
                                                 bgr2ycbcr(hr.permute(0, 2, 3, 1).detach(), y_only=True), mk_)
            # psnr, ssim = calc_psnr_and_ssim_cuda(sr.view(B*N, C, H, W).permute(0, 2, 3, 1).detach(), \
                                                #  hr.view(B*N, C, H, W).permute(0, 2, 3, 1).detach(), mk_)
            psnr_y_cuda_list.append(psnr.cpu().item())
            ssim_y_cuda_list.append(ssim.cpu().item())

            loop.set_description(f"Epoch[{current_epoch}/{self.args.num_epochs}](Train)")
            # loop.set_postfix(loss=loss.item(), psnr=psnr.item(), ssim=ssim.item())
            loop.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'], lr_flow=self.optimizer.param_groups[1]['lr'])

            self.cur_iter += 1

            if self.cur_iter % self.args.save_every == 0:
                tmp = self.model.state_dict()
                model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp}
                model_name = self.args.save_dir.strip('/')+'/model/model_{}_{}.pt'.format(str(current_epoch).zfill(5), str(self.cur_iter).zfill(6))
                torch.save(model_state_dict, model_name)
                self.train_loss_list.append(sum(loss_list)/len(loss_list))
                self.train_psnr_cuda_list.append(sum(psnr_cuda_list)/len(psnr_cuda_list))
                self.train_ssim_cuda_list.append(sum(ssim_cuda_list)/len(ssim_cuda_list))
                self.train_psnr_y_cuda_list.append(sum(psnr_y_cuda_list)/len(psnr_y_cuda_list))
                self.train_ssim_y_cuda_list.append(sum(ssim_y_cuda_list)/len(ssim_y_cuda_list))
                loss_list.clear()
                psnr_cuda_list.clear()
                ssim_cuda_list.clear()
                psnr_y_cuda_list.clear()
                ssim_y_cuda_list.clear()

            # if self.cur_iter % self.args.print_every == 0:
                # self.test_basicvsr(save_img=False)

    def eval_basicvsr(self, current_epoch=0):
        self.model.eval()
        psnr_cuda_list = []
        ssim_cuda_list = []
        psnr_y_cuda_list = []
        ssim_y_cuda_list = []
        # kernel = np.array([ [1, 1, 1],
        #                     [1, 1, 1],
        #                     [1, 1, 1] ], dtype=np.float32)
        # kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)) # size: (1, 1, 3, 3)
        loop = tqdm(enumerate(self.dataloader['eval']), total=len(self.dataloader['eval']), leave=False)
        with torch.no_grad():
            for i_batch, sample_batched in loop:
                sample_batched = self.prepare(sample_batched)
                lr = sample_batched['LR']
                lr_sr = sample_batched['LR_sr']
                hr = sample_batched['HR']
                ref = sample_batched['Ref']
                ref_sp = sample_batched['Ref_sp']

                B, N, C, H, W = ref_sp.size()
                mk = torch.split(ref_sp.view(B*N, 1, H, W).clone().detach(), 1, dim=0)
                mk_bd = ref_sp.view(B*N, 1, H, W).float().clone().detach()
                sr = self.model(lrs=lr, fvs=ref, mks=ref_sp)
                # LR_fv = (sr * 255.).clip(0., 255.).squeeze().clone().detach()
                # for n in range(N):
                    # self.viz.viz.image(LR_fv[n , :, :, :].cpu().numpy(), win='{}'.format('FV'), opts=dict(title='{}, Image size : {}'.format('FV', LR_fv.size())))
                # sr = lr_sr

                ### calculate psnr and ssim
                B, N, C, H, W = sr.size()
                sr = sr.view(B*N, C, H, W)
                B, N, C, H, W = hr.size()
                hr = hr.view(B*N, C, H, W)

                if self.args.y_only:
                    B, N, C, H, W = lr_sr.size()
                    lr_sr = lr_sr.view(B*N, C, H, W)
                    lr_sr = rgb2yuv(lr_sr, y_only=False)
                    sr = yuv2rgb(torch.cat((sr[:,0:1,:,:], lr_sr[:,1:3,:,:]), dim=1))

                # LR_fv = (sr * 255.).clip(0., 255.).squeeze().clone().detach()
                # for i in range(B*N):
                    # self.viz.viz.image(LR_fv[i].cpu().numpy(), win='{}'.format('FV'), opts=dict(title='{}, Image size : {}'.format('FV', LR_fv.size())))

                # mk = torch.split(ref_sp.view(B*N, 1, H, W), 1, dim=0)
                # mk_bd = ref_sp.view(B*N, 1, H, W).float()
                # for _ in range(10):
                    # mk_bd = torch.clamp(F.conv2d(mk_bd, kernel_tensor.to(mk_bd.device), padding=(1, 1)), 0, 1)
                # mk_bd = torch.split(mk_bd, 1, dim=0)
                # past_len = 3
                # if i_batch % 50 == 0:
                    # mk_pre = [mk_bd[0]]
                mk_ = torch.ones((B*N, 1, H, W)).to(sr.device)
                for idx in range(0, N):
                    if idx == 0 and i_batch % 50 == 0:
                        continue
                    sr_ = sr[idx, :, :, :].unsqueeze(0)
                    hr_ = hr[idx, :, :, :].unsqueeze(0)
                    # mk_ = mk[idx]
                    # mk_ = torch.logical_and(mk_, torch.logical_not(mk[idx]))
                    # mk_ = torch.logical_or(mk_, mk[idx - 1])
                    # mk_ = torch.logical_and(torch.logical_not(mk[idx]), mk_bd[idx])
                    # mk_ = torch.sum(torch.cat(mk_pre, dim=1), dim=1, keepdim=True).clip(0, 1)
                    # psnr, ssim = calc_psnr_and_ssim_cuda(sr.view(B*N, C, H, W).detach(), hr.view(B*N, C, H, W).detach())
                    # psnr, ssim = calc_psnr_and_ssim_cuda((sr_*mk_).detach(), (hr_*mk_).detach())
                    psnr, ssim = calc_psnr_and_ssim_cuda(sr_.detach(), hr_.detach(), mk_)
                    psnr_cuda_list.append(psnr.cpu().item())
                    ssim_cuda_list.append(ssim.cpu().item())
                    psnr, ssim = calc_psnr_and_ssim_cuda(bgr2ycbcr(sr_.permute(0, 2, 3, 1).detach(), y_only=True), \
                                                         bgr2ycbcr(hr_.permute(0, 2, 3, 1).detach(), y_only=True), mk_)
                    # psnr, ssim = calc_psnr_and_ssim_cuda(sr_.detach(), \
                                                        #  hr_.detach(), mk_)
                    psnr_y_cuda_list.append(psnr.cpu().item())
                    ssim_y_cuda_list.append(ssim.cpu().item())
                    
                    # mk_pre.append(mk_bd[idx])
                    # if len(mk_pre) > past_len:
                        # mk_pre.pop(0)
                
                psnr_ave = sum(psnr_cuda_list)/len(psnr_cuda_list)
                ssim_ave = sum(ssim_cuda_list)/len(ssim_cuda_list)
                psnr_ave_y = sum(psnr_y_cuda_list)/len(psnr_y_cuda_list)
                ssim_ave_y = sum(ssim_y_cuda_list)/len(ssim_y_cuda_list)
                loop.set_description(f"Epoch[{current_epoch}/{self.args.num_epochs}](Eval)")
                loop.set_postfix(psnr=psnr_ave, ssim=ssim_ave, psnr_y=psnr_ave_y, ssim_y=ssim_ave_y)

            psnr_ave = sum(psnr_cuda_list)/len(psnr_cuda_list)
            ssim_ave = sum(ssim_cuda_list)/len(ssim_cuda_list)
            self.eval_psnr_cuda_list.append(psnr_ave)
            self.eval_ssim_cuda_list.append(ssim_ave)
            self.logger.info('Ref  PSNR (now): %.3f \t SSIM (now): %.4f' %(psnr_ave, ssim_ave))
            if (psnr_ave > self.max_psnr):
                self.max_psnr = psnr_ave
                self.max_psnr_epoch = current_epoch
            if (ssim_ave > self.max_ssim):
                self.max_ssim = ssim_ave
                self.max_ssim_epoch = current_epoch
            self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)' 
                %(self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))

            psnr_y_ave = sum(psnr_y_cuda_list)/len(psnr_y_cuda_list)
            ssim_y_ave = sum(ssim_y_cuda_list)/len(ssim_y_cuda_list)
            self.eval_psnr_y_cuda_list.append(psnr_y_ave)
            self.eval_ssim_y_cuda_list.append(ssim_y_ave)
            self.logger.info('Ref  PSNR_Y (now): %.3f \t SSIM_Y (now): %.4f' %(psnr_y_ave, ssim_y_ave))
            if (psnr_y_ave > self.max_y_psnr):
                self.max_y_psnr = psnr_y_ave
                self.max_y_psnr_epoch = current_epoch
            if (ssim_y_ave > self.max_y_ssim):
                self.max_y_ssim = ssim_y_ave
                self.max_y_ssim_epoch = current_epoch
            self.logger.info('Ref  PSNR_Y (max): %.3f (%d) \t SSIM_Y (max): %.4f (%d)' 
                %(self.max_y_psnr, self.max_y_psnr_epoch, self.max_y_ssim, self.max_y_ssim_epoch))

            psnr_cuda_list.clear()
            ssim_cuda_list.clear()
            psnr_y_cuda_list.clear()
            ssim_y_cuda_list.clear()

    def test_basicvsr(self, save_img=True):
        if save_img:
            self.print_network(self.model)
            self.logger.info('Test process...')

        crop_h = self.args.FV_size
        crop_w = self.args.FV_size
        # kernel_size = self.args.FV_size
        kernel_size = 10
        stride_size = 5
        self.model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(self.dataloader['test']):
                sample_batched = self.prepare(sample_batched)
                lr = sample_batched['LR']
                lr_sr = sample_batched['LR_sr']
                hr = sample_batched['HR']
                ref = sample_batched['Ref']
                ref_sp = sample_batched['Ref_sp']
                fv_sp = sample_batched['FV_sp']
                B, N, C, H, W = hr.size()
                
                sr = self.model(lrs=lr, fvs=ref, mks=ref_sp)
                if self.args.y_only:
                    B, N, C, H, W = sr.size()
                    sr = sr.view(B*N, C, H, W)
                    B, N, C, H, W = lr_sr.size()
                    lr_sr = lr_sr.view(B*N, C, H, W)
                    lr_sr = rgb2yuv(lr_sr, y_only=False)
                    sr = yuv2rgb(torch.cat((sr[:,0:1,:,:], lr_sr[:,1:3,:,:]), dim=1))

                LR = (lr * 255.).squeeze().clone().detach()
                HR = (hr * 255.).squeeze().clone().detach()
                LR_fv = (sr * 255.).clip(0., 255.).squeeze().clone().detach()
                Ref_sp = fv_sp.squeeze().clone().detach()

                # direction = -1
                # scan_step = 8
                # accm_step = W - scan_step
                        
                for n in range(N):
                    psnr_score, ssim_score = self.foveated_metric(LR[n , :, :, :], LR_fv[n , :, :, :], HR[n , :, :, :], Ref_sp[n], (H, W), (crop_h, crop_w), kernel_size, stride_size)    
                    self.test_psnr.append((psnr_score.unsqueeze(2).repeat(1, 1, 3) * 255).round().cpu().detach().numpy())
                    self.test_ssim.append((ssim_score.unsqueeze(2).repeat(1, 1, 3) * 255).round().cpu().detach().numpy())
                    # self.result_comp(LR_fv[n , :, :, :], accm_step)
                    # accm_step += direction * scan_step
                    # if accm_step < 0:
                    #     direction *= -1
                    #     accm_step += direction * scan_step
                    # elif accm_step >= W:
                    #     direction *= -1
                    #     accm_step += direction * scan_step
                    time.sleep(0.1)
                   
                print('Process: {:.2f} %  ...\r'.format(i_batch*100/len(self.dataloader['test'])), end='')
                
                for n in range(N):
                    self.test_lr.append(LR[n, :, :, :].round().cpu().numpy())
                    self.test_hr.append(HR[n, :, :, :].round().cpu().numpy())
                    self.test_sr.append(LR_fv[n, :, :, :].round().cpu().numpy())
                
                if save_img:
                    if len(self.test_sr) == 100:
                        N = 100
                        save_path = os.path.join(self.args.save_dir, 'save_results', '{:05d}'.format(self.cur_clip))
                        if not os.path.isdir(save_path):
                            os.mkdir(save_path)
                        with get_writer(os.path.join(save_path, '{:05d}.gif'.format(i_batch)), mode="I", fps=5) as writer:
                            for n in range(N):
                                imsave(os.path.join(save_path, '{}_sr.png'.format(n)), np.transpose(self.test_sr[n], (1, 2, 0)).astype(np.uint8))
                                writer.append_data(np.transpose(self.test_sr[n], (1, 2, 0)).astype(np.uint8))
                        with get_writer(os.path.join(save_path, '{:05d}_gt.gif'.format(i_batch)), mode="I", fps=5) as writer:
                            for n in range(N):
                                imsave(os.path.join(save_path, '{}_hr.png'.format(n)), np.transpose(self.test_hr[n], (1, 2, 0)).astype(np.uint8))
                                writer.append_data(np.transpose(self.test_hr[n], (1, 2, 0)).astype(np.uint8))
                        with get_writer(os.path.join(save_path, '{:05d}_lr.gif'.format(i_batch)), mode="I", fps=5) as writer:
                            for n in range(N):
                                imsave(os.path.join(save_path, '{}_lr.png'.format(n)), np.transpose(self.test_lr[n], (1, 2, 0)).astype(np.uint8))
                                writer.append_data(np.transpose(self.test_lr[n], (1, 2, 0)).astype(np.uint8))
                        with get_writer(os.path.join(save_path, '{:05d}_psnr.gif'.format(i_batch)), mode="I", fps=5) as writer:
                            for n in range(N):
                                imsave(os.path.join(save_path, '{}_psnr.png'.format(n)), self.test_psnr[n].astype(np.uint8))
                                writer.append_data(self.test_psnr[n].astype(np.uint8))
                        with get_writer(os.path.join(save_path, '{:05d}_ssim.gif'.format(i_batch)), mode="I", fps=5) as writer:
                            for n in range(N):
                                imsave(os.path.join(save_path, '{}_ssim.png'.format(n)), self.test_ssim[n].astype(np.uint8))
                                writer.append_data(self.test_ssim[n].astype(np.uint8))
                        self.test_lr.clear()
                        self.test_hr.clear()
                        self.test_sr.clear()
                        self.test_psnr.clear()
                        self.test_ssim.clear()
                        self.cur_clip += 1
                else:
                    break
            
        if save_img:
            self.logger.info('Test over.')

    def test_basicvsr_scan(self, save_img=True):
        if save_img:
            self.print_network(self.model)
            self.logger.info('Test process...')
            self.logger.info('lr path:     %s' %(self.args.lr_path))
            self.logger.info('ref path:    %s' %(self.args.ref_path))

        self.model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(self.dataloader['test']):
                sample_batched = self.prepare(sample_batched)
                lr = sample_batched['LR']
                hr = sample_batched['HR']
                ref = sample_batched['Ref']
                ref_sp = sample_batched['Ref_sp']
                fv_sp = sample_batched['FV_sp']
                B, N, C, H, W = hr.size()
                
                sr = self.model(lrs=lr, fvs=ref, mks=ref_sp)
                #### Save images ####
                if save_img:
                    sr_save = (sr  * 255.).clip(0., 225.).squeeze()
                    SR_t = sr
                    sr_save = np.transpose(sr_save.round().cpu().detach().numpy(), (0, 2, 3, 1)).astype(np.uint8)
                    for t in range(N):
                        save_path = os.path.join(self.args.save_dir, 'save_results', '{:05d}'.format(i_batch))
                        if not os.path.isdir(save_path):
                            os.mkdir(save_path)
                        imsave(os.path.join(save_path, '{}.png'.format(t)), sr_save[t])
                #### Save images ####
                LR = (lr * 255.).squeeze().clone().detach()
                HR = (hr * 255.).squeeze().clone().detach()
                LR_fv = (sr * 255.).clip(0., 255.).squeeze().clone().detach()
                Ref_sp = fv_sp.squeeze().clone().detach()
                for n in range(N):
                    self.result_comp(LR_fv[n , :, :, :])
                    time.sleep(0.1)
                if save_img:
                    with get_writer(os.path.join(save_path, '{:05d}.gif'.format(i_batch)), mode="I", fps=5) as writer:
                        for n in range(N):
                            writer.append_data(np.transpose(LR_fv[n].round().cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
                    print('Process: {:.2f} %  ...\r'.format(i_batch*100/len(self.dataloader['test'])), end='')
                else:
                    break
        
        if save_img:
            self.logger.info('Test over.')

    def vis_plot_metric(self, phase):
        if phase == 'train':
            self.viz.plot_metric(loss=self.train_loss_list, psnr=self.train_psnr_list, ssim=self.train_ssim_list, \
                                 psnr_cuda=self.train_psnr_cuda_list, ssim_cuda=self.train_ssim_cuda_list, \
                                 psnr_y_cuda=self.train_psnr_y_cuda_list, ssim_y_cuda=self.train_ssim_y_cuda_list, \
                                 phase='train')
        elif phase == 'eval':
            self.viz.plot_metric(loss=self.eval_loss_list,  psnr=self.eval_psnr_list,  ssim=self.eval_ssim_list,  \
                                 psnr_cuda=self.eval_psnr_cuda_list,  ssim_cuda=self.eval_ssim_cuda_list,  \
                                 psnr_y_cuda=self.eval_psnr_y_cuda_list,  ssim_y_cuda=self.eval_ssim_y_cuda_list,  \
                                 phase='eval')

    def _get_network_description(self, net):
        """Get the string and total parameters of the network"""
        if isinstance(net, nn.DataParallel):
            net = net.module
        return str(net), sum(map(lambda x: x.numel(), net.parameters()))

    def print_network(self, net):
        """Print the str and parameter number of a network.
        Args:
            net (nn.Module)
        """
        net_str, net_params = self._get_network_description(net)
        if isinstance(net, nn.DataParallel):
            net_cls_str = (f'{net.__class__.__name__} - '
                           f'{net.module.__class__.__name__}')
        else:
            net_cls_str = f'{net.__class__.__name__}'

        self.logger.info(
            f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        self.logger.info(net_str)

    def before_run(self):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in self.optimizer.param_groups
        ]

    def before_train_iter(self):
        self.regular_lr = self.get_regular_lr()
        self._set_lr(self.regular_lr)

    def get_regular_lr(self):
        return [self.get_lr(_base_lr) for _base_lr in self.base_lr]

    def get_lr(self, base_lr):
        progress = self.cur_iter
        target_lr = self.min_lr

        idx = get_position_from_periods(progress, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((progress - nearest_restart) / current_periods, 1)
        return annealing_cos(base_lr, target_lr, alpha, current_weight)

    def _set_lr(self, lr_groups):
        for param_group, lr in zip(self.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def foveated_metric(self, LR, LR_fv, HR, mn, hw, crop, kernel_size, stride_size):
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

        psnr_y_idx = (torch.argmax(psnr_score) // Wr) * stride_size
        psnr_x_idx = (torch.argmax(psnr_score) %  Wr) * stride_size
        ssim_y_idx = (torch.argmax(ssim_score) // Wr) * stride_size
        ssim_x_idx = (torch.argmax(ssim_score) %  Wr) * stride_size

        LR_fv[:, m:m+crop_h, n]          = torch.tensor([255., 0., 0.]).unsqueeze(1).repeat((1,crop_h))
        LR_fv[:, m:m+crop_h, n+crop_w-1] = torch.tensor([255., 0., 0.]).unsqueeze(1).repeat((1,crop_h))
        LR_fv[:, m,          n:n+crop_w] = torch.tensor([255., 0., 0.]).unsqueeze(1).repeat((1,crop_w))
        LR_fv[:, m+crop_h-1, n:n+crop_w] = torch.tensor([255., 0., 0.]).unsqueeze(1).repeat((1,crop_w))

        psnr_score_discrete = torch.zeros_like(psnr_score)
        ssim_score_discrete = torch.zeros_like(ssim_score)

        psnr_score = (psnr_score - psnr_score.min()) / (psnr_score.max() - psnr_score.min())
        ssim_score = (ssim_score - ssim_score.min()) / (ssim_score.max() - ssim_score.min())

        psnr_score_discrete[psnr_score <= 1.0] = 1.0
        psnr_score_discrete[psnr_score <= 0.9] = 0.9
        psnr_score_discrete[psnr_score <= 0.8] = 0.8
        psnr_score_discrete[psnr_score <= 0.7] = 0.7
        psnr_score_discrete[psnr_score <= 0.6] = 0.6
        psnr_score_discrete[psnr_score <= 0.5] = 0.5
        psnr_score_discrete[psnr_score <= 0.4] = 0.4
        psnr_score_discrete[psnr_score <= 0.3] = 0.3
        psnr_score_discrete[psnr_score <= 0.2] = 0.2
        psnr_score_discrete[psnr_score <= 0.1] = 0.1

        ssim_score_discrete[ssim_score <= 1.0] = 1.0
        ssim_score_discrete[ssim_score <= 0.9] = 0.9
        ssim_score_discrete[ssim_score <= 0.8] = 0.8
        ssim_score_discrete[ssim_score <= 0.7] = 0.7
        ssim_score_discrete[ssim_score <= 0.6] = 0.6
        ssim_score_discrete[ssim_score <= 0.5] = 0.5
        ssim_score_discrete[ssim_score <= 0.4] = 0.4
        ssim_score_discrete[ssim_score <= 0.3] = 0.3
        ssim_score_discrete[ssim_score <= 0.2] = 0.2
        ssim_score_discrete[ssim_score <= 0.1] = 0.1

        # self.viz.viz.image(HR.cpu().numpy(), win='{}'.format('HR'), opts=dict(title='{}, Image size : {}'.format('HR', HR.size())))
        # self.viz.viz.image(LR.cpu().numpy(), win='{}'.format('LR'), opts=dict(title='{}, Image size : {}'.format('LR', LR.size())))
        self.viz.viz.image(LR_fv.cpu().numpy(), win='{}'.format('FV'), opts=dict(title='{}, Image size : {}'.format('FV', LR_fv.size())))
        # self.viz.viz.image(psnr_score.cpu().numpy(), win='{}'.format('PSNR_score'), opts=dict(title='{}, Image size : {}'.format('PSNR_score', psnr_score.size())))
        # self.viz.viz.image(ssim_score.cpu().numpy(), win='{}'.format('SSIM_score'), opts=dict(title='{}, Image size : {}'.format('SSIM_score', ssim_score.size())))
        # self.viz.viz.image(psnr_score_discrete.cpu().numpy(), win='{}'.format('PSNR_score_discrete'), opts=dict(title='{}, Image size : {}'.format('PSNR_score_discrete', psnr_score_discrete.size())))
        # self.viz.viz.image(ssim_score_discrete.cpu().numpy(), win='{}'.format('SSIM_score_discrete'), opts=dict(title='{}, Image size : {}'.format('SSIM_score_discrete', ssim_score_discrete.size())))
    
        return psnr_score, ssim_score

    def result_comp(self, LR_fv, SP_W):
        C, H, W = LR_fv.size()
        LR_fv[:, :, SP_W] = torch.tensor([255., 255., 255.]).unsqueeze(1).repeat((1,H))
        self.viz.viz.image(LR_fv.cpu().numpy(), win='{}'.format('LR_fv'), opts=dict(title='{}, Image size : {}'.format('LR_fv', LR_fv.size())))
