from model import MRCF_runtime as MRCF

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_memlab import LineProfiler
from pytorch_memlab import MemReporter
from dcn_v2 import DCNv2

def conv_identify(weight, bias):
    weight.data.zero_()
    bias.data.zero_()
    o, i, h, w = weight.shape
    y = h//2
    x = w//2
    for p in range(i):
        for q in range(o):
            if p == q:
                weight.data[q, p, y, x] = 1.0

if __name__ == '__main__':
    device = torch.device('cuda')

    mid_channels = 32
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # print(torch.cuda.memory_allocated(0), '-3')
    # print(torch.cuda.max_memory_allocated(0), '-3')
    # torch.cuda.reset_max_memory_allocated(0)
    y_only = False
    hr_dcn = True
    offset_prop = True
    split_ratio = 3
    # model = MRCF.MRCF_simple_v0(mid_channels=32, y_only=y_only, hr_dcn=hr_dcn, offset_prop=offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
    # model = MRCF.MRCF_simple_v13(mid_channels=32, y_only=y_only, hr_dcn=hr_dcn, offset_prop=offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
    # model = MRCF.MRCF_simple_v13_nodcn(mid_channels=32, y_only=y_only, hr_dcn=hr_dcn, offset_prop=offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
    # model = MRCF.MRCF_simple_v15(mid_channels=32, y_only=y_only, hr_dcn=hr_dcn, offset_prop=offset_prop, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
    model = MRCF.MRCF_simple_v18(mid_channels=32, y_only=y_only, hr_dcn=hr_dcn, offset_prop=offset_prop, split_ratio=split_ratio, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)
    # model = MRCF.MRCF_simple_v18_nofv(mid_channels=32, y_only=y_only, hr_dcn=hr_dcn, offset_prop=offset_prop, split_ratio=split_ratio, spynet_pretrained='pretrained_models/fnet.pth', device=device).to(device)

    # model = MRCF.MRCF_simple(mid_channels=mid_channels, num_blocks=1, spynet_pretrained='pretrained_models/spynet_20210409-c6c1bd09.pth', device=device).to(device)
    # model = MRCF.MRCF_simple_v1_dcn2_v4_kai(mid_channels=32, y_only=y_only, spynet_pretrained='pretrained_models/250.pth', device=device).to(device)
    # model = MRCF.MRCF_simple_v1_dcn2_v4_kai(mid_channels=32, y_only=y_only, spynet_pretrained='pretrained_models/spynet_20210409-c6c1bd09.pth', device=device).to(device)
    # model = MRCF.MRCF_simple_v4(mid_channels=mid_channels, y_only=False, spynet_pretrained='pretrained_models/spynet_20210409-c6c1bd09.pth', device=device).to(device)
    # model = MRCF.MRCF_CRA_x8(mid_channels=mid_channels, num_blocks=1, spynet_pretrained='pretrained_models/spynet_20210409-c6c1bd09.pth', device=device).to(device)
    # model = MRCF.MRCF_CRA_x8_v1(mid_channels=mid_channels, num_blocks=1, spynet_pretrained='pretrained_models/spynet_20210409-c6c1bd09.pth', device=device).to(device)
    # model_spy = model.spynet
    # model_en_hr = model.encoder_hr
    # model_en_lr = model.encoder_lr
    # model = nn.Upsample(scale_factor=8, mode='bicubic', align_corners=False)
    # for k, v in model.named_parameters():
        # v.requires_grad_(False)

    # model = MRCF.ResidualBlocksWithInputConv(mid_channels * 2, mid_channels, 3).to(device)
    # model = nn.Sequential(
    #         nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
    #         nn.LeakyReLU(0.1, inplace=True),
    #         nn.Conv2d(mid_channels,  3, 3, 1 ,1)).to(device)

    # group = 1
    # model = nn.Sequential(
    #         nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True),
    #         nn.LeakyReLU(0.1, inplace=True),
    #         nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
    #         nn.LeakyReLU(0.1, inplace=True),
    #         nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
    #         nn.LeakyReLU(0.1, inplace=True)
    #     ).to(device)
    # dcn_offset = nn.Conv2d(mid_channels, group*2*3*3, 3, 1, 1).to(device)
    # dcn_mask = nn.Conv2d(mid_channels, group*1*3*3, 3, 1, 1).to(device)
    # dcn = DCNv2(mid_channels, mid_channels, 3, stride=1, padding=1, dilation=1, deformable_groups=group).to(device)
    # dcn_offset.weight.data.zero_()
    # dcn_offset.bias.data.zero_()
    # dcn_mask.weight.data.zero_()
    # dcn_mask.bias.data.zero_()
    # conv_identify(dcn.weight, dcn.bias)

    scale = 1
    # HR_h = 720
    # HR_w = 1280
    HR_h = 1080
    HR_w = 1920
    # HR_h = 512
    # HR_w = 512
    LR_h = HR_h // 8
    LR_w = HR_w // 8
    FV_h = 96
    FV_w = 96
    WP_h = 720
    WP_w = 720
    # WP_h = 1080
    # WP_w = 1920

    t = 5
    repeat_time = 30
    warm_up = 10
    infer_time = 0

    model.eval()
    # dcn_offset.eval()
    # dcn_mask.eval()
    # dcn.eval()
    
    with torch.no_grad():
        # print(torch.cuda.memory_allocated(0), '-2')
        # print(torch.cuda.max_memory_allocated(0), '-2')
        # torch.cuda.reset_max_memory_allocated(0)

        # x = torch.rand(1, mid_channels, HR_h, HR_w).cuda()
        # i = torch.rand(1, mid_channels * 2, HR_h//scale, HR_w//scale).cuda()

        # f = torch.rand(1, 2, HR_h//scale, HR_w//scale).cuda()
        # i = torch.rand(1, mid_channels*2+2, HR_h//scale, HR_w//scale).cuda()
        
        # f = torch.rand(1, 2, HR_h//scale, HR_w//scale).cuda()
        # x = torch.rand(1, mid_channels, HR_h//scale, HR_w//scale).cuda()

        # i = torch.rand(1, 3, HR_h//scale, HR_w//scale).cuda()
        # f = model(i)
        # o = dcn_offset(f)
        # o = 10. * torch.tanh(o)
        # m = dcn_mask(f)
        # m = torch.sigmoid(m)

        lr = torch.rand(1, t, 3, LR_h, LR_w).cuda()
        fv = torch.rand(1, t, 3, FV_h, FV_w).cuda()
        # mk = torch.ones(1, t, 1, HR_h, HR_w).cuda()

        # ref = torch.rand(1, 3, LR_h, LR_w).cuda()
        # sup = torch.rand(1, 3, LR_h, LR_w).cuda()

        # x_lr = torch.rand(1, 3, LR_h, LR_w).cuda()
        # x_fv = torch.rand(1, 6, FV_h, FV_w).cuda()

        # print(torch.cuda.memory_allocated(0), '-1')
        # print(torch.cuda.max_memory_allocated(0), '-1')
        # torch.cuda.reset_max_memory_allocated(0)

        for idx in range(repeat_time):
            if idx < warm_up:
                infer_time = 0
            torch.cuda.synchronize()
            # start_time = time.time()
            start.record()
            
            y = model(lr, fv, warp_size=(WP_h, WP_w))
            # y = model(lr, fv)
            # y = model(lr, fv, mk)
            # y = model_spy(ref, sup)
            
            # y0, y1, y2 = model_en_lr(x_lr, islr=True)
            # y0, y1, y2 = model_en_hr(x_fv, islr=True)
            
            # y = MRCF.flow_warp(x, f.permute(0, 2, 3, 1))
            
            # y = dcn(f, o, m)
            
            # y = model(i)
            # o = dcn_offset(y)
            # o = 10. * torch.tanh(o)
            # f = torch.cat((f[:, 1:2, :, :], f[:, 0:1, :, :]), dim=1)
            # f = f.repeat(1, o.size(1) // 2, 1, 1)
            # o = o + f
            # m = dcn_mask(y)
            # m = torch.sigmoid(m)
            
            # y = model(f)
            # y = model(x)
            # y = model(x_lr)
            
            # print(torch.cuda.memory_allocated(0), '0')
            # print(torch.cuda.max_memory_allocated(0), '0')
            # torch.cuda.reset_max_memory_allocated(0)
            end.record()
            torch.cuda.synchronize()
            # infer_time += (time.time() - start_time)
            infer_time += (start.elapsed_time(end)/1000)

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            # with record_function("model_inference"):
                # y = model(lr, fv, mk)

    print(y.shape, infer_time / (repeat_time - warm_up + 1) / t)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    # prof.export_chrome_trace('./mrcf_profile.json')
    # reporter.report(verbose=True)