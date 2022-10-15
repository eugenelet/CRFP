import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from dcn_v2 import DCNv2
from model import LTE

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def rgb2yuv(rgb):
    # rgb_ = rgb.permute(0,2,3,1)
    # A = torch.tensor([[0.299, -0.14714119,0.61497538],
                    #   [0.587, -0.28886916, -0.51496512],
                    #   [0.114, 0.43601035, -0.10001026]])
    # yuv = torch.tensordot(rgb_,A,1).transpose(0,2)
    r = rgb[:, 0, :, :]
    g = rgb[:, 1, :, :]
    b = rgb[:, 2, :, :]

    y =  0.299 * r + 0.587 * g + 0.114 * b
    # u = -0.147 * r - 0.289 * g + 0.436 * b
    # v =  0.615 * r - 0.515 * g - 0.100 * b
    # yuv = torch.stack((y,u,v), dim=1)
    return y.unsqueeze(1)

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=16,
                     stride=stride, padding=0, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def default_init_weights(module, scale=1):
    """Initialize network weights.
    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale

def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid[:h, :w, :] + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    
    return output

def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)

class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=True)
        self.act  = nn.ReLU()
    def forward(self, x):
        return self.conv(self.act(x))

class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.
    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x

class PixelUnShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.
    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 downsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.downsample_kernel = downsample_kernel
        self.downsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels // (scale_factor * scale_factor),
            # self.in_channels * (scale_factor * scale_factor),
            # self.out_channels,
            self.downsample_kernel,
            padding=(self.downsample_kernel - 1) // 2)
        self.init_weights()
        assert out_channels % (scale_factor * scale_factor) == 0

    def init_weights(self):
        """Initialize weights for PixelUnShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelUnShufflePack.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x = self.downsample_conv(x)
        x = pixel_unshuffle(x, self.scale_factor)
        return x

class PixelUnShufflePack_v2(nn.Module):
    """ Pixel Shuffle upsample layer.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.
    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 downsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.downsample_kernel = downsample_kernel
        self.downsample_conv = nn.Conv2d(
            self.in_channels * (scale_factor * scale_factor),
            self.out_channels,
            self.downsample_kernel,
            padding=(self.downsample_kernel - 1) // 2)
        self.init_weights()
        assert out_channels % (scale_factor * scale_factor) == 0

    def init_weights(self):
        """Initialize weights for PixelUnShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelUnShufflePack.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x = pixel_unshuffle(x, self.scale_factor)
        x = self.downsample_conv(x)
        return x

class DCN_module(nn.Module):
    def __init__(self, mid_channels=64, dg=16, dk=3, max_mag=10, repeat=False, pre_offset=False, interpolate='none', offset_only=False):
        super().__init__()
        self.mid_channels = mid_channels
        self.dg_num = dg
        self.dk = dk
        self.max_residue_magnitude = max_mag
        self.pre_offset = pre_offset
        self.repeat = repeat
        self.interpolate = interpolate
        self.offset_only = offset_only
        if pre_offset:
            if interpolate == 'bilinear':
                self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
            elif interpolate == 'pixelshuffle':
                self.upsample = PixelShufflePack(mid_channels*8, mid_channels, 4, upsample_kernel=3)
            self.conv_fuse = nn.Conv2d(mid_channels*2, mid_channels, 3, 1, 1)
            # self.init_channels = mid_channels*3+2
        # else:
            # self.init_channels = mid_channels*2+2
        self.init_channels = mid_channels*2+2

        self.dcn_block = nn.Sequential(
            nn.Conv2d(self.init_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        if self.repeat:
            self.dcn_offset = nn.Conv2d(mid_channels, (self.dg_num)*2, 3, 1, 1, bias=True)
            if self.offset_only:
                self.dcn_mask = nn.Conv2d(mid_channels, (self.dg_num)*1*self.dk*self.dk, 3, 1, 1, bias=True)
            else:
                self.dcn_mask = nn.Conv2d(mid_channels, (self.dg_num)*1, 3, 1, 1, bias=True)
        else:
            self.dcn_offset = nn.Conv2d(mid_channels, (self.dg_num)*2*self.dk*self.dk, 3, 1, 1, bias=True)
            self.dcn_mask = nn.Conv2d(mid_channels, (self.dg_num)*1*self.dk*self.dk, 3, 1, 1, bias=True)
        self.dcn = DCNv2(mid_channels, mid_channels, self.dk,
                         stride=1, padding=(self.dk-1)//2, dilation=1,
                         deformable_groups=self.dg_num)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.init_dcn()
    
    def forward(self, cur_x, pre_x, pre_x_aligned, flow, pre_offset=None):
        # if torch.is_tensor(pre_offset):
        #     if self.interpolate != 'none':
        #         pre_offset = self.upsample(pre_offset) * 2.
        #     pre_x_aligned = torch.cat([cur_x, pre_x_aligned, pre_offset, flow], dim=1)
        # else:
        #     pre_x_aligned = torch.cat([cur_x, pre_x_aligned, flow], dim=1)
        pre_x_aligned = torch.cat([cur_x, pre_x_aligned, flow], dim=1)
        pre_x_aligned = self.dcn_block(pre_x_aligned)
        if torch.is_tensor(pre_offset):
            if self.interpolate != 'none':
                pre_offset = self.upsample(pre_offset) * 2.
            pre_x_aligned = self.lrelu(self.conv_fuse(torch.cat([pre_x_aligned, pre_offset], dim=1)))
        offset = self.dcn_offset(pre_x_aligned)
        offset = self.max_residue_magnitude * torch.tanh(offset)
        mask = self.dcn_mask(pre_x_aligned)
        mask = torch.sigmoid(mask)
        if self.repeat:
            B, C, H, W = offset.size()
            offset = offset.view(B, 2, C//2, H, W)
            offset = offset + flow.flip(1).unsqueeze(2).repeat(1, 1, C//2, 1, 1)
            offset = offset.repeat(1, self.dk**2, 1, 1, 1).view(B, C*(self.dk**2), H, W)
            if not self.offset_only:
                mask = mask.repeat(1, self.dk**2, 1, 1)
        else:
            offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        pre_x = self.dcn(pre_x, offset, mask)
        
        return pre_x, pre_x_aligned

    def init_dcn(self):
        self.dcn_offset.weight.data.zero_()
        self.dcn_offset.bias.data.zero_()
        self.dcn_mask.weight.data.zero_()
        self.dcn_mask.bias.data.zero_()
        self.conv_identify(self.dcn.weight, self.dcn.bias)

    def conv_identify(self, weight, bias):
        weight.data.zero_()
        bias.data.zero_()
        o, i, h, w = weight.shape
        y = h//2
        x = w//2
        for p in range(i):
            for q in range(o):
                if p == q:
                    weight.data[q, p, y, x] = 1.0

class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels. [From EDVR]
    '''

    def __init__(self, nf=64, groups=8, kernel=3, max_mag=10):
        super(PCD_Align, self).__init__()

        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        
        # L3: level 3, 1/4 spatial size
        self.L3_dcnpack = DCN_module(nf, groups, kernel, max_mag)

        # L2: level 2, 1/2 spatial size
        self.L2_dcnpack = DCN_module(nf, groups, kernel, max_mag, True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # L1: level 1, original spatial size
        self.L1_dcnpack = DCN_module(nf, groups, kernel, max_mag, True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # Cascading DCN
        self.cas_dcnpack = DCN_module(nf, groups, kernel, max_mag)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, cur_x_lv1, pre_x_lv1, pre_x_aligned_lv1, flow_lv1):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        L1_fea = torch.cat((cur_x_lv1, pre_x_lv1, pre_x_aligned_lv1), dim=0)
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        cur_x_lv3, pre_x_lv3, pre_x_aligned_lv3 = torch.chunk(L3_fea, 3, dim=0)
        cur_x_lv2, pre_x_lv2, pre_x_aligned_lv2 = torch.chunk(L2_fea, 3, dim=0)

        flow_lv2 = F.interpolate(flow_lv1, scale_factor=0.5, mode='bilinear', align_corners=False)
        flow_lv3 = F.interpolate(flow_lv2, scale_factor=0.5, mode='bilinear', align_corners=False)

        # L3
        L3_fea, L3_offset = self.L3_dcnpack(cur_x_lv3, pre_x_lv3, pre_x_aligned_lv3, flow_lv3)
        L3_fea = self.lrelu(L3_fea)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)

        # L2
        L2_fea, L2_offset = self.L2_dcnpack(cur_x_lv2, pre_x_lv2, pre_x_aligned_lv2, flow_lv2, L3_offset)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)

        # L1
        L1_fea, _ = self.L1_dcnpack(cur_x_lv1, pre_x_lv1, pre_x_aligned_lv1, flow_lv1, L2_offset)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))

        # Cascading
        L1_fea, _ = self.cas_dcnpack(cur_x_lv1, L1_fea, L1_fea, flow_lv1)
        L1_fea = self.lrelu(L1_fea)
        # L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))

        return L1_fea

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
    ::
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.
        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class ResidualBlocks(nn.Module):
    """Residual blocks.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels=64, num_blocks=30):
        super().__init__()

        main = []
        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=in_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)

class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)

class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained, device):
        super().__init__()

        self.device = device
        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=True, logger=logger)
            model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
            model_state_dict = self.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.load_state_dict(model_state_dict, strict=True)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(0,len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow_ = flow_warp(supp[level],
                              flow_up.permute(0, 2, 3, 1),
                              padding_mode='border')
            in_cat = torch.cat([ref[level], flow_, flow_up], 1)
            out = self.basic_module[level](in_cat)
            flow = flow_up + out

        return flow

    def forward(self, ref, supp):
    # def forward(self, x):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        # ref, supp = x
        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)
        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow

class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            conv(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3),
            conv(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3),
            conv(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3),
            conv(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3),
            conv(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)

class FNet(nn.Module):
    """ Optical flow estimation network
    """

    def __init__(self, in_nc):
        super(FNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(2*in_nc, 32, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2))

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2))

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        self.flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True))

    def forward(self, x1, x2):
        """ Compute optical flow from x1 to x2
        """
        _, c, h, w = x1.size()
        out = self.encoder1(torch.cat([x1, x2], dim=1))
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = self.decoder1(out)
        out = self.decoder2(out)
        out = self.decoder3(out)
        out = torch.tanh(self.flow(out)) * 256
        out = F.interpolate(
                input=out,
                size=(h, w),
                mode='bilinear',
                align_corners=False)

        return out

class CRFP_simple(nn.Module):
    def __init__(self, device, mid_channels=16, y_only=False, hr_dcn=True, offset_prop=True, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels // 8
        self.dg_num = 8
        self.dk = 3
        self.max_residue_magnitude = 10

        self.y_only = y_only
        self.hr_dcn = hr_dcn
        self.offset_prop = offset_prop

        # optical flow network for feature alignment
        # self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)
        self.spynet = FNet(in_nc=3)
        self.spynet.load_state_dict(torch.load(spynet_pretrained))
        
        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        self.dcn_2 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        if self.hr_dcn:
            self.dcn_3 = DCN_module(self.last_channels, 1, self.dk, self.max_residue_magnitude, repeat=True, pre_offset=self.offset_prop, interpolate='pixelshuffle')
        else:
            self.dcn_3 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_single(self.last_channels)

        self.conv_tttf = conv3x3(self.last_channels * 2, self.last_channels)

        # propagation branches
        self.forward_resblocks_0 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_2 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        if self.hr_dcn:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv(
                self.last_channels*2, self.last_channels, 1)
        else:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv(
                mid_channels*2, mid_channels, 1)

        # downsample
        self.downsample = PixelUnShufflePack_v2(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # self.downsample = nn.Sequential(
        #                   nn.Conv2d(self.last_channels, mid_channels, 3, 1, 1),
        #                   nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False))

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)

        # self.upsample = nn.Sequential(
        #                 nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # self.upsample_post = nn.Sequential(
        #                 nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1),
        #                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False))

        # self.upsample = nn.Sequential(
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #                 nn.Conv2d(mid_channels, mid_channels, 3, 1, 1))
        # self.upsample_post = nn.Sequential(
        #                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        #                 nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1))

        if self.y_only:
            self.conv_last = nn.Conv2d(self.last_channels, 1, 3, 1, 1)
        else:
            self.conv_last = nn.Conv2d(self.last_channels, 3, 3, 1, 1)

        ### 8x settings
        self.img_downsample_4x = nn.Upsample(
            scale_factor=0.25, mode='bilinear', align_corners=False)
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_upsample_4x = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()

        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        # flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_backward = None
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, fvs, mks):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        # print(lrs.size(), mks.size())
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)
        
        ### forward-time propagation and upsampling
        outputs = []

        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv3_0 = lrs.new_zeros(n, self.last_channels, h*8, w*8)

        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        lrs_lv3_view = lrs_lv3.view(B, N, C, H*8, W*8)
        mks_float = mks.float()
        fvs = (fvs * mks_float + lrs_lv3_view * (1 - mks_float))

        B, N, C, H, W = fvs.size()
        _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        
        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            mk_cur = mks[:, i, :, :, :]
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                if self.hr_dcn:
                    flow_lv0 = self.img_upsample_8x(flow) * 8.

                if self.hr_dcn:
                    # feat_prop_lv3_0 = feat_prop_lv3
                    # feat_prop_lv3 = self.downsample(feat_prop_lv3)
                    # feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                    # feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))

                    feat_prop_lv3_0 = feat_prop_lv3
                    feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))
                    feat_prop_lv3_ = self.downsample(feat_prop_lv3_0_)
                    feat_prop_lv3 = self.downsample(feat_prop_lv3)
                else:
                    feat_prop_lv3 = self.downsample(feat_prop_lv3)
                    feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))

                #### v13
                # L0
                feat_prop_lv3_a, offset = self.dcn_0(feat_prop_lv0, feat_prop_lv3, feat_prop_lv3_, flow_lv3)
                if not self.offset_prop:
                    offset = None

                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)
                
                # L1
                feat_prop_lv3_a, offset = self.dcn_1(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_)

                # L2
                feat_prop_lv3_a, offset = self.dcn_2(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_)

                # L3
                if self.hr_dcn:
                    feat_prop_lv0_ = self.lrelu(self.upsample_post(feat_prop_lv0_))
                    feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0_, feat_prop_lv3_0, feat_prop_lv3_0_, flow_lv0, offset)
                else:
                    feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                
                feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3)
            else:
                #### v13
                # L0           
                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)

                # L1
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_)

                # L2
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_)

                # L3
                if self.hr_dcn:
                    feat_prop_lv0_ = self.lrelu(self.upsample_post(feat_prop_lv0_))
                    feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_0], dim=1)
                else:
                    feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3)

            if not self.hr_dcn:
                feat_prop_lv3 = self.lrelu(self.upsample_post(feat_prop_lv3))
            feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3 = mk_cur.float() * feat_prop_lv3_ + (1 - mk_cur.float()) * feat_prop_lv3
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            out = feat_prop_lv3

            out = self.conv_last(out)
            if self.y_only:
                base = self.img_upsample_8x(rgb2yuv(lr_cur))
            else:
                base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)

        return torch.stack(outputs, dim=1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=strict, logger=logger)
            model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
            model_state_dict = self.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.load_state_dict(model_state_dict, strict=strict)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class CRFP(nn.Module):
    def __init__(self, device, mid_channels=16, y_only=False, hr_dcn=True, offset_prop=True, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels // 8
        self.dg_num = 8
        self.dk = 3
        self.max_residue_magnitude = 10

        self.y_only = y_only
        self.hr_dcn = hr_dcn
        self.offset_prop = offset_prop

        # optical flow network for feature alignment
        # self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)
        self.spynet = FNet(in_nc=3)
        self.spynet.load_state_dict(torch.load(spynet_pretrained))
        
        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        self.dcn_2 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        if self.hr_dcn:
            self.dcn_3 = DCN_module(self.last_channels, 1, self.dk, self.max_residue_magnitude, repeat=True, pre_offset=self.offset_prop, interpolate='pixelshuffle')
        else:
            self.dcn_3 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_single(self.last_channels)

        self.conv_tttf = conv3x3(self.last_channels * 2, self.last_channels)

        # propagation branches
        self.forward_resblocks_0 = ResidualBlocksWithInputConv(
            mid_channels*3, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv(
            mid_channels*3, mid_channels, 1)
        self.forward_resblocks_2 = ResidualBlocksWithInputConv(
            mid_channels*3, mid_channels, 1)
        if self.hr_dcn:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv(
                self.last_channels*3, self.last_channels, 1)
        else:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv(
                mid_channels*3, mid_channels, 1)

        # downsample
        self.downsample = PixelUnShufflePack_v2(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # self.downsample = nn.Sequential(
        #                   nn.Conv2d(self.last_channels, mid_channels, 3, 1, 1),
        #                   nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False))

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)

        # self.upsample = nn.Sequential(
        #                 nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # self.upsample_post = nn.Sequential(
        #                 nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1),
        #                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False))

        # self.upsample = nn.Sequential(
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #                 nn.Conv2d(mid_channels, mid_channels, 3, 1, 1))
        # self.upsample_post = nn.Sequential(
        #                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        #                 nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1))

        if self.y_only:
            self.conv_last = nn.Conv2d(self.last_channels, 1, 3, 1, 1)
        else:
            self.conv_last = nn.Conv2d(self.last_channels, 3, 3, 1, 1)

        ### 8x settings
        self.img_downsample_4x = nn.Upsample(
            scale_factor=0.25, mode='bilinear', align_corners=False)
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_upsample_4x = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()

        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        # flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_backward = None
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, fvs, mks):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        # print(lrs.size(), mks.size())
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)
        
        ### forward-time propagation and upsampling
        outputs = []

        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv3_0 = lrs.new_zeros(n, self.last_channels, h*8, w*8)

        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        lrs_lv3_view = lrs_lv3.view(B, N, C, H*8, W*8)
        mks_float = mks.float()
        fvs = (fvs * mks_float + lrs_lv3_view * (1 - mks_float))

        B, N, C, H, W = fvs.size()
        _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        
        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            mk_cur = mks[:, i, :, :, :]
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                flow_lv0 = self.img_upsample_8x(flow) * 8.

                if self.hr_dcn:
                    # feat_prop_lv3_0 = feat_prop_lv3
                    # feat_prop_lv3 = self.downsample(feat_prop_lv3)
                    # feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                    # feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))

                    feat_prop_lv3_0 = feat_prop_lv3
                    feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))
                    feat_prop_lv3_ = self.downsample(feat_prop_lv3_0_)
                    feat_prop_lv3 = self.downsample(feat_prop_lv3)
                else:
                    feat_prop_lv3 = self.downsample(feat_prop_lv3)
                    feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                
                #### v15
                # L0
                feat_prop_lv3_a, offset = self.dcn_0(feat_prop_lv0, feat_prop_lv3, feat_prop_lv3_, flow_lv3)
                if not self.offset_prop:
                    offset = None

                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3_a, feat_prop_lv3_], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)
                
                # L1
                feat_prop_lv3_a, offset = self.dcn_1(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a, feat_prop_lv3_], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_)

                # L2
                feat_prop_lv3_a, offset = self.dcn_2(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a, feat_prop_lv3_], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_)

                # L3
                if self.hr_dcn:
                    feat_prop_lv0_ = self.lrelu(self.upsample_post(feat_prop_lv0_))
                    feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0_, feat_prop_lv3_0, feat_prop_lv3_0_, flow_lv0, offset)
                    feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_a, feat_prop_lv3_0_], dim=1)
                else:
                    feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                    feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_a, feat_prop_lv3_], dim=1)
                
                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3)
            else:
                #### v15
                # L0
                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)
                
                # L1
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_)

                # L2
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_)

                # L3
                if self.hr_dcn:
                    feat_prop_lv0_ = self.lrelu(self.upsample_post(feat_prop_lv0_))
                    feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_0, feat_prop_lv3_0], dim=1)
                else:
                    feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3], dim=1)

                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3)

            if not self.hr_dcn:
                feat_prop_lv3 = self.lrelu(self.upsample_post(feat_prop_lv3))            
            feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3 = mk_cur.float() * feat_prop_lv3_ + (1 - mk_cur.float()) * feat_prop_lv3
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            out = feat_prop_lv3

            out = self.conv_last(out)
            if self.y_only:
                base = self.img_upsample_8x(rgb2yuv(lr_cur))
            else:
                base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)

        return torch.stack(outputs, dim=1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=strict, logger=logger)
            model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
            model_state_dict = self.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.load_state_dict(model_state_dict, strict=strict)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class CRFP_DSV(nn.Module):
    def __init__(self, device, mid_channels=16, y_only=False, hr_dcn=True, offset_prop=True, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels // 8
        self.dg_num = 8
        self.dk = 3
        self.max_residue_magnitude = 10

        self.y_only = y_only
        self.hr_dcn = hr_dcn
        self.offset_prop = offset_prop
        self.split_ratio = 3

        # optical flow network for feature alignment
        # self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)
        self.spynet = FNet(in_nc=3)
        self.spynet.load_state_dict(torch.load(spynet_pretrained))
        
        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        self.dcn_2 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        if self.hr_dcn:
            self.dcn_3 = DCN_module(self.last_channels, 1, self.dk, self.max_residue_magnitude, repeat=True, pre_offset=self.offset_prop, interpolate='pixelshuffle')
        else:
            self.dcn_3 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_single(self.last_channels)

        self.conv_tttf = conv3x3(self.last_channels * 2, self.last_channels)

        # propagation branches
        self.forward_resblocks_0 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_2 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        if self.hr_dcn:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv(
                self.last_channels*2, self.last_channels, 1)
        else:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv(
                mid_channels*2, mid_channels, 1)

        # downsample
        self.downsample = PixelUnShufflePack_v2(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # self.downsample = nn.Sequential(
        #                   nn.Conv2d(self.last_channels, mid_channels, 3, 1, 1),
        #                   nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False))

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, (mid_channels * self.split_ratio) // 4, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            (mid_channels * self.split_ratio) // 4, self.last_channels, 4, upsample_kernel=3)

        # self.upsample = nn.Sequential(
        #                 nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # self.upsample_post = nn.Sequential(
        #                 nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1),
        #                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False))

        # self.upsample = nn.Sequential(
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #                 nn.Conv2d(mid_channels, mid_channels, 3, 1, 1))
        # self.upsample_post = nn.Sequential(
        #                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        #                 nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1))

        if self.y_only:
            self.conv_last = nn.Conv2d(self.last_channels, 1, 3, 1, 1)
        else:
            self.conv_last = nn.Conv2d(self.last_channels, 3, 3, 1, 1)

        ### 8x settings
        self.img_downsample_4x = nn.Upsample(
            scale_factor=0.25, mode='bilinear', align_corners=False)
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_upsample_4x = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()

        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        # flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_backward = None
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, fvs, mks):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        # print(lrs.size(), mks.size())
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)
        
        ### forward-time propagation and upsampling
        outputs = []

        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv3_0 = lrs.new_zeros(n, self.last_channels, h*8, w*8)

        feat_lv0 = lrs.new_zeros(n, (self.mid_channels * (4 - self.split_ratio)) // 4, h*2, w*2)
        feat_lv1 = lrs.new_zeros(n, (self.mid_channels * (4 - self.split_ratio)) // 4, h*2, w*2)
        feat_lv2 = lrs.new_zeros(n, (self.mid_channels * (4 - self.split_ratio)) // 4, h*2, w*2)

        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        lrs_lv3_view = lrs_lv3.view(B, N, C, H*8, W*8)
        mks_float = mks.float()
        fvs = (fvs * mks_float + lrs_lv3_view * (1 - mks_float))

        B, N, C, H, W = fvs.size()
        _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        
        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            mk_cur = mks[:, i, :, :, :]
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                flow_lv0 = self.img_upsample_8x(flow) * 8.

                feat_prop_lv3_0 = feat_prop_lv3
                feat_prop_lv3 = self.downsample(feat_prop_lv3)
                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))
                
                feat_mix = torch.cat((feat_lv0,
                                      feat_lv1,
                                      feat_lv2), dim=1)
                
                feat_mix = flow_warp(feat_mix, flow_lv3.permute(0, 2, 3, 1))
                
                feat_mix = torch.chunk(feat_mix, 3, dim=1)
                feat_lv0 = feat_mix[0]
                feat_lv1 = feat_mix[1]
                feat_lv2 = feat_mix[2]

                #### v18
                # L0      
                feat_prop_lv0 = torch.cat((feat_prop_lv0, feat_lv0), dim=1)
                feat_prop_lv3_a, offset = self.dcn_0(feat_prop_lv0, feat_prop_lv3, feat_prop_lv3_, flow_lv3)

                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3_a], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3_a, feat_lv0], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1) # 5 -> 4
                # feat_lv0 = feat_prop_lv0_[3]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:3], dim=1)
                feat_lv0 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)
                feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)
                
                # L1
                feat_prop_lv0_ = torch.cat((feat_prop_lv0_, feat_lv1), dim=1)
                feat_prop_lv3_a, offset = self.dcn_1(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a, feat_lv1], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_)
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1) # 5 -> 4
                # feat_lv1 = feat_prop_lv0_[3]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:3], dim=1)
                feat_lv1 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)
                feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)

                # L2
                feat_prop_lv0_ = torch.cat((feat_prop_lv0_, feat_lv2), dim=1)
                feat_prop_lv3_a, offset = self.dcn_2(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a, feat_lv2], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_)
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1) # 5 -> 4
                # feat_lv2 = feat_prop_lv0_[3]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:3], dim=1)
                feat_lv2 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)
                feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)

                # L3
                feat_prop_lv0_ = self.lrelu(self.upsample_post(feat_prop_lv0_))
                feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0_, feat_prop_lv3_0, feat_prop_lv3_0_, flow_lv0, offset)
                
                # feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_a, feat_lv3], dim=1)
                feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3)
                # feat_prop_lv3 = torch.chunk(feat_prop_lv3, 3, dim=1)
                # feat_lv3 = torch.cat(feat_prop_lv3[1:], dim=1)
                # feat_prop_lv3 = feat_prop_lv3[0]
            else:
                #### v18
                # L0
                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3, feat_lv0], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1)
                # feat_lv0 = feat_prop_lv0_[3]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:3], dim=1)
                feat_lv0 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)
                feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)
                
                # L1
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3, feat_lv1], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_)
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1)
                # feat_lv1 = feat_prop_lv0_[3]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:3], dim=1)
                feat_lv1 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)
                feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)

                # L2
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3, feat_lv2], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_)
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1)
                # feat_lv2 = feat_prop_lv0_[3]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:3], dim=1)
                feat_lv2 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)
                feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)

                # L3
                feat_prop_lv0_ = self.lrelu(self.upsample_post(feat_prop_lv0_))
                # feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_0, feat_lv3], dim=1)
                feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_0], dim=1)
                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3)
                # feat_prop_lv3 = torch.chunk(feat_prop_lv3, 3, dim=1)
                # feat_lv3 = torch.cat(feat_prop_lv3[1:], dim=1)
                # feat_prop_lv3 = feat_prop_lv3[0]

            feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3 = mk_cur.float() * feat_prop_lv3_ + (1 - mk_cur.float()) * feat_prop_lv3
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            out = feat_prop_lv3
            
            out = self.conv_last(out)
            if self.y_only:
                base = self.img_upsample_8x(rgb2yuv(lr_cur))
            else:
                base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)

        return torch.stack(outputs, dim=1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=strict, logger=logger)
            model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
            model_state_dict = self.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.load_state_dict(model_state_dict, strict=strict)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class BasicFVSR(nn.Module):
    def __init__(self, device, mid_channels=16, y_only=False, hr_dcn=True, offset_prop=True, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels // 8
        self.dg_num = 8
        self.dk = 3
        self.max_residue_magnitude = 10

        self.y_only = y_only
        self.hr_dcn = hr_dcn
        self.offset_prop = offset_prop

        # optical flow network for feature alignment
        # self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)
        self.spynet = FNet(in_nc=3)
        self.spynet.load_state_dict(torch.load(spynet_pretrained))
        
        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        self.dcn_2 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        if self.hr_dcn:
            self.dcn_3 = DCN_module(self.last_channels, 1, self.dk, self.max_residue_magnitude, repeat=True, pre_offset=self.offset_prop, interpolate='pixelshuffle')
        else:
            self.dcn_3 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr(mid_channels)

        self.conv_tttf = conv3x3(mid_channels * 2, mid_channels)

        # propagation branches
        self.forward_resblocks_0 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_2 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        if self.hr_dcn:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv(
                self.last_channels*2, self.last_channels, 1)
        else:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv(
                mid_channels*2, mid_channels, 1)

        # downsample
        self.downsample = PixelUnShufflePack_v2(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # self.downsample = nn.Sequential(
        #                   nn.Conv2d(self.last_channels, mid_channels, 3, 1, 1),
        #                   nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False))

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)

        # self.upsample = nn.Sequential(
        #                 nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # self.upsample_post = nn.Sequential(
        #                 nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1),
        #                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False))

        # self.upsample = nn.Sequential(
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #                 nn.Conv2d(mid_channels, mid_channels, 3, 1, 1))
        # self.upsample_post = nn.Sequential(
        #                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        #                 nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1))

        if self.y_only:
            self.conv_last = nn.Conv2d(self.last_channels, 1, 3, 1, 1)
        else:
            self.conv_last = nn.Conv2d(self.last_channels, 3, 3, 1, 1)

        ### 8x settings
        self.img_downsample_4x = nn.Upsample(
            scale_factor=0.25, mode='bilinear', align_corners=False)
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_upsample_4x = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()

        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        # flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_backward = None
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, fvs, mks):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        # print(lrs.size(), mks.size())
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)
        
        ### forward-time propagation and upsampling
        outputs = []

        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv3_0 = lrs.new_zeros(n, self.last_channels, h*8, w*8)

        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        lrs_lv3_view = lrs_lv3.view(B, N, C, H*8, W*8)
        mks_float = mks.float()
        fvs = (fvs * mks_float + lrs_lv3_view * (1 - mks_float))

        B, N, C, H, W = fvs.size()
        x_hr_lv3, _, _ = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        
        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            mk_cur = mks[:, i, :, :, :]
            mk_cur_lv2 = self.img_downsample_4x(mk_cur.float())
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)
            feat_prop_lv0_ = torch.cat([feat_prop_lv0, x_hr_lv3_cur], dim=1)
            feat_prop_lv0_ = self.conv_tttf(feat_prop_lv0_)
            feat_prop_lv0 = mk_cur_lv2.float() * feat_prop_lv0_ + (1 - mk_cur_lv2.float()) * feat_prop_lv0

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                if self.hr_dcn:
                    flow_lv0 = self.img_upsample_8x(flow) * 8.

                if self.hr_dcn:
                    # feat_prop_lv3_0 = feat_prop_lv3
                    # feat_prop_lv3 = self.downsample(feat_prop_lv3)
                    # feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                    # feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))

                    feat_prop_lv3_0 = feat_prop_lv3
                    feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))
                    feat_prop_lv3_ = self.downsample(feat_prop_lv3_0_)
                    feat_prop_lv3 = self.downsample(feat_prop_lv3)
                else:
                    feat_prop_mix = torch.cat((feat_prop_0, feat_prop_1, feat_prop_2, feat_prop_3), dim=1)
                    feat_prop_mix = flow_warp(feat_prop_mix, flow_lv3.permute(0, 2, 3, 1))
                    feat_prop_split = torch.split(feat_prop_mix, self.mid_channels, dim=1)
                    feat_prop_0_ = feat_prop_split[0]
                    feat_prop_1_ = feat_prop_split[1]
                    feat_prop_2_ = feat_prop_split[2]
                    feat_prop_3_ = feat_prop_split[3]

                #### v13
                # L0
                feat_prop_0, offset = self.dcn_0(feat_prop_lv0, feat_prop_0, feat_prop_0_, flow_lv3)
                if not self.offset_prop:
                    offset = None

                feat_prop_0 = torch.cat([feat_prop_lv0, feat_prop_0], dim=1)
                feat_prop_0 = self.forward_resblocks_0(feat_prop_0)
                
                # L1
                feat_prop_1, offset = self.dcn_1(feat_prop_0, feat_prop_1, feat_prop_1_, flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_1 = torch.cat([feat_prop_0, feat_prop_1], dim=1)
                feat_prop_1 = self.forward_resblocks_1(feat_prop_1)

                # L2
                feat_prop_2, offset = self.dcn_2(feat_prop_1, feat_prop_2, feat_prop_2_, flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_2 = torch.cat([feat_prop_1, feat_prop_2], dim=1)
                feat_prop_2 = self.forward_resblocks_2(feat_prop_2)

                # L3
                if self.hr_dcn:
                    feat_prop_2_ = self.lrelu(self.upsample_post(feat_prop_2))
                    feat_prop_3, _ = self.dcn_3(feat_prop_2_, feat_prop_3, feat_prop_3_, flow_lv0, offset)
                else:
                    feat_prop_3, _ = self.dcn_3(feat_prop_2, feat_prop_3, feat_prop_3_, flow_lv3, offset)
                
                feat_prop_3 = torch.cat([feat_prop_2, feat_prop_3], dim=1)
                feat_prop_3 = self.forward_resblocks_3(feat_prop_3)
            else:
                #### v13
                # L0           
                feat_prop_0 = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_0 = self.forward_resblocks_0(feat_prop_0)

                # L1
                feat_prop_1 = torch.cat([feat_prop_0, feat_prop_lv3], dim=1)
                feat_prop_1 = self.forward_resblocks_1(feat_prop_1)

                # L2
                feat_prop_2 = torch.cat([feat_prop_1, feat_prop_lv3], dim=1)
                feat_prop_2 = self.forward_resblocks_2(feat_prop_2)

                # L3
                if self.hr_dcn:
                    feat_prop_2_ = self.lrelu(self.upsample_post(feat_prop_2))
                    feat_prop_3 = torch.cat([feat_prop_2_, feat_prop_3], dim=1)
                else:
                    feat_prop_3 = torch.cat([feat_prop_2, feat_prop_lv3], dim=1)
                feat_prop_3 = self.forward_resblocks_3(feat_prop_3)

            if self.hr_dcn:
                feat_prop_lv3 = feat_prop_3
            else:
                feat_prop_lv3 = self.lrelu(self.upsample_post(feat_prop_3))
            out = feat_prop_lv3

            out = self.conv_last(out)
            if self.y_only:
                base = self.img_upsample_8x(rgb2yuv(lr_cur))
            else:
                base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)

        return torch.stack(outputs, dim=1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=strict, logger=logger)
            model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
            model_state_dict = self.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.load_state_dict(model_state_dict, strict=strict)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class CRFP_simple_noDCN(nn.Module):
    def __init__(self, device, mid_channels=16, y_only=False, hr_dcn=True, offset_prop=True, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels // 8
        self.dg_num = 8
        self.dk = 3
        self.max_residue_magnitude = 10

        self.y_only = y_only
        self.hr_dcn = hr_dcn
        self.offset_prop = offset_prop

        # optical flow network for feature alignment
        # self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)
        self.spynet = FNet(in_nc=3)
        self.spynet.load_state_dict(torch.load(spynet_pretrained))
        
        # self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude)
        # self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        # self.dcn_2 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        # if self.hr_dcn:
        #     self.dcn_3 = DCN_module(self.last_channels, 1, self.dk, self.max_residue_magnitude, repeat=True, pre_offset=self.offset_prop, interpolate='pixelshuffle')
        # else:
        #     self.dcn_3 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')

        self.dcn_0 = nn.Sequential(
            nn.Conv2d(mid_channels*2 + 2, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_1 = nn.Sequential(
            nn.Conv2d(mid_channels*2 + 2, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_2 = nn.Sequential(
            nn.Conv2d(mid_channels*2 + 2, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_3 = nn.Sequential(
            nn.Conv2d(mid_channels*2 + 2, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_single(self.last_channels)

        self.conv_tttf = conv3x3(self.last_channels * 2, self.last_channels)

        # propagation branches
        self.forward_resblocks_0 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_2 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        if self.hr_dcn:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv(
                self.last_channels*2, self.last_channels, 1)
        else:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv(
                mid_channels*2, mid_channels, 1)

        # downsample
        self.downsample = PixelUnShufflePack_v2(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # self.downsample = nn.Sequential(
        #                   nn.Conv2d(self.last_channels, mid_channels, 3, 1, 1),
        #                   nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False))

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)

        # self.upsample = nn.Sequential(
        #                 nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # self.upsample_post = nn.Sequential(
        #                 nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1),
        #                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False))

        # self.upsample = nn.Sequential(
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #                 nn.Conv2d(mid_channels, mid_channels, 3, 1, 1))
        # self.upsample_post = nn.Sequential(
        #                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        #                 nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1))

        if self.y_only:
            self.conv_last = nn.Conv2d(self.last_channels, 1, 3, 1, 1)
        else:
            self.conv_last = nn.Conv2d(self.last_channels, 3, 3, 1, 1)

        ### 8x settings
        self.img_downsample_4x = nn.Upsample(
            scale_factor=0.25, mode='bilinear', align_corners=False)
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_upsample_4x = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()

        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        # flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_backward = None
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, fvs, mks):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        # print(lrs.size(), mks.size())
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)
        
        ### forward-time propagation and upsampling
        outputs = []

        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv3_0 = lrs.new_zeros(n, self.last_channels, h*8, w*8)

        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        lrs_lv3_view = lrs_lv3.view(B, N, C, H*8, W*8)
        mks_float = mks.float()
        fvs = (fvs * mks_float + lrs_lv3_view * (1 - mks_float))

        B, N, C, H, W = fvs.size()
        _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        
        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            mk_cur = mks[:, i, :, :, :]
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                if self.hr_dcn:
                    flow_lv0 = self.img_upsample_8x(flow) * 8.

                if self.hr_dcn:
                    # feat_prop_lv3_0 = feat_prop_lv3
                    # feat_prop_lv3 = self.downsample(feat_prop_lv3)
                    # feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                    # feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))

                    feat_prop_lv3_0 = feat_prop_lv3
                    feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))
                    feat_prop_lv3_ = self.downsample(feat_prop_lv3_0_)
                    feat_prop_lv3 = self.downsample(feat_prop_lv3)
                else:
                    feat_prop_lv3 = self.downsample(feat_prop_lv3)
                    feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))

                #### v13
                # L0
                # feat_prop_lv3_a, offset = self.dcn_0(feat_prop_lv0, feat_prop_lv3, feat_prop_lv3_, flow_lv3)
                feat_prop_lv3_a = self.dcn_0(torch.cat((feat_prop_lv0, feat_prop_lv3_, flow_lv3), dim=1))
                if not self.offset_prop:
                    offset = None

                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)
                
                # L1
                # feat_prop_lv3_a, offset = self.dcn_1(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                feat_prop_lv3_a = self.dcn_1(torch.cat((feat_prop_lv0_, feat_prop_lv3_, flow_lv3), dim=1))
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_)

                # L2
                # feat_prop_lv3_a, offset = self.dcn_2(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                feat_prop_lv3_a = self.dcn_2(torch.cat((feat_prop_lv0_, feat_prop_lv3_, flow_lv3), dim=1))
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_)

                # L3
                if self.hr_dcn:
                    feat_prop_lv0_ = self.lrelu(self.upsample_post(feat_prop_lv0_))
                    feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0_, feat_prop_lv3_0, feat_prop_lv3_0_, flow_lv0, offset)
                else:
                    # feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                    feat_prop_lv3_a = self.dcn_3(torch.cat((feat_prop_lv0_, feat_prop_lv3_, flow_lv3), dim=1))
                
                feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3)
            else:
                #### v13
                # L0           
                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)

                # L1
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_)

                # L2
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_)

                # L3
                if self.hr_dcn:
                    feat_prop_lv0_ = self.lrelu(self.upsample_post(feat_prop_lv0_))
                    feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_0], dim=1)
                else:
                    feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3)

            if not self.hr_dcn:
                feat_prop_lv3 = self.lrelu(self.upsample_post(feat_prop_lv3))
            feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3 = mk_cur.float() * feat_prop_lv3_ + (1 - mk_cur.float()) * feat_prop_lv3
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            out = feat_prop_lv3

            out = self.conv_last(out)
            if self.y_only:
                base = self.img_upsample_8x(rgb2yuv(lr_cur))
            else:
                base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)

        return torch.stack(outputs, dim=1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=strict, logger=logger)
            model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
            model_state_dict = self.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.load_state_dict(model_state_dict, strict=strict)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class CRFP_DSV_CRA(nn.Module):
    def __init__(self, device, mid_channels=16, y_only=False, hr_dcn=True, offset_prop=True, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels // 8
        self.dg_num = 8
        self.dk = 3
        self.max_residue_magnitude = 10

        self.y_only = y_only
        self.hr_dcn = hr_dcn
        self.offset_prop = offset_prop
        self.split_ratio = 3

        # optical flow network for feature alignment
        # self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)
        self.spynet = FNet(in_nc=3)
        self.spynet.load_state_dict(torch.load(spynet_pretrained))
        
        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        self.dcn_2 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        if self.hr_dcn:
            self.dcn_3 = DCN_module(self.last_channels, 1, self.dk, self.max_residue_magnitude, repeat=True, pre_offset=self.offset_prop, interpolate='pixelshuffle')
        else:
            self.dcn_3 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        # self.encoder_hr = LTE.LTE_simple_hr_single(self.last_channels)
        self.encoder_hr = LTE.LTE_simple_hr_ps(self.last_channels)

        self.conv_tttf = conv3x3(self.last_channels * 2, self.last_channels)
        self.conv_tttf_0 = conv3x3(mid_channels + self.last_channels * 4, mid_channels)
        self.conv_tttf_1 = conv3x3(mid_channels + self.last_channels * 4, mid_channels)
        self.conv_tttf_2 = conv3x3(mid_channels + self.last_channels * 4, mid_channels)

        # propagation branches
        self.forward_resblocks_0 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_2 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        if self.hr_dcn:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv(
                self.last_channels*2, self.last_channels, 1)
        else:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv(
                mid_channels*2, mid_channels, 1)

        # downsample
        self.downsample = PixelUnShufflePack_v2(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # self.downsample = nn.Sequential(
        #                   nn.Conv2d(self.last_channels, mid_channels, 3, 1, 1),
        #                   nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False))

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, (mid_channels * self.split_ratio) // 4, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            (mid_channels * self.split_ratio) // 4, self.last_channels, 4, upsample_kernel=3)

        # self.upsample = nn.Sequential(
        #                 nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # self.upsample_post = nn.Sequential(
        #                 nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1),
        #                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False))

        # self.upsample = nn.Sequential(
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #                 nn.Conv2d(mid_channels, mid_channels, 3, 1, 1))
        # self.upsample_post = nn.Sequential(
        #                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        #                 nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1))

        if self.y_only:
            self.conv_last = nn.Conv2d(self.last_channels, 1, 3, 1, 1)
        else:
            self.conv_last = nn.Conv2d(self.last_channels, 3, 3, 1, 1)

        ### 8x settings
        self.img_downsample_4x = nn.Upsample(
            scale_factor=0.25, mode='bilinear', align_corners=False)
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_upsample_4x = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()

        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        # flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_backward = None
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, fvs, mks):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        # print(lrs.size(), mks.size())
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)
        
        ### forward-time propagation and upsampling
        outputs = []

        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv3_0 = lrs.new_zeros(n, self.last_channels, h*8, w*8)

        feat_lv0 = lrs.new_zeros(n, (self.mid_channels * (4 - self.split_ratio)) // 4, h*2, w*2)
        feat_lv1 = lrs.new_zeros(n, (self.mid_channels * (4 - self.split_ratio)) // 4, h*2, w*2)
        feat_lv2 = lrs.new_zeros(n, (self.mid_channels * (4 - self.split_ratio)) // 4, h*2, w*2)

        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        lrs_lv3_view = lrs_lv3.view(B, N, C, H*8, W*8)
        mks_float = mks.float()
        fvs = (fvs * mks_float + lrs_lv3_view * (1 - mks_float))

        B, N, C, H, W = fvs.size()
        x_hr_lv0, x_hr_lv1, x_hr_lv2, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1))

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv2.size()
        x_hr_lv2 = x_hr_lv2.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv1.size()
        x_hr_lv1 = x_hr_lv1.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv0.size()
        x_hr_lv0 = x_hr_lv0.contiguous().view(B, N, C, H, W)

        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            x_hr_lv2_cur = x_hr_lv2[:, i, :, :, :]
            x_hr_lv1_cur = x_hr_lv1[:, i, :, :, :]
            x_hr_lv0_cur = x_hr_lv0[:, i, :, :, :]
            mk_cur = mks[:, i, :, :, :]
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)
            mk_cur_lv2 = self.img_downsample_4x(mk_cur.float())

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                flow_lv0 = self.img_upsample_8x(flow) * 8.

                feat_prop_lv3_0 = feat_prop_lv3
                feat_prop_lv3 = self.downsample(feat_prop_lv3)
                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))
                
                feat_mix = torch.cat((feat_lv0,
                                      feat_lv1,
                                      feat_lv2), dim=1)
                
                feat_mix = flow_warp(feat_mix, flow_lv3.permute(0, 2, 3, 1))
                
                feat_mix = torch.chunk(feat_mix, 3, dim=1)
                feat_lv0 = feat_mix[0]
                feat_lv1 = feat_mix[1]
                feat_lv2 = feat_mix[2]

                #### v18
                # L0      
                feat_prop_lv0 = torch.cat((feat_prop_lv0, feat_lv0), dim=1)
                feat_prop_lv3_a, offset = self.dcn_0(feat_prop_lv0, feat_prop_lv3, feat_prop_lv3_, flow_lv3)

                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3_a], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3_a, feat_lv0], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)
                feat_prop_lv0 = torch.cat([feat_prop_lv0_, x_hr_lv0_cur], dim=1)
                feat_prop_lv0 = self.conv_tttf_0(feat_prop_lv0)
                feat_prop_lv0_ = mk_cur_lv2 * feat_prop_lv0 + (1 - mk_cur_lv2) * feat_prop_lv0_
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1) # 5 -> 4
                # feat_lv0 = feat_prop_lv0_[3]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:3], dim=1)
                feat_lv0 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)
                feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)
                
                # L1
                feat_prop_lv0_ = torch.cat((feat_prop_lv0_, feat_lv1), dim=1)
                feat_prop_lv3_a, offset = self.dcn_1(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a, feat_lv1], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_)
                feat_prop_lv0 = torch.cat([feat_prop_lv0_, x_hr_lv1_cur], dim=1)
                feat_prop_lv0 = self.conv_tttf_1(feat_prop_lv0)
                feat_prop_lv0_ = mk_cur_lv2 * feat_prop_lv0 + (1 - mk_cur_lv2) * feat_prop_lv0_
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1) # 5 -> 4
                # feat_lv1 = feat_prop_lv0_[3]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:3], dim=1)
                feat_lv1 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)
                feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)

                # L2
                feat_prop_lv0_ = torch.cat((feat_prop_lv0_, feat_lv2), dim=1)
                feat_prop_lv3_a, offset = self.dcn_2(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a, feat_lv2], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_)
                feat_prop_lv0 = torch.cat([feat_prop_lv0_, x_hr_lv2_cur], dim=1)
                feat_prop_lv0 = self.conv_tttf_2(feat_prop_lv0)
                feat_prop_lv0_ = mk_cur_lv2 * feat_prop_lv0 + (1 - mk_cur_lv2) * feat_prop_lv0_
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1) # 5 -> 4
                # feat_lv2 = feat_prop_lv0_[3]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:3], dim=1)
                feat_lv2 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)
                feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)

                # L3
                feat_prop_lv0_ = self.lrelu(self.upsample_post(feat_prop_lv0_))
                feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0_, feat_prop_lv3_0, feat_prop_lv3_0_, flow_lv0, offset)
                
                # feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_a, feat_lv3], dim=1)
                feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3)
                # feat_prop_lv3 = torch.chunk(feat_prop_lv3, 3, dim=1)
                # feat_lv3 = torch.cat(feat_prop_lv3[1:], dim=1)
                # feat_prop_lv3 = feat_prop_lv3[0]
            else:
                #### v18
                # L0
                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3, feat_lv0], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)
                feat_prop_lv0 = torch.cat([feat_prop_lv0_, x_hr_lv0_cur], dim=1)
                feat_prop_lv0 = self.conv_tttf_0(feat_prop_lv0)
                feat_prop_lv0_ = mk_cur_lv2 * feat_prop_lv0 + (1 - mk_cur_lv2) * feat_prop_lv0_
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1)
                # feat_lv0 = feat_prop_lv0_[3]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:3], dim=1)
                feat_lv0 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)
                feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)
                
                # L1
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3, feat_lv1], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_)
                feat_prop_lv0 = torch.cat([feat_prop_lv0_, x_hr_lv1_cur], dim=1)
                feat_prop_lv0 = self.conv_tttf_1(feat_prop_lv0)
                feat_prop_lv0_ = mk_cur_lv2 * feat_prop_lv0 + (1 - mk_cur_lv2) * feat_prop_lv0_
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1)
                # feat_lv1 = feat_prop_lv0_[3]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:3], dim=1)
                feat_lv1 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)
                feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)

                # L2
                feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3, feat_lv2], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_)
                feat_prop_lv0 = torch.cat([feat_prop_lv0_, x_hr_lv2_cur], dim=1)
                feat_prop_lv0 = self.conv_tttf_2(feat_prop_lv0)
                feat_prop_lv0_ = mk_cur_lv2 * feat_prop_lv0 + (1 - mk_cur_lv2) * feat_prop_lv0_
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1)
                # feat_lv2 = feat_prop_lv0_[3]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:3], dim=1)
                feat_lv2 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)
                feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)

                # L3
                feat_prop_lv0_ = self.lrelu(self.upsample_post(feat_prop_lv0_))
                # feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_0, feat_lv3], dim=1)
                feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_0], dim=1)
                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3)
                # feat_prop_lv3 = torch.chunk(feat_prop_lv3, 3, dim=1)
                # feat_lv3 = torch.cat(feat_prop_lv3[1:], dim=1)
                # feat_prop_lv3 = feat_prop_lv3[0]

            feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3 = mk_cur.float() * feat_prop_lv3_ + (1 - mk_cur.float()) * feat_prop_lv3
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            out = feat_prop_lv3
            
            out = self.conv_last(out)
            if self.y_only:
                base = self.img_upsample_8x(rgb2yuv(lr_cur))
            else:
                base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)

        return torch.stack(outputs, dim=1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=strict, logger=logger)
            model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
            model_state_dict = self.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.load_state_dict(model_state_dict, strict=strict)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')