import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from memory_profiler import profile
from dcn_v2 import DCNv2
from model import LTE

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

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

grid_y, grid_x = torch.meshgrid(torch.arange(0, 1080), torch.arange(0, 1920))
grid = torch.stack((grid_x, grid_y), 2).to(torch.device('cuda:0')) # (w, h, 2)
grid.requires_grad = False

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
    # a = time.time()
    # torch.cuda.synchronize()
    # start.record()

    # grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    # grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    # grid.requires_grad = False

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
    
    # end.record()
    # torch.cuda.synchronize()
    # print(time.time() - a, 'cpu -> gpu')
    # print(start.elapsed_time(end) / 1000, 'torch')
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

class DCN_module(nn.Module):
    def __init__(self, mid_channels=64, dg=16, dk=3, max_mag=10, repeat=False, pre_offset=False, interpolate='none'):
        super().__init__()
        self.mid_channels = mid_channels
        self.dg_num = dg
        self.dk = dk
        self.max_residue_magnitude = max_mag
        self.pre_offset = pre_offset
        self.repeat = repeat
        self.interpolate = interpolate
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

    def __init__(self, nf=64, groups=8, kernel=3):
        super(PCD_Align, self).__init__()

        self.fea_L2_conv1 = nn.Conv2d(nf, nf*2, 3, 2, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf*2, nf*4, 3, 2, 1, bias=True)
        
        # L3: level 3, 1/4 spatial size
        self.L3_dcnpack = DCN_module(nf, groups, kernel)

        # L2: level 2, 1/2 spatial size
        self.L2_dcnpack = DCN_module(nf, groups, kernel, True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # L1: level 1, original spatial size
        self.L1_dcnpack = DCN_module(nf, groups, kernel)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # Cascading DCN
        self.cas_dcnpack = DCN_module(nf, groups, kernel)

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
        L3_fea, L3_offset = self.lrelu(self.L3_dcnpack(cur_x_lv3, pre_x_lv3, pre_x_aligned_lv3, flow_lv3))
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
        x = F.pixel_unshuffle(x, self.scale_factor)
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
        x = F.pixel_unshuffle(x, self.scale_factor)
        x = self.downsample_conv(x)
        return x

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
        self.conv1 = nn.Conv2d(mid_channels, mid_channels//2, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels//2, mid_channels, 3, 1, 1, bias=True)
        # self.conv1 = nn.Sequential(
        #              nn.Conv2d(mid_channels, mid_channels//2, 3, 1, 1, bias=True, groups=8),
        #              nn.Conv2d(mid_channels//2, mid_channels//2, 5, 1, 2, bias=True, groups=mid_channels//2),
        #              nn.Conv2d(mid_channels//2, mid_channels, 3, 1, 1, bias=True, groups=8))
        # self.conv2 = nn.Sequential(
        #              nn.Conv2d(mid_channels, mid_channels//2, 3, 1, 1, bias=True, groups=8),
        #              nn.Conv2d(mid_channels//2, mid_channels//2, 5, 1, 2, bias=True, groups=mid_channels//2),
        #              nn.Conv2d(mid_channels//2, mid_channels, 3, 1, 1, bias=True, groups=8))

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
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(in_channels // 3, out_channels, 3, 1, 1, bias=True)
        # main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat1, feat2=None):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        if torch.is_tensor(feat2):
            N, C, H, W = feat1.size()
            o1 = self.conv1(feat1)
            feat = self.conv2(feat2)
            feat[:, :, :H, :W] = o1
        else:
            feat = self.conv1(feat1)
        return self.main(feat)

class ResidualBlocksWithInputConv_v2(nn.Module):
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, 3, 1, 1, bias=True)
        # main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat1, feat2=None):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        if torch.is_tensor(feat2):
            N, C, H, W = feat1.size()
            o1 = self.conv1(feat1)
            feat = self.conv2(feat2)
            feat[:, :, :H, :W] = o1
        else:
            feat = self.conv1(feat1)
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

class MRCF_x4(nn.Module):
    def __init__(self, device, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.dg_num = 16
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

        self.dcn_pre_lv0 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv1 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv2 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv3 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)

        self.dcn_block_lv0 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_block_lv1 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_block_lv2 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_block_lv3 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_offset_lv0 = nn.Conv2d(mid_channels, self.dg_num*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv0 = nn.Conv2d(mid_channels, self.dg_num*1*3*3, 3, 1, 1, bias=True)

        self.dcn_offset_lv1 = nn.Conv2d(mid_channels, self.dg_num*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv1 = nn.Conv2d(mid_channels, self.dg_num*1*3*3, 3, 1, 1, bias=True)

        self.dcn_offset_lv2 = nn.Conv2d(mid_channels, (self.dg_num//4)*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv2 = nn.Conv2d(mid_channels, (self.dg_num//4)*1*3*3, 3, 1, 1, bias=True)

        self.dcn_offset_lv3 = nn.Conv2d(mid_channels, (self.dg_num//16)*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv3 = nn.Conv2d(mid_channels, (self.dg_num//16)*1*3*3, 3, 1, 1, bias=True)
        
        self.dcn_lv0 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num)
        self.dcn_lv1 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num)
        self.dcn_lv2 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num//4)
        self.dcn_lv3 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num//16)
        self.init_dcn()

        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr(mid_channels)
    
        self.conv_tttf_lv3 = conv3x3(64 + 64, mid_channels)

        # propagation branches
        self.forward_resblocks_lv0 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 3)
        self.forward_resblocks_lv1 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 3)
        self.forward_resblocks_lv2 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)
        self.forward_resblocks_lv3 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)

        # upsample
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        
        self.conv_hr_lv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last_lv3 = nn.Conv2d(64, 3, 3, 1, 1)

        ### 4x settings
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_downsample_2x = nn.Upsample(
            scale_factor=0.5, mode='bilinear', align_corners=False)
        self.img_upsample_4x = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_downsample_4x = nn.Upsample(
            scale_factor=0.25, mode='bilinear', align_corners=False)

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

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, fvs, mks):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lrs.size()

        ### compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)
        
        ### forward-time propagation and upsampling
        outputs = []
        
        feat_prop_lv0 = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_lv1 = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_lv2 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*4, w*4)

        ### texture transfer
        B, N, C, H, W = mks.size()
        mk_lv3 = mks.float()
        mk_lv2 = self.img_downsample_2x(mk_lv3.view(B*N, 1, H, W)).view(B, N, 1, H//2, W//2)
        mk_lv1 = self.img_downsample_2x(mk_lv2.view(B*N, 1, H//2, W//2)).view(B, N, 1, H//4, W//4)
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_2x(lrs_lv0)
        lrs_lv3 = self.img_upsample_2x(lrs_lv3)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        fvs = fvs * mks.float() + lrs_lv3.view(B, N, C, H*4, W*4) * (1 - mks.float())
        B, N, C, H, W = fvs.size()
        x_hr_lv1, x_hr_lv2, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv1.size()
        x_hr_lv1 = x_hr_lv1.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv2.size()
        x_hr_lv2 = x_hr_lv2.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)

        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]

            x_hr_lv1_cur = x_hr_lv1[:, i, :, :, :]
            x_hr_lv2_cur = x_hr_lv2[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            mk_cur_lv3 = mk_lv3[:, i, :, :, :]
            mk_cur_lv2 = mk_lv2[:, i, :, :, :]
            mk_cur_lv1 = mk_lv1[:, i, :, :, :]

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]

                flow_lv0 = flow
                flow_lv1 = flow_lv0
                flow_lv2 = self.img_upsample_2x(flow_lv1)
                flow_lv3 = self.img_upsample_2x(flow_lv2)

                feat_prop_lv3 = self.feat_prop_lv3
                feat_prop_lv2 = self.img_downsample_2x(feat_prop_lv3)
                feat_prop_lv1 = self.img_downsample_2x(feat_prop_lv2)
                feat_prop_lv0 = feat_prop_lv1

                feat_prop_lv0_ = flow_warp(feat_prop_lv0, flow_lv0.permute(0, 2, 3, 1))
                feat_prop_lv1_ = feat_prop_lv0_
                feat_prop_lv2_ = flow_warp(feat_prop_lv2, flow_lv2.permute(0, 2, 3, 1))
                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))

                feat_prop_lv0_ = torch.cat([x_lr_lv0_cur, feat_prop_lv0_, flow_lv0], dim=1)
                feat_prop_lv0_ = self.dcn_pre_lv0(feat_prop_lv0_)
                feat_prop_lv0_ = self.dcn_block_lv0(feat_prop_lv0_)
                feat_offset_lv0 = self.dcn_offset_lv0(feat_prop_lv0_)
                feat_offset_lv0 = self.max_residue_magnitude * torch.tanh(feat_offset_lv0)
                feat_offset_lv0 = feat_offset_lv0 + flow_lv0.flip(1).repeat(1, feat_offset_lv0.size(1) // 2, 1, 1)
                feat_mask_lv0 = self.dcn_mask_lv0(feat_prop_lv0_)
                feat_mask_lv0 = torch.sigmoid(feat_mask_lv0)
                feat_prop_lv0 = self.dcn_lv0(feat_prop_lv0, feat_offset_lv0, feat_mask_lv0)
                
                feat_prop_lv0 = torch.cat([x_lr_lv0_cur, feat_prop_lv0], dim=1)
                feat_prop_lv0 = self.forward_resblocks_lv0(feat_prop_lv0)
                feat_prop_lv0 = self.lrelu(feat_prop_lv0)

                feat_prop_lv1_ = torch.cat([feat_prop_lv0, feat_prop_lv1_, flow_lv1], dim=1)
                feat_prop_lv1_ = self.dcn_pre_lv1(feat_prop_lv1_)
                feat_prop_lv1_ = self.dcn_block_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.dcn_offset_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.max_residue_magnitude * torch.tanh(feat_offset_lv1)
                feat_offset_lv1 = feat_offset_lv1 + flow_lv1.flip(1).repeat(1, feat_offset_lv1.size(1) // 2, 1, 1)
                feat_mask_lv1 = self.dcn_mask_lv1(feat_prop_lv1_)
                feat_mask_lv1 = torch.sigmoid(feat_mask_lv1)
                feat_prop_lv1 = self.dcn_lv1(feat_prop_lv1, feat_offset_lv1, feat_mask_lv1)

                feat_prop_lv1 = torch.cat([feat_prop_lv0, feat_prop_lv1], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))

                feat_prop_lv2_ = torch.cat([feat_prop_lv1, feat_prop_lv2_, flow_lv2], dim=1)
                feat_prop_lv2_ = self.dcn_pre_lv2(feat_prop_lv2_)
                feat_prop_lv2_ = self.dcn_block_lv2(feat_prop_lv2_)
                feat_offset_lv2 = self.dcn_offset_lv2(feat_prop_lv2_)
                feat_offset_lv2 = self.max_residue_magnitude * torch.tanh(feat_offset_lv2)
                feat_offset_lv2 = feat_offset_lv2 + flow_lv2.flip(1).repeat(1, feat_offset_lv2.size(1) // 2, 1, 1)
                feat_mask_lv2 = self.dcn_mask_lv2(feat_prop_lv2_)
                feat_mask_lv2 = torch.sigmoid(feat_mask_lv2)
                feat_prop_lv2 = self.dcn_lv2(feat_prop_lv2, feat_offset_lv2, feat_mask_lv2)

                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))

                feat_prop_lv3_ = torch.cat([feat_prop_lv2, feat_prop_lv3_, flow_lv3], dim=1)
                feat_prop_lv3_ = self.dcn_pre_lv3(feat_prop_lv3_)
                feat_prop_lv3_ = self.dcn_block_lv3(feat_prop_lv3_)
                feat_offset_lv3 = self.dcn_offset_lv3(feat_prop_lv3_)
                feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                feat_mask_lv3 = self.dcn_mask_lv3(feat_prop_lv3_)
                feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
                feat_prop_lv3 = self.dcn_lv3(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)

                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3
            
            else:
                feat_prop_lv0 = torch.cat([x_lr_lv0_cur, feat_prop_lv0], dim=1)
                feat_prop_lv0 = self.forward_resblocks_lv0(feat_prop_lv0)
                feat_prop_lv0 = self.lrelu(feat_prop_lv0)

                feat_prop_lv1 = torch.cat([feat_prop_lv0, feat_prop_lv1], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))
            
                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))
                
                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3

            self.feat_prop_lv3 = feat_prop_lv3
            out_lv3 = feat_prop_lv3
            out_lv3 = self.lrelu(self.conv_hr_lv3(out_lv3))
            out_lv3 = self.conv_last_lv3(out_lv3)
            base_lv3 = self.img_upsample_4x(lr_cur)
            out_lv3 += base_lv3
            outputs.append(out_lv3)

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
    def init_dcn(self):

        self.dcn_offset_lv0.weight.data.zero_()
        self.dcn_offset_lv0.bias.data.zero_()
        self.dcn_mask_lv0.weight.data.zero_()
        self.dcn_mask_lv0.bias.data.zero_()
        self.conv_identify(self.dcn_lv0.weight, self.dcn_lv0.bias)

        self.dcn_offset_lv1.weight.data.zero_()
        self.dcn_offset_lv1.bias.data.zero_()
        self.dcn_mask_lv1.weight.data.zero_()
        self.dcn_mask_lv1.bias.data.zero_()
        self.conv_identify(self.dcn_lv1.weight, self.dcn_lv1.bias)
        
        self.dcn_offset_lv2.weight.data.zero_()
        self.dcn_offset_lv2.bias.data.zero_()
        self.dcn_mask_lv2.weight.data.zero_()
        self.dcn_mask_lv2.bias.data.zero_()
        self.conv_identify(self.dcn_lv2.weight, self.dcn_lv2.bias)
        
        self.dcn_offset_lv3.weight.data.zero_()
        self.dcn_offset_lv3.bias.data.zero_()
        self.dcn_mask_lv3.weight.data.zero_()
        self.dcn_mask_lv3.bias.data.zero_()
        self.conv_identify(self.dcn_lv3.weight, self.dcn_lv3.bias)

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

class MRCF_CRA_x4(nn.Module):

    def __init__(self, device, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.dg_num = 16
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

        self.dcn_pre_lv0 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv1 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv2 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv3 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)

        self.dcn_block_lv0 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_block_lv1 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_block_lv2 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_block_lv3 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.dcn_offset_lv0 = nn.Conv2d(mid_channels, self.dg_num*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv0 = nn.Conv2d(mid_channels, self.dg_num*1*3*3, 3, 1, 1, bias=True)

        self.dcn_offset_lv1 = nn.Conv2d(mid_channels, self.dg_num*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv1 = nn.Conv2d(mid_channels, self.dg_num*1*3*3, 3, 1, 1, bias=True)

        self.dcn_offset_lv2 = nn.Conv2d(mid_channels, (self.dg_num//4)*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv2 = nn.Conv2d(mid_channels, (self.dg_num//4)*1*3*3, 3, 1, 1, bias=True)

        self.dcn_offset_lv3 = nn.Conv2d(mid_channels, (self.dg_num//16)*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv3 = nn.Conv2d(mid_channels, (self.dg_num//16)*1*3*3, 3, 1, 1, bias=True)
        
        self.dcn_lv0 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num)
        self.dcn_lv1 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num)
        self.dcn_lv2 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num//4)
        self.dcn_lv3 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num//16)
        self.init_dcn()

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr(mid_channels)
        self.conv_tttf_lv1 = conv3x3(64 + 64, mid_channels)
        self.conv_tttf_lv2 = conv3x3(64 + 64, mid_channels)
        self.conv_tttf_lv3 = conv3x3(64 + 64, mid_channels)

        # propagation branches
        self.forward_resblocks_lv0 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 3)
        self.forward_resblocks_lv1 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 3)
        self.forward_resblocks_lv2 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)
        self.forward_resblocks_lv3 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)

        # upsample
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.conv_hr_lv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last_lv3 = nn.Conv2d(64, 3, 3, 1, 1)

        ### 4x settings
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_downsample_2x = nn.Upsample(
            scale_factor=0.5, mode='bilinear', align_corners=False)
        self.img_upsample_4x = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_downsample_4x = nn.Upsample(
            scale_factor=0.25, mode='bilinear', align_corners=False)

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

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, fvs, mks):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lrs.size()

        ### compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)

        ### forward-time propagation and upsampling
        outputs = []
        self.offset_lv0 = []
        self.offset_lv1 = []
        self.offset_lv2 = []
        self.offset_lv3 = []

        feat_prop_lv0 = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_lv1 = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_lv2 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*4, w*4)

        ### texture transfer
        B, N, C, H, W = mks.size()
        mk_lv3 = mks.float()
        mk_lv2 = self.img_downsample_2x(mk_lv3.view(B*N, 1, H, W)).view(B, N, 1, H//2, W//2)
        mk_lv1 = self.img_downsample_2x(mk_lv2.view(B*N, 1, H//2, W//2)).view(B, N, 1, H//4, W//4)

        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_2x(lrs_lv0)
        lrs_lv3 = self.img_upsample_2x(lrs_lv3)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)
        fvs = fvs * mks.float() + lrs_lv3.view(B, N, C, H*4, W*4) * (1 - mks.float())
        B, N, C, H, W = fvs.size()
        x_hr_lv1, x_hr_lv2, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)

        #### Temporal fusion
        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv1.size()
        x_hr_lv1 = x_hr_lv1.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv2.size()
        x_hr_lv2 = x_hr_lv2.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)

        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            x_hr_lv1_cur = x_hr_lv1[:, i, :, :, :]
            x_hr_lv2_cur = x_hr_lv2[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            mk_cur_lv3 = mk_lv3[:, i, :, :, :]
            mk_cur_lv2 = mk_lv2[:, i, :, :, :]
            mk_cur_lv1 = mk_lv1[:, i, :, :, :]

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]

                flow_lv0 = flow
                flow_lv1 = flow_lv0
                flow_lv2 = self.img_upsample_2x(flow_lv1)
                flow_lv3 = self.img_upsample_2x(flow_lv2)

                feat_prop_lv2 = self.img_downsample_2x(feat_prop_lv3)
                feat_prop_lv1 = self.img_downsample_2x(feat_prop_lv2)
                feat_prop_lv0 = feat_prop_lv1

                feat_prop_lv0_ = flow_warp(feat_prop_lv0, flow_lv0.permute(0, 2, 3, 1))
                feat_prop_lv1_ = feat_prop_lv0_
                feat_prop_lv2_ = flow_warp(feat_prop_lv2, flow_lv2.permute(0, 2, 3, 1))
                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))

                feat_prop_lv0_ = torch.cat([x_lr_lv0_cur, feat_prop_lv0_, flow_lv0], dim=1)
                feat_prop_lv0_ = self.dcn_pre_lv0(feat_prop_lv0_)
                feat_prop_lv0_ = self.dcn_block_lv0(feat_prop_lv0_)
                feat_offset_lv0 = self.dcn_offset_lv0(feat_prop_lv0_)
                feat_offset_lv0 = self.max_residue_magnitude * torch.tanh(feat_offset_lv0)
                feat_offset_lv0 = feat_offset_lv0 + flow_lv0.flip(1).repeat(1, feat_offset_lv0.size(1) // 2, 1, 1)
                feat_mask_lv0 = self.dcn_mask_lv0(feat_prop_lv0_)
                feat_mask_lv0 = torch.sigmoid(feat_mask_lv0)
                feat_prop_lv0 = self.dcn_lv0(feat_prop_lv0, feat_offset_lv0, feat_mask_lv0)
                self.offset_lv0 = feat_offset_lv0.cpu()
                
                feat_prop_lv0 = torch.cat([x_lr_lv0_cur, feat_prop_lv0], dim=1)
                feat_prop_lv0 = self.forward_resblocks_lv0(feat_prop_lv0)
                feat_prop_lv0 = self.lrelu(feat_prop_lv0)

                feat_prop_lv1_ = torch.cat([feat_prop_lv0, feat_prop_lv1_, flow_lv1], dim=1)
                feat_prop_lv1_ = self.dcn_pre_lv1(feat_prop_lv1_)
                feat_prop_lv1_ = self.dcn_block_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.dcn_offset_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.max_residue_magnitude * torch.tanh(feat_offset_lv1)
                feat_offset_lv1 = feat_offset_lv1 + flow_lv1.flip(1).repeat(1, feat_offset_lv1.size(1) // 2, 1, 1)
                feat_mask_lv1 = self.dcn_mask_lv1(feat_prop_lv1_)
                feat_mask_lv1 = torch.sigmoid(feat_mask_lv1)
                feat_prop_lv1 = self.dcn_lv1(feat_prop_lv1, feat_offset_lv1, feat_mask_lv1)
                self.offset_lv1 = feat_offset_lv1.cpu()

                feat_prop_lv1 = torch.cat([feat_prop_lv0, feat_prop_lv1], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                feat_prop_lv1_ = torch.cat([feat_prop_lv1, x_hr_lv1_cur], dim=1)
                feat_prop_lv1_ = self.conv_tttf_lv1(feat_prop_lv1_)
                feat_prop_lv1 = mk_cur_lv1 * feat_prop_lv1_ + (1 - mk_cur_lv1) * feat_prop_lv1
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))

                feat_prop_lv2_ = torch.cat([feat_prop_lv1, feat_prop_lv2_, flow_lv2], dim=1)
                feat_prop_lv2_ = self.dcn_pre_lv2(feat_prop_lv2_)
                feat_prop_lv2_ = self.dcn_block_lv2(feat_prop_lv2_)
                feat_offset_lv2 = self.dcn_offset_lv2(feat_prop_lv2_)
                feat_offset_lv2 = self.max_residue_magnitude * torch.tanh(feat_offset_lv2)
                feat_offset_lv2 = feat_offset_lv2 + flow_lv2.flip(1).repeat(1, feat_offset_lv2.size(1) // 2, 1, 1)
                feat_mask_lv2 = self.dcn_mask_lv2(feat_prop_lv2_)
                feat_mask_lv2 = torch.sigmoid(feat_mask_lv2)
                feat_prop_lv2 = self.dcn_lv2(feat_prop_lv2, feat_offset_lv2, feat_mask_lv2)
                self.offset_lv2 = feat_offset_lv2.cpu()

                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                feat_prop_lv2_ = torch.cat([feat_prop_lv2, x_hr_lv2_cur], dim=1)
                feat_prop_lv2_ = self.conv_tttf_lv2(feat_prop_lv2_)
                feat_prop_lv2 = mk_cur_lv2 * feat_prop_lv2_ + (1 - mk_cur_lv2) * feat_prop_lv2
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))

                feat_prop_lv3_ = torch.cat([feat_prop_lv2, feat_prop_lv3_, flow_lv3], dim=1)
                feat_prop_lv3_ = self.dcn_pre_lv3(feat_prop_lv3_)
                feat_prop_lv3_ = self.dcn_block_lv3(feat_prop_lv3_)
                feat_offset_lv3 = self.dcn_offset_lv3(feat_prop_lv3_)
                feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                feat_mask_lv3 = self.dcn_mask_lv3(feat_prop_lv3_)
                feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
                feat_prop_lv3 = self.dcn_lv3(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)
                self.offset_lv3 = feat_offset_lv3.cpu()

                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3
            
            else:
                feat_prop_lv0 = torch.cat([x_lr_lv0_cur, feat_prop_lv0], dim=1)
                feat_prop_lv0 = self.forward_resblocks_lv0(feat_prop_lv0)
                feat_prop_lv0 = self.lrelu(feat_prop_lv0)

                feat_prop_lv1 = torch.cat([feat_prop_lv0, feat_prop_lv1], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                feat_prop_lv1_ = torch.cat([feat_prop_lv1, x_hr_lv1_cur], dim=1)
                feat_prop_lv1_ = self.conv_tttf_lv1(feat_prop_lv1_)
                feat_prop_lv1 = mk_cur_lv1 * feat_prop_lv1_ + (1 - mk_cur_lv1) * feat_prop_lv1
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))
            
                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                feat_prop_lv2_ = torch.cat([feat_prop_lv2, x_hr_lv2_cur], dim=1)
                feat_prop_lv2_ = self.conv_tttf_lv2(feat_prop_lv2_)
                feat_prop_lv2 = mk_cur_lv2 * feat_prop_lv2_ + (1 - mk_cur_lv2) * feat_prop_lv2
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))
                
                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3

            self.feat_prop_lv3 = feat_prop_lv3
            out_lv3 = feat_prop_lv3
            out_lv3 = self.lrelu(self.conv_hr_lv3(out_lv3))
            out_lv3 = self.conv_last_lv3(out_lv3)
            base_lv3 = self.img_upsample_4x(lr_cur)
            out_lv3 += base_lv3
            outputs.append(out_lv3)
            
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
    def init_dcn(self):

        self.dcn_offset_lv0.weight.data.zero_()
        self.dcn_offset_lv0.bias.data.zero_()
        self.dcn_mask_lv0.weight.data.zero_()
        self.dcn_mask_lv0.bias.data.zero_()
        self.conv_identify(self.dcn_lv0.weight, self.dcn_lv0.bias)

        self.dcn_offset_lv1.weight.data.zero_()
        self.dcn_offset_lv1.bias.data.zero_()
        self.dcn_mask_lv1.weight.data.zero_()
        self.dcn_mask_lv1.bias.data.zero_()
        self.conv_identify(self.dcn_lv1.weight, self.dcn_lv1.bias)
        
        self.dcn_offset_lv2.weight.data.zero_()
        self.dcn_offset_lv2.bias.data.zero_()
        self.dcn_mask_lv2.weight.data.zero_()
        self.dcn_mask_lv2.bias.data.zero_()
        self.conv_identify(self.dcn_lv2.weight, self.dcn_lv2.bias)
        
        self.dcn_offset_lv3.weight.data.zero_()
        self.dcn_offset_lv3.bias.data.zero_()
        self.dcn_mask_lv3.weight.data.zero_()
        self.dcn_mask_lv3.bias.data.zero_()
        self.conv_identify(self.dcn_lv3.weight, self.dcn_lv3.bias)

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

class MRCF_x8(nn.Module):

    def __init__(self, device, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.dg_num = 16
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

        self.dcn_pre_lv0 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv1 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv2 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv3 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)

        self.dcn_block_lv0 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_block_lv1 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_block_lv2 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_block_lv3 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_offset_lv0 = nn.Conv2d(mid_channels, self.dg_num*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv0 = nn.Conv2d(mid_channels, self.dg_num*1*3*3, 3, 1, 1, bias=True)

        self.dcn_offset_lv1 = nn.Conv2d(mid_channels, self.dg_num*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv1 = nn.Conv2d(mid_channels, self.dg_num*1*3*3, 3, 1, 1, bias=True)

        self.dcn_offset_lv2 = nn.Conv2d(mid_channels, (self.dg_num//4)*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv2 = nn.Conv2d(mid_channels, (self.dg_num//4)*1*3*3, 3, 1, 1, bias=True)

        self.dcn_offset_lv3 = nn.Conv2d(mid_channels, (self.dg_num//16)*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv3 = nn.Conv2d(mid_channels, (self.dg_num//16)*1*3*3, 3, 1, 1, bias=True)
        
        self.dcn_lv0 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num)
        self.dcn_lv1 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num)
        self.dcn_lv2 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num//4)
        self.dcn_lv3 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num//16)
        self.init_dcn()

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr(mid_channels)

        self.conv_tttf_lv3 = conv3x3(64 + 64, mid_channels)

        # propagation branches
        self.forward_resblocks_lv0 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 3)
        self.forward_resblocks_lv1 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 3)
        self.forward_resblocks_lv2 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)
        self.forward_resblocks_lv3 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)

        # upsample
        self.upsample0 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)

        self.conv_hr_lv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last_lv3 = nn.Conv2d(64, 3, 3, 1, 1)

        ### 4x settings
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_downsample_2x = nn.Upsample(
            scale_factor=0.5, mode='bilinear', align_corners=False)

        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)
        self.img_downsample_8x = nn.Upsample(
            scale_factor=0.125, mode='bilinear', align_corners=False)

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

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, fvs, mks):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lrs.size()

        ### compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)
        
        ### forward-time propagation and upsampling
        outputs = []
        feat_prop_lv0 = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_lv1 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv2 = lrs.new_zeros(n, self.mid_channels, h*4, w*4)
        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*8, w*8)

        ### texture transfer
        B, N, C, H, W = mks.size()
        mk_lv3 = mks.float()
        mk_lv2 = self.img_downsample_2x(mk_lv3.view(B*N, 1, H, W)).view(B, N, 1, H//2, W//2)
        mk_lv1 = self.img_downsample_2x(mk_lv2.view(B*N, 1, H//2, W//2)).view(B, N, 1, H//4, W//4)
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_2x(lrs_lv0)
        lrs_lv3 = self.img_upsample_2x(lrs_lv3)
        lrs_lv3 = self.img_upsample_2x(lrs_lv3)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        fvs = fvs * mks.float() + lrs_lv3.view(B, N, C, H*8, W*8) * (1 - mks.float())
        B, N, C, H, W = fvs.size()
        x_hr_lv1, x_hr_lv2, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv1.size()
        x_hr_lv1 = x_hr_lv1.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv2.size()
        x_hr_lv2 = x_hr_lv2.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)

        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]

            x_hr_lv1_cur = x_hr_lv1[:, i, :, :, :]
            x_hr_lv2_cur = x_hr_lv2[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            mk_cur_lv3 = mk_lv3[:, i, :, :, :]
            mk_cur_lv2 = mk_lv2[:, i, :, :, :]
            mk_cur_lv1 = mk_lv1[:, i, :, :, :]

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]

                flow_lv0 = flow
                flow_lv1 = self.img_upsample_2x(flow_lv0)
                flow_lv2 = self.img_upsample_2x(flow_lv1)
                flow_lv3 = self.img_upsample_2x(flow_lv2)

                feat_prop_lv2 = self.img_downsample_2x(feat_prop_lv3)
                feat_prop_lv1 = self.img_downsample_2x(feat_prop_lv2)
                feat_prop_lv0 = self.img_downsample_2x(feat_prop_lv1)

                feat_prop_lv0_ = flow_warp(feat_prop_lv0, flow_lv0.permute(0, 2, 3, 1))
                feat_prop_lv1_ = flow_warp(feat_prop_lv1, flow_lv1.permute(0, 2, 3, 1))
                feat_prop_lv2_ = flow_warp(feat_prop_lv2, flow_lv2.permute(0, 2, 3, 1))
                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))

                feat_prop_lv0_ = torch.cat([x_lr_lv0_cur, feat_prop_lv0_, flow_lv0], dim=1)
                feat_prop_lv0_ = self.dcn_pre_lv0(feat_prop_lv0_)
                feat_prop_lv0_ = self.dcn_block_lv0(feat_prop_lv0_)
                feat_offset_lv0 = self.dcn_offset_lv0(feat_prop_lv0_)
                feat_offset_lv0 = self.max_residue_magnitude * torch.tanh(feat_offset_lv0)
                feat_offset_lv0 = feat_offset_lv0 + flow_lv0.flip(1).repeat(1, feat_offset_lv0.size(1) // 2, 1, 1)
                feat_mask_lv0 = self.dcn_mask_lv0(feat_prop_lv0_)
                feat_mask_lv0 = torch.sigmoid(feat_mask_lv0)
                feat_prop_lv0 = self.dcn_lv0(feat_prop_lv0, feat_offset_lv0, feat_mask_lv0)
                
                feat_prop_lv0 = torch.cat([x_lr_lv0_cur, feat_prop_lv0], dim=1)
                feat_prop_lv0 = self.forward_resblocks_lv0(feat_prop_lv0)
                feat_prop_lv0 = self.lrelu(self.upsample0(feat_prop_lv0))

                feat_prop_lv1_ = torch.cat([feat_prop_lv0, feat_prop_lv1_, flow_lv1], dim=1)
                feat_prop_lv1_ = self.dcn_pre_lv1(feat_prop_lv1_)
                feat_prop_lv1_ = self.dcn_block_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.dcn_offset_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.max_residue_magnitude * torch.tanh(feat_offset_lv1)
                feat_offset_lv1 = feat_offset_lv1 + flow_lv1.flip(1).repeat(1, feat_offset_lv1.size(1) // 2, 1, 1)
                feat_mask_lv1 = self.dcn_mask_lv1(feat_prop_lv1_)
                feat_mask_lv1 = torch.sigmoid(feat_mask_lv1)
                feat_prop_lv1 = self.dcn_lv1(feat_prop_lv1, feat_offset_lv1, feat_mask_lv1)

                feat_prop_lv1 = torch.cat([feat_prop_lv0, feat_prop_lv1], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))

                feat_prop_lv2_ = torch.cat([feat_prop_lv1, feat_prop_lv2_, flow_lv2], dim=1)
                feat_prop_lv2_ = self.dcn_pre_lv2(feat_prop_lv2_)
                feat_prop_lv2_ = self.dcn_block_lv2(feat_prop_lv2_)
                feat_offset_lv2 = self.dcn_offset_lv2(feat_prop_lv2_)
                feat_offset_lv2 = self.max_residue_magnitude * torch.tanh(feat_offset_lv2)
                feat_offset_lv2 = feat_offset_lv2 + flow_lv2.flip(1).repeat(1, feat_offset_lv2.size(1) // 2, 1, 1)
                feat_mask_lv2 = self.dcn_mask_lv2(feat_prop_lv2_)
                feat_mask_lv2 = torch.sigmoid(feat_mask_lv2)
                feat_prop_lv2 = self.dcn_lv2(feat_prop_lv2, feat_offset_lv2, feat_mask_lv2)

                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))

                feat_prop_lv3_ = torch.cat([feat_prop_lv2, feat_prop_lv3_, flow_lv3], dim=1)
                feat_prop_lv3_ = self.dcn_pre_lv3(feat_prop_lv3_)
                feat_prop_lv3_ = self.dcn_block_lv3(feat_prop_lv3_)
                feat_offset_lv3 = self.dcn_offset_lv3(feat_prop_lv3_)
                feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                feat_mask_lv3 = self.dcn_mask_lv3(feat_prop_lv3_)
                feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
                feat_prop_lv3 = self.dcn_lv3(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)

                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3
            
            else:
                feat_prop_lv0 = torch.cat([x_lr_lv0_cur, feat_prop_lv0], dim=1)
                feat_prop_lv0 = self.forward_resblocks_lv0(feat_prop_lv0)
                feat_prop_lv0 = self.lrelu(self.upsample0(feat_prop_lv0))

                feat_prop_lv1 = torch.cat([feat_prop_lv0, feat_prop_lv1], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))
            
                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))
                
                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3

            out_lv3 = feat_prop_lv3
            out_lv3 = self.lrelu(self.conv_hr_lv3(out_lv3))
            out_lv3 = self.conv_last_lv3(out_lv3)
            base_lv3 = self.img_upsample_8x(lr_cur)
            out_lv3 += base_lv3
            outputs.append(out_lv3)

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
    def init_dcn(self):

        self.dcn_offset_lv0.weight.data.zero_()
        self.dcn_offset_lv0.bias.data.zero_()
        self.dcn_mask_lv0.weight.data.zero_()
        self.dcn_mask_lv0.bias.data.zero_()
        self.conv_identify(self.dcn_lv0.weight, self.dcn_lv0.bias)

        self.dcn_offset_lv1.weight.data.zero_()
        self.dcn_offset_lv1.bias.data.zero_()
        self.dcn_mask_lv1.weight.data.zero_()
        self.dcn_mask_lv1.bias.data.zero_()
        self.conv_identify(self.dcn_lv1.weight, self.dcn_lv1.bias)
        
        self.dcn_offset_lv2.weight.data.zero_()
        self.dcn_offset_lv2.bias.data.zero_()
        self.dcn_mask_lv2.weight.data.zero_()
        self.dcn_mask_lv2.bias.data.zero_()
        self.conv_identify(self.dcn_lv2.weight, self.dcn_lv2.bias)
        
        self.dcn_offset_lv3.weight.data.zero_()
        self.dcn_offset_lv3.bias.data.zero_()
        self.dcn_mask_lv3.weight.data.zero_()
        self.dcn_mask_lv3.bias.data.zero_()
        self.conv_identify(self.dcn_lv3.weight, self.dcn_lv3.bias)

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

# class MRCF_CRA_x8(nn.Module):

#     def __init__(self, device, mid_channels=64, num_blocks=30, spynet_pretrained=None):

#         super().__init__()

#         self.device = device
#         self.mid_channels = mid_channels
#         self.dg_num = 16
#         self.max_residue_magnitude = 10

#         # optical flow network for feature alignment
#         self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

#         self.dcn_pre_lv0 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
#         self.dcn_pre_lv1 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
#         self.dcn_pre_lv2 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
#         self.dcn_pre_lv3 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)

#         self.dcn_block_lv0 = nn.Sequential(
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#         self.dcn_block_lv1 = nn.Sequential(
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#         self.dcn_block_lv2 = nn.Sequential(
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#         self.dcn_block_lv3 = nn.Sequential(
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
        
#         self.dcn_offset_lv0 = nn.Conv2d(mid_channels, (self.dg_num//16)*2*3*3, 3, 1, 1, bias=True)
#         self.dcn_mask_lv0 = nn.Conv2d(mid_channels, (self.dg_num//16)*1*3*3, 3, 1, 1, bias=True)

#         self.dcn_offset_lv1 = nn.Conv2d(mid_channels, (self.dg_num//16)*2*3*3, 3, 1, 1, bias=True)
#         self.dcn_mask_lv1 = nn.Conv2d(mid_channels, (self.dg_num//16)*1*3*3, 3, 1, 1, bias=True)

#         self.dcn_offset_lv2 = nn.Conv2d(mid_channels, (self.dg_num//16)*2*3*3, 3, 1, 1, bias=True)
#         self.dcn_mask_lv2 = nn.Conv2d(mid_channels, (self.dg_num//16)*1*3*3, 3, 1, 1, bias=True)

#         self.dcn_offset_lv3 = nn.Conv2d(mid_channels, (self.dg_num//16)*2*3*3, 3, 1, 1, bias=True)
#         self.dcn_mask_lv3 = nn.Conv2d(mid_channels, (self.dg_num//16)*1*3*3, 3, 1, 1, bias=True)
        
#         self.dcn_lv0 = DCNv2(mid_channels, mid_channels, 3,
#                         stride=1, padding=1, dilation=1,
#                         deformable_groups=self.dg_num//16)
#         self.dcn_lv1 = DCNv2(mid_channels, mid_channels, 3,
#                         stride=1, padding=1, dilation=1,
#                         deformable_groups=self.dg_num//16)
#         self.dcn_lv2 = DCNv2(mid_channels, mid_channels, 3,
#                         stride=1, padding=1, dilation=1,
#                         deformable_groups=self.dg_num//16)
#         self.dcn_lv3 = DCNv2(mid_channels, mid_channels, 3,
#                         stride=1, padding=1, dilation=1,
#                         deformable_groups=self.dg_num//16)
#         self.init_dcn()

#         # feature extractor
#         self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
#         self.encoder_hr = LTE.LTE_simple_hr(mid_channels)

#         self.conv_tttf_lv1 = conv3x3(64 + 64, mid_channels)
#         self.conv_tttf_lv2 = conv3x3(64 + 64, mid_channels)
#         self.conv_tttf_lv3 = conv3x3(64 + 64, mid_channels)

#         # propagation branches
#         self.forward_resblocks_lv0 = ResidualBlocksWithInputConv(
#             mid_channels * 2, mid_channels, 1)
#         self.forward_resblocks_lv1 = ResidualBlocksWithInputConv(
#             mid_channels * 2, mid_channels, 1)
#         self.forward_resblocks_lv2 = ResidualBlocksWithInputConv(
#             mid_channels * 2, mid_channels, 1)
#         self.forward_resblocks_lv3 = ResidualBlocksWithInputConv(
#             mid_channels * 2, mid_channels, 1)

#         # upsample
#         self.upsample0 = PixelShufflePack(
#             mid_channels, mid_channels, 2, upsample_kernel=3)
#         self.upsample1 = PixelShufflePack(
#             mid_channels, mid_channels, 2, upsample_kernel=3)
#         self.upsample2 = PixelShufflePack(
#             mid_channels, mid_channels, 2, upsample_kernel=3)

#         self.conv_hr_lv3 = nn.Conv2d(64, 64, 3, 1, 1)
#         self.conv_last_lv3 = nn.Conv2d(64, 3, 3, 1, 1)

#         ### 4x settings
#         self.img_upsample_2x = nn.Upsample(
#             scale_factor=2, mode='bilinear', align_corners=False)
#         self.img_downsample_2x = nn.Upsample(
#             scale_factor=0.5, mode='bilinear', align_corners=False)
#         ### 8x settings
#         self.img_upsample_8x = nn.Upsample(
#             scale_factor=8, mode='bilinear', align_corners=False)
#         self.img_downsample_8x = nn.Upsample(
#             scale_factor=0.125, mode='bilinear', align_corners=False)

#         # activation function
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#     def compute_flow(self, lrs):
#         """Compute optical flow using SPyNet for feature warping.

#         Note that if the input is an mirror-extended sequence, 'flows_forward'
#         is not needed, since it is equal to 'flows_backward.flip(1)'.

#         Args:
#             lrs (tensor): Input LR images with shape (n, t, c, h, w)

#         Return:
#             tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
#                 flows used for forward-time propagation (current to previous).
#                 'flows_backward' corresponds to the flows used for
#                 backward-time propagation (current to next).
#         """

#         n, t, c, h, w = lrs.size()
#         lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
#         lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

#         # flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
#         flows_backward = None

#         flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

#         return flows_forward, flows_backward
#     # @profile
#     def forward(self, lrs, fvs, mks):
#         """Forward function for BasicVSR.

#         Args:
#             lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

#         Returns:
#             Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
#         """

#         # print(lrs.size(), mks.size())
#         n, t, c, h, w = lrs.size()

#         ### compute optical flow
#         flows_forward, flows_backward = self.compute_flow(lrs)
        
#         ### forward-time propagation and upsampling
#         outputs = []

#         feat_prop_lv0 = lrs.new_zeros(n, self.mid_channels, h, w)
#         feat_prop_lv1 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
#         feat_prop_lv2 = lrs.new_zeros(n, self.mid_channels, h*4, w*4)
#         feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*8, w*8)

#         ### texture transfer
#         B, N, C, H, W = mks.size()
#         mk_lv3 = mks.float()
#         mk_lv2 = self.img_downsample_2x(mk_lv3.view(B*N, 1, H, W)).view(B, N, 1, H//2, W//2)
#         mk_lv1 = self.img_downsample_2x(mk_lv2.view(B*N, 1, H//2, W//2)).view(B, N, 1, H//4, W//4)
#         B, N, C, H, W = lrs.size()
#         lrs_lv0 = lrs.view(B*N, C, H, W)
#         lrs_lv3 = self.img_upsample_2x(lrs_lv0)
#         lrs_lv3 = self.img_upsample_2x(lrs_lv3)
#         lrs_lv3 = self.img_upsample_2x(lrs_lv3)

#         _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

#         fvs = fvs * mks.float() + lrs_lv3.view(B, N, C, H*8, W*8) * (1 - mks.float())
#         B, N, C, H, W = fvs.size()
#         x_hr_lv1, x_hr_lv2, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)

#         _, C, H, W = x_lr_lv0.size()
#         x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

#         _, C, H, W = x_hr_lv1.size()
#         x_hr_lv1 = x_hr_lv1.contiguous().view(B, N, C, H, W)
#         _, C, H, W = x_hr_lv2.size()
#         x_hr_lv2 = x_hr_lv2.contiguous().view(B, N, C, H, W)
#         _, C, H, W = x_hr_lv3.size()
#         x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)

#         res_lv0_list = []
#         res_lv1_list = []
#         res_lv2_list = []
#         res_lv3_list = []        
        
#         dcn_pre_lv0_list = []
#         dcn_pre_lv1_list = []
#         dcn_pre_lv2_list = []

#         dcn_pre_lv3_list = []
#         dcn_block_lv3_list = []
#         dcn_offset_lv3_list = []
#         dcn_mask_lv3_list = []

#         dcn_lv0_list = []
#         dcn_lv1_list = []
#         dcn_lv2_list = []
#         dcn_lv3_list = []
        
#         flow_lv0_list = []
#         flow_lv1_list = []
#         flow_lv2_list = []
#         flow_lv3_list = []

#         last_list = []

#         for i in range(0, t):
#             lr_cur = lrs[:, i, :, :, :]
#             x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]

#             x_hr_lv1_cur = x_hr_lv1[:, i, :, :, :]
#             x_hr_lv2_cur = x_hr_lv2[:, i, :, :, :]
#             x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
#             mk_cur_lv3 = mk_lv3[:, i, :, :, :]
#             mk_cur_lv2 = mk_lv2[:, i, :, :, :]
#             mk_cur_lv1 = mk_lv1[:, i, :, :, :]

#             if i > 0:  # no warping required for the first timestep
#                 flow = flows_forward[:, i - 1, :, :, :]

#                 flow_lv0 = flow
#                 flow_lv1 = self.img_upsample_2x(flow_lv0)
#                 flow_lv2 = self.img_upsample_2x(flow_lv1)
#                 flow_lv3 = self.img_upsample_2x(flow_lv2)

#                 feat_prop_lv2 = self.img_downsample_2x(feat_prop_lv3)
#                 feat_prop_lv1 = self.img_downsample_2x(feat_prop_lv2)
#                 feat_prop_lv0 = self.img_downsample_2x(feat_prop_lv1)

#                 feat_prop_lv0_ = flow_warp(feat_prop_lv0, flow_lv0.permute(0, 2, 3, 1))
#                 feat_prop_lv1_ = flow_warp(feat_prop_lv1, flow_lv1.permute(0, 2, 3, 1))
#                 feat_prop_lv2_ = flow_warp(feat_prop_lv2, flow_lv2.permute(0, 2, 3, 1))
#                 feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))

#                 feat_prop_lv0_ = torch.cat([x_lr_lv0_cur, feat_prop_lv0_, flow_lv0], dim=1)
#                 feat_prop_lv0_ = self.dcn_pre_lv0(feat_prop_lv0_)
#                 feat_prop_lv0_ = self.dcn_block_lv0(feat_prop_lv0_)
#                 feat_offset_lv0 = self.dcn_offset_lv0(feat_prop_lv0_)
#                 feat_offset_lv0 = self.max_residue_magnitude * torch.tanh(feat_offset_lv0)
#                 feat_offset_lv0 = feat_offset_lv0 + flow_lv0.flip(1).repeat(1, feat_offset_lv0.size(1) // 2, 1, 1)
#                 # flow_lv0 = torch.cat((flow_lv0[:, 1:2, :, :], flow_lv0[:, 0:1, :, :]), dim=1)
#                 # flow_lv0 = flow_lv0.repeat(1, feat_offset_lv0.size(1) // 2, 1, 1)
#                 # feat_offset_lv0 = feat_offset_lv0 + flow_lv0
#                 feat_mask_lv0 = self.dcn_mask_lv0(feat_prop_lv0_)
#                 feat_mask_lv0 = torch.sigmoid(feat_mask_lv0)

#                 feat_prop_lv0 = self.dcn_lv0(feat_prop_lv0, feat_offset_lv0, feat_mask_lv0)
                
#                 feat_prop_lv0 = torch.cat([x_lr_lv0_cur, feat_prop_lv0], dim=1)
#                 feat_prop_lv0 = self.forward_resblocks_lv0(feat_prop_lv0)
#                 feat_prop_lv0 = self.lrelu(self.upsample0(feat_prop_lv0))

#                 feat_prop_lv1_ = torch.cat([feat_prop_lv0, feat_prop_lv1_, flow_lv1], dim=1)
#                 feat_prop_lv1_ = self.dcn_pre_lv1(feat_prop_lv1_)
#                 feat_prop_lv1_ = self.dcn_block_lv1(feat_prop_lv1_)
#                 feat_offset_lv1 = self.dcn_offset_lv1(feat_prop_lv1_)
#                 feat_offset_lv1 = self.max_residue_magnitude * torch.tanh(feat_offset_lv1)
#                 feat_offset_lv1 = feat_offset_lv1 + flow_lv1.flip(1).repeat(1, feat_offset_lv1.size(1) // 2, 1, 1)
#                 # flow_lv1 = torch.cat((flow_lv1[:, 1:2, :, :], flow_lv1[:, 0:1, :, :]), dim=1)
#                 # flow_lv1 = flow_lv1.repeat(1, feat_offset_lv1.size(1) // 2, 1, 1)
#                 # feat_offset_lv1 = feat_offset_lv1 + flow_lv1
#                 feat_mask_lv1 = self.dcn_mask_lv1(feat_prop_lv1_)
#                 feat_mask_lv1 = torch.sigmoid(feat_mask_lv1)

#                 feat_prop_lv1 = self.dcn_lv1(feat_prop_lv1, feat_offset_lv1, feat_mask_lv1)

#                 feat_prop_lv1 = torch.cat([feat_prop_lv0, feat_prop_lv1], dim=1)
#                 feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
#                 feat_prop_lv1_ = torch.cat([feat_prop_lv1, x_hr_lv1_cur], dim=1)
#                 feat_prop_lv1_ = self.conv_tttf_lv1(feat_prop_lv1_)
#                 feat_prop_lv1 = mk_cur_lv1 * feat_prop_lv1_ + (1 - mk_cur_lv1) * feat_prop_lv1
#                 feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))

#                 feat_prop_lv2_ = torch.cat([feat_prop_lv1, feat_prop_lv2_, flow_lv2], dim=1)
#                 feat_prop_lv2_ = self.dcn_pre_lv2(feat_prop_lv2_)
#                 feat_prop_lv2_ = self.dcn_block_lv2(feat_prop_lv2_)
#                 feat_offset_lv2 = self.dcn_offset_lv2(feat_prop_lv2_)
#                 feat_offset_lv2 = self.max_residue_magnitude * torch.tanh(feat_offset_lv2)
#                 feat_offset_lv2 = feat_offset_lv2 + flow_lv2.flip(1).repeat(1, feat_offset_lv2.size(1) // 2, 1, 1)
#                 # flow_lv2 = torch.cat((flow_lv2[:, 1:2, :, :], flow_lv2[:, 0:1, :, :]), dim=1)
#                 # flow_lv2 = flow_lv2.repeat(1, feat_offset_lv2.size(1) // 2, 1, 1)
#                 # feat_offset_lv2 = feat_offset_lv2 + flow_lv2
#                 feat_mask_lv2 = self.dcn_mask_lv2(feat_prop_lv2_)
#                 feat_mask_lv2 = torch.sigmoid(feat_mask_lv2)

#                 feat_prop_lv2 = self.dcn_lv2(feat_prop_lv2, feat_offset_lv2, feat_mask_lv2)

#                 feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
#                 feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
#                 feat_prop_lv2_ = torch.cat([feat_prop_lv2, x_hr_lv2_cur], dim=1)
#                 feat_prop_lv2_ = self.conv_tttf_lv2(feat_prop_lv2_)
#                 feat_prop_lv2 = mk_cur_lv2 * feat_prop_lv2_ + (1 - mk_cur_lv2) * feat_prop_lv2
#                 feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))

#                 feat_prop_lv3_ = torch.cat([feat_prop_lv2, feat_prop_lv3_, flow_lv3], dim=1)
#                 feat_prop_lv3_ = self.dcn_pre_lv3(feat_prop_lv3_)
#                 feat_prop_lv3_ = self.dcn_block_lv3(feat_prop_lv3_)
#                 feat_offset_lv3 = self.dcn_offset_lv3(feat_prop_lv3_)
#                 feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
#                 feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
#                 # flow_lv3 = torch.cat((flow_lv3[:, 1:2, :, :], flow_lv3[:, 0:1, :, :]), dim=1)
#                 # flow_lv3 = flow_lv3.repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
#                 # feat_offset_lv3 = feat_offset_lv3 + flow_lv3
#                 feat_mask_lv3 = self.dcn_mask_lv3(feat_prop_lv3_)
#                 feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)

#                 feat_prop_lv3 = self.dcn_lv3(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)

#                 feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
#                 feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
#                 feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
#                 feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
#                 feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3
            
#             else:
#                 feat_prop_lv0 = torch.cat([x_lr_lv0_cur, feat_prop_lv0], dim=1)
#                 feat_prop_lv0 = self.forward_resblocks_lv0(feat_prop_lv0)
#                 feat_prop_lv0 = self.lrelu(self.upsample0(feat_prop_lv0))

#                 feat_prop_lv1 = torch.cat([feat_prop_lv0, feat_prop_lv1], dim=1)
#                 feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
#                 feat_prop_lv1_ = torch.cat([feat_prop_lv1, x_hr_lv1_cur], dim=1)
#                 feat_prop_lv1_ = self.conv_tttf_lv1(feat_prop_lv1_)
#                 feat_prop_lv1 = mk_cur_lv1 * feat_prop_lv1_ + (1 - mk_cur_lv1) * feat_prop_lv1
#                 feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))
            
#                 feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
#                 feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
#                 feat_prop_lv2_ = torch.cat([feat_prop_lv2, x_hr_lv2_cur], dim=1)
#                 feat_prop_lv2_ = self.conv_tttf_lv2(feat_prop_lv2_)
#                 feat_prop_lv2 = mk_cur_lv2 * feat_prop_lv2_ + (1 - mk_cur_lv2) * feat_prop_lv2
#                 feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))
                
#                 feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
#                 feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
#                 feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
#                 feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
#                 feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3

#             out_lv3 = feat_prop_lv3
#             out_lv3 = self.lrelu(self.conv_hr_lv3(out_lv3))
#             out_lv3 = self.conv_last_lv3(out_lv3)
#             base_lv3 = self.img_upsample_8x(lr_cur)
#             out_lv3 += base_lv3
#             outputs.append(out_lv3)

#         return torch.stack(outputs, dim=1)

#     def init_weights(self, pretrained=None, strict=True):
#         """Init weights for models.

#         Args:
#             pretrained (str, optional): Path for pretrained weights. If given
#                 None, pretrained weights will not be loaded. Defaults: None.
#             strict (boo, optional): Whether strictly load the pretrained model.
#                 Defaults to True.
#         """
#         if isinstance(pretrained, str):
#             # logger = get_root_logger()
#             # load_checkpoint(self, pretrained, strict=strict, logger=logger)
#             model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
#             model_state_dict = self.state_dict()
#             model_state_dict.update(model_state_dict_save)
#             self.load_state_dict(model_state_dict, strict=strict)
#         elif pretrained is not None:
#             raise TypeError(f'"pretrained" must be a str or None. '
#                             f'But received {type(pretrained)}.')
#     def init_dcn(self):

#         self.dcn_offset_lv0.weight.data.zero_()
#         self.dcn_offset_lv0.bias.data.zero_()
#         self.dcn_mask_lv0.weight.data.zero_()
#         self.dcn_mask_lv0.bias.data.zero_()
#         self.conv_identify(self.dcn_lv0.weight, self.dcn_lv0.bias)

#         self.dcn_offset_lv1.weight.data.zero_()
#         self.dcn_offset_lv1.bias.data.zero_()
#         self.dcn_mask_lv1.weight.data.zero_()
#         self.dcn_mask_lv1.bias.data.zero_()
#         self.conv_identify(self.dcn_lv1.weight, self.dcn_lv1.bias)
        
#         self.dcn_offset_lv2.weight.data.zero_()
#         self.dcn_offset_lv2.bias.data.zero_()
#         self.dcn_mask_lv2.weight.data.zero_()
#         self.dcn_mask_lv2.bias.data.zero_()
#         self.conv_identify(self.dcn_lv2.weight, self.dcn_lv2.bias)
        
#         self.dcn_offset_lv3.weight.data.zero_()
#         self.dcn_offset_lv3.bias.data.zero_()
#         self.dcn_mask_lv3.weight.data.zero_()
#         self.dcn_mask_lv3.bias.data.zero_()
#         self.conv_identify(self.dcn_lv3.weight, self.dcn_lv3.bias)

#     def conv_identify(self, weight, bias):
#         weight.data.zero_()
#         bias.data.zero_()
#         o, i, h, w = weight.shape
#         y = h//2
#         x = w//2
#         for p in range(i):
#             for q in range(o):
#                 if p == q:
#                     weight.data[q, p, y, x] = 1.0

class MRCF_CRA_x8(nn.Module):

    def __init__(self, device, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.dg_num = 16
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

        self.dcn_pre_lv0 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv1 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv2 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv3 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)

        self.dcn_block_lv0 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_block_lv1 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_block_lv2 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_block_lv3 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.dcn_offset_lv0 = nn.Conv2d(mid_channels, (self.dg_num//16)*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv0 = nn.Conv2d(mid_channels, (self.dg_num//16)*1*3*3, 3, 1, 1, bias=True)

        self.dcn_offset_lv1 = nn.Conv2d(mid_channels, (self.dg_num//16)*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv1 = nn.Conv2d(mid_channels, (self.dg_num//16)*1*3*3, 3, 1, 1, bias=True)

        self.dcn_offset_lv2 = nn.Conv2d(mid_channels, (self.dg_num//16)*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv2 = nn.Conv2d(mid_channels, (self.dg_num//16)*1*3*3, 3, 1, 1, bias=True)

        self.dcn_offset_lv3 = nn.Conv2d(mid_channels, (self.dg_num//16)*2*3*3, 3, 1, 1, bias=True)
        self.dcn_mask_lv3 = nn.Conv2d(mid_channels, (self.dg_num//16)*1*3*3, 3, 1, 1, bias=True)
        
        self.dcn_lv0 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num//16)
        self.dcn_lv1 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num//16)
        self.dcn_lv2 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num//16)
        self.dcn_lv3 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num//16)
        self.init_dcn()

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr(mid_channels)

        self.conv_tttf_lv1 = conv3x3(64 + 64, mid_channels)
        self.conv_tttf_lv2 = conv3x3(64 + 64, mid_channels)
        self.conv_tttf_lv3 = conv3x3(64 + 64, mid_channels)

        # propagation branches
        self.forward_resblocks_lv0 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 3)
        self.forward_resblocks_lv1 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 3)
        self.forward_resblocks_lv2 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)
        self.forward_resblocks_lv3 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)

        # upsample
        self.upsample0 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)

        self.conv_hr_lv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last_lv3 = nn.Conv2d(64, 3, 3, 1, 1)

        ### 4x settings
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_downsample_2x = nn.Upsample(
            scale_factor=0.5, mode='bilinear', align_corners=False)
        ### 8x settings
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)
        self.img_downsample_8x = nn.Upsample(
            scale_factor=0.125, mode='bilinear', align_corners=False)

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
    # @profile
    def forward(self, lrs, fvs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        # print(lrs.size(), mks.size())
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        a = time.time()
        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs)
        end.record()
        torch.cuda.synchronize()
        print(time.time() - a, 'flow')
        print(start.elapsed_time(end) / 1000, 'torch')
        
        ### forward-time propagation and upsampling
        outputs = []

        feat_prop_lv0 = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_lv1 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv2 = lrs.new_zeros(n, self.mid_channels, h*4, w*4)
        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*8, w*8)

        ### texture transfer
        # B, N, C, H, W = mks.size()
        # mk_lv3 = mks.float()
        # mk_lv2 = self.img_downsample_2x(mk_lv3.view(B*N, 1, H, W)).view(B, N, 1, H//2, W//2)
        # mk_lv1 = self.img_downsample_2x(mk_lv2.view(B*N, 1, H//2, W//2)).view(B, N, 1, H//4, W//4)
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_2x(lrs_lv0)
        lrs_lv3 = self.img_upsample_2x(lrs_lv3)
        lrs_lv3 = self.img_upsample_2x(lrs_lv3)

        a = time.time()
        torch.cuda.synchronize()
        start.record()
        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        # fvs = fvs * mks.float() + lrs_lv3.view(B, N, C, H*8, W*8) * (1 - mks.float())
        B, N, C, H, W = fvs.size()
        x_hr_lv1, x_hr_lv2, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3[:, :, :H, :W]), dim=1), islr=True)
        end.record()
        torch.cuda.synchronize()
        print(time.time() - a, 'encode')
        print(start.elapsed_time(end) / 1000, 'torch')

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv1.size()
        x_hr_lv1 = x_hr_lv1.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv2.size()
        x_hr_lv2 = x_hr_lv2.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)

        res_lv0_list = []
        res_lv1_list = []
        res_lv2_list = []
        res_lv3_list = []        
        
        dcn_pre_lv0_list = []
        dcn_pre_lv1_list = []
        dcn_pre_lv2_list = []
        dcn_pre_lv3_list = []

        dcn_lv0_list = []
        dcn_lv1_list = []
        dcn_lv2_list = []
        dcn_lv3_list = []
        
        flow_lv0_list = []
        flow_lv1_list = []
        flow_lv2_list = []
        flow_lv3_list = []

        last_list = []

        res_lv0_torch_list = []
        res_lv1_torch_list = []
        res_lv2_torch_list = []
        res_lv3_torch_list = []        
        
        dcn_pre_lv0_torch_list = []
        dcn_pre_lv1_torch_list = []
        dcn_pre_lv2_torch_list = []
        dcn_pre_lv3_torch_list = []

        dcn_lv0_torch_list = []
        dcn_lv1_torch_list = []
        dcn_lv2_torch_list = []
        dcn_lv3_torch_list = []
        
        flow_lv0_torch_list = []
        flow_lv1_torch_list = []
        flow_lv2_torch_list = []
        flow_lv3_torch_list = []

        last_torch_list = []

        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]

            x_hr_lv1_cur = x_hr_lv1[:, i, :, :, :]
            x_hr_lv2_cur = x_hr_lv2[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            # mk_cur_lv3 = mk_lv3[:, i, :, :, :]
            # mk_cur_lv2 = mk_lv2[:, i, :, :, :]
            # mk_cur_lv1 = mk_lv1[:, i, :, :, :]

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]

                flow_lv0 = flow
                flow_lv1 = self.img_upsample_2x(flow_lv0)
                flow_lv2 = self.img_upsample_2x(flow_lv1)
                flow_lv3 = self.img_upsample_2x(flow_lv2)

                feat_prop_lv2 = self.img_downsample_2x(feat_prop_lv3)
                feat_prop_lv1 = self.img_downsample_2x(feat_prop_lv2)
                feat_prop_lv0 = self.img_downsample_2x(feat_prop_lv1)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv0_ = flow_warp(feat_prop_lv0, flow_lv0.permute(0, 2, 3, 1))
                end.record()
                torch.cuda.synchronize()
                flow_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                flow_lv0_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv1_ = flow_warp(feat_prop_lv1, flow_lv1.permute(0, 2, 3, 1))
                end.record()
                torch.cuda.synchronize()
                flow_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                flow_lv1_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv2_ = flow_warp(feat_prop_lv2, flow_lv2.permute(0, 2, 3, 1))
                end.record()
                torch.cuda.synchronize()
                flow_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                flow_lv2_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                end.record()
                torch.cuda.synchronize()
                flow_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                flow_lv3_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv0_ = torch.cat([x_lr_lv0_cur, feat_prop_lv0_, flow_lv0], dim=1)
                feat_prop_lv0_ = self.dcn_pre_lv0(feat_prop_lv0_)
                feat_prop_lv0_ = self.dcn_block_lv0(feat_prop_lv0_)
                feat_offset_lv0 = self.dcn_offset_lv0(feat_prop_lv0_)
                feat_offset_lv0 = self.max_residue_magnitude * torch.tanh(feat_offset_lv0)
                # feat_offset_lv0 = feat_offset_lv0 + flow_lv0.flip(1).repeat(1, feat_offset_lv0.size(1) // 2, 1, 1)
                flow_lv0 = torch.cat((flow_lv0[:, 1:2, :, :], flow_lv0[:, 0:1, :, :]), dim=1)
                flow_lv0 = flow_lv0.repeat(1, feat_offset_lv0.size(1) // 2, 1, 1)
                feat_offset_lv0 = feat_offset_lv0 + flow_lv0
                feat_mask_lv0 = self.dcn_mask_lv0(feat_prop_lv0_)
                feat_mask_lv0 = torch.sigmoid(feat_mask_lv0)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_pre_lv0_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv0 = self.dcn_lv0(feat_prop_lv0, feat_offset_lv0, feat_mask_lv0)
                end.record()
                torch.cuda.synchronize()
                dcn_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_lv0_list.append(time.time() - a)
                
                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv0 = torch.cat([x_lr_lv0_cur, feat_prop_lv0], dim=1)
                feat_prop_lv0 = self.forward_resblocks_lv0(feat_prop_lv0)
                feat_prop_lv0 = self.lrelu(self.upsample0(feat_prop_lv0))
                end.record()
                torch.cuda.synchronize()
                res_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv0_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv1_ = torch.cat([feat_prop_lv0, feat_prop_lv1_, flow_lv1], dim=1)
                feat_prop_lv1_ = self.dcn_pre_lv1(feat_prop_lv1_)
                feat_prop_lv1_ = self.dcn_block_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.dcn_offset_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.max_residue_magnitude * torch.tanh(feat_offset_lv1)
                # feat_offset_lv1 = feat_offset_lv1 + flow_lv1.flip(1).repeat(1, feat_offset_lv1.size(1) // 2, 1, 1)
                flow_lv1 = torch.cat((flow_lv1[:, 1:2, :, :], flow_lv1[:, 0:1, :, :]), dim=1)
                flow_lv1 = flow_lv1.repeat(1, feat_offset_lv1.size(1) // 2, 1, 1)
                feat_offset_lv1 = feat_offset_lv1 + flow_lv1
                feat_mask_lv1 = self.dcn_mask_lv1(feat_prop_lv1_)
                feat_mask_lv1 = torch.sigmoid(feat_mask_lv1)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_pre_lv1_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv1 = self.dcn_lv1(feat_prop_lv1, feat_offset_lv1, feat_mask_lv1)
                end.record()
                torch.cuda.synchronize()
                dcn_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_lv1_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv1 = torch.cat([feat_prop_lv0, feat_prop_lv1], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                B, C, H, W = x_hr_lv1_cur.size()
                feat_prop_lv1_ = torch.cat([feat_prop_lv1[:, :, :H, :W], x_hr_lv1_cur], dim=1)
                # feat_prop_lv1_ = torch.cat([feat_prop_lv1, x_hr_lv1_cur], dim=1)
                feat_prop_lv1_ = self.conv_tttf_lv1(feat_prop_lv1_)
                # feat_prop_lv1 = mk_cur_lv1 * feat_prop_lv1_ + (1 - mk_cur_lv1) * feat_prop_lv1
                feat_prop_lv1[:, :, :H, :W] = feat_prop_lv1_[:, :, :H, :W]
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))
                end.record()
                torch.cuda.synchronize()
                res_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv1_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv2_ = torch.cat([feat_prop_lv1, feat_prop_lv2_, flow_lv2], dim=1)
                feat_prop_lv2_ = self.dcn_pre_lv2(feat_prop_lv2_)
                feat_prop_lv2_ = self.dcn_block_lv2(feat_prop_lv2_)
                feat_offset_lv2 = self.dcn_offset_lv2(feat_prop_lv2_)
                feat_offset_lv2 = self.max_residue_magnitude * torch.tanh(feat_offset_lv2)
                # feat_offset_lv2 = feat_offset_lv2 + flow_lv2.flip(1).repeat(1, feat_offset_lv2.size(1) // 2, 1, 1)
                flow_lv2 = torch.cat((flow_lv2[:, 1:2, :, :], flow_lv2[:, 0:1, :, :]), dim=1)
                flow_lv2 = flow_lv2.repeat(1, feat_offset_lv2.size(1) // 2, 1, 1)
                feat_offset_lv2 = feat_offset_lv2 + flow_lv2
                feat_mask_lv2 = self.dcn_mask_lv2(feat_prop_lv2_)
                feat_mask_lv2 = torch.sigmoid(feat_mask_lv2)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_pre_lv2_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv2 = self.dcn_lv2(feat_prop_lv2, feat_offset_lv2, feat_mask_lv2)
                end.record()
                torch.cuda.synchronize()
                dcn_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_lv2_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                B, C, H, W = x_hr_lv2_cur.size()
                feat_prop_lv2_ = torch.cat([feat_prop_lv2[:, :, :H, :W], x_hr_lv2_cur], dim=1)
                # feat_prop_lv2_ = torch.cat([feat_prop_lv2, x_hr_lv2_cur], dim=1)
                feat_prop_lv2_ = self.conv_tttf_lv2(feat_prop_lv2_)
                # feat_prop_lv2 = mk_cur_lv2 * feat_prop_lv2_ + (1 - mk_cur_lv2) * feat_prop_lv2
                feat_prop_lv2[:, :, :H, :W] = feat_prop_lv2_[:, :, :H, :W]
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))
                end.record()
                torch.cuda.synchronize()
                res_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv2_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3_ = torch.cat([feat_prop_lv2, feat_prop_lv3_, flow_lv3], dim=1)
                feat_prop_lv3_ = self.dcn_pre_lv3(feat_prop_lv3_)
                feat_prop_lv3_ = self.dcn_block_lv3(feat_prop_lv3_)
                feat_offset_lv3 = self.dcn_offset_lv3(feat_prop_lv3_)
                feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                # feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                flow_lv3 = torch.cat((flow_lv3[:, 1:2, :, :], flow_lv3[:, 0:1, :, :]), dim=1)
                flow_lv3 = flow_lv3.repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                feat_offset_lv3 = feat_offset_lv3 + flow_lv3
                feat_mask_lv3 = self.dcn_mask_lv3(feat_prop_lv3_)
                feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_pre_lv3_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = self.dcn_lv3(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)
                end.record()
                torch.cuda.synchronize()
                dcn_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_lv3_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                B, C, H, W = x_hr_lv3_cur.size()
                feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
                # feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                # feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3
                feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
                end.record()
                torch.cuda.synchronize()
                res_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv3_list.append(time.time() - a)
            
            else:
                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv0 = torch.cat([x_lr_lv0_cur, feat_prop_lv0], dim=1)
                feat_prop_lv0 = self.forward_resblocks_lv0(feat_prop_lv0)
                feat_prop_lv0 = self.lrelu(self.upsample0(feat_prop_lv0))
                end.record()
                torch.cuda.synchronize()
                res_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv0_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv1 = torch.cat([feat_prop_lv0, feat_prop_lv1], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                B, C, H, W = x_hr_lv1_cur.size()
                feat_prop_lv1_ = torch.cat([feat_prop_lv1[:, :, :H, :W], x_hr_lv1_cur], dim=1)
                # feat_prop_lv1_ = torch.cat([feat_prop_lv1, x_hr_lv1_cur], dim=1)
                feat_prop_lv1_ = self.conv_tttf_lv1(feat_prop_lv1_)
                # feat_prop_lv1 = mk_cur_lv1 * feat_prop_lv1_ + (1 - mk_cur_lv1) * feat_prop_lv1
                feat_prop_lv1[:, :, :H, :W] = feat_prop_lv1_[:, :, :H, :W]
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))
                end.record()
                torch.cuda.synchronize()
                res_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv1_list.append(time.time() - a)
            
                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                B, C, H, W = x_hr_lv2_cur.size()
                feat_prop_lv2_ = torch.cat([feat_prop_lv2[:, :, :H, :W], x_hr_lv2_cur], dim=1)
                # feat_prop_lv2_ = torch.cat([feat_prop_lv2, x_hr_lv2_cur], dim=1)
                feat_prop_lv2_ = self.conv_tttf_lv2(feat_prop_lv2_)
                # feat_prop_lv2 = mk_cur_lv2 * feat_prop_lv2_ + (1 - mk_cur_lv2) * feat_prop_lv2
                feat_prop_lv2[:, :, :H, :W] = feat_prop_lv2_[:, :, :H, :W]
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))
                end.record()
                torch.cuda.synchronize()
                res_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv2_list.append(time.time() - a)
                
                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                B, C, H, W = x_hr_lv3_cur.size()
                feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
                # feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                # feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3
                feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
                end.record()
                torch.cuda.synchronize()
                res_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv3_list.append(time.time() - a)

            a = time.time()
            torch.cuda.synchronize()
            start.record()
            out_lv3 = feat_prop_lv3
            out_lv3 = self.lrelu(self.conv_hr_lv3(out_lv3))
            out_lv3 = self.conv_last_lv3(out_lv3)
            base_lv3 = self.img_upsample_8x(lr_cur)
            out_lv3 += base_lv3
            outputs.append(out_lv3)
            end.record()
            torch.cuda.synchronize()
            last_torch_list.append(start.elapsed_time(end) / 1000)
            last_list.append(time.time() - a)

        a = 0
        print(sum(flow_lv0_list)/len(flow_lv0_list), 'flow_lv0')
        print(sum(flow_lv1_list)/len(flow_lv1_list), 'flow_lv1')
        print(sum(flow_lv2_list)/len(flow_lv2_list), 'flow_lv2')
        print(sum(flow_lv3_list)/len(flow_lv3_list), 'flow_lv3')
        print(sum(dcn_pre_lv0_list)/len(dcn_pre_lv0_list), 'dcn_pre_lv0')
        print(sum(dcn_pre_lv1_list)/len(dcn_pre_lv1_list), 'dcn_pre_lv1')
        print(sum(dcn_pre_lv2_list)/len(dcn_pre_lv2_list), 'dcn_pre_lv2')
        print(sum(dcn_pre_lv3_list)/len(dcn_pre_lv3_list), 'dcn_pre_lv3')
        print(sum(dcn_lv0_list)/len(dcn_lv0_list), 'dcn_lv0')
        print(sum(dcn_lv1_list)/len(dcn_lv1_list), 'dcn_lv1')
        print(sum(dcn_lv2_list)/len(dcn_lv2_list), 'dcn_lv2')
        print(sum(dcn_lv3_list)/len(dcn_lv3_list), 'dcn_lv3')
        print(sum(res_lv0_list)/len(res_lv0_list), 'res_lv0')
        print(sum(res_lv1_list)/len(res_lv1_list), 'res_lv1')
        print(sum(res_lv2_list)/len(res_lv2_list), 'res_lv2')
        print(sum(res_lv3_list)/len(res_lv3_list), 'res_lv3')
        print(sum(last_list)/len(last_list), 'last')
        
        a+=sum(flow_lv0_list)/len(flow_lv0_list)
        a+=sum(flow_lv1_list)/len(flow_lv1_list)
        a+=sum(flow_lv2_list)/len(flow_lv2_list)
        a+=sum(flow_lv3_list)/len(flow_lv3_list)
        a+=sum(dcn_pre_lv0_list)/len(dcn_pre_lv0_list)
        a+=sum(dcn_pre_lv1_list)/len(dcn_pre_lv1_list)
        a+=sum(dcn_pre_lv2_list)/len(dcn_pre_lv2_list)
        a+=sum(dcn_pre_lv3_list)/len(dcn_pre_lv3_list)
        a+=sum(dcn_lv0_list)/len(dcn_lv0_list)
        a+=sum(dcn_lv1_list)/len(dcn_lv1_list)
        a+=sum(dcn_lv2_list)/len(dcn_lv2_list)
        a+=sum(dcn_lv3_list)/len(dcn_lv3_list)
        a+=sum(res_lv0_list)/len(res_lv0_list)
        a+=sum(res_lv1_list)/len(res_lv1_list)
        a+=sum(res_lv2_list)/len(res_lv2_list)
        a+=sum(res_lv3_list)/len(res_lv3_list)
        a+=sum(last_list)/len(last_list)
        
        print(a, 'total')

        a = 0
        print(sum(flow_lv0_torch_list)/len(flow_lv0_torch_list), 'flow_lv0_torch')
        print(sum(flow_lv1_torch_list)/len(flow_lv1_torch_list), 'flow_lv1_torch')
        print(sum(flow_lv2_torch_list)/len(flow_lv2_torch_list), 'flow_lv2_torch')
        print(sum(flow_lv3_torch_list)/len(flow_lv3_torch_list), 'flow_lv3_torch')
        print(sum(dcn_pre_lv0_torch_list)/len(dcn_pre_lv0_torch_list), 'dcn_pre_lv0_torch')
        print(sum(dcn_pre_lv1_torch_list)/len(dcn_pre_lv1_torch_list), 'dcn_pre_lv1_torch')
        print(sum(dcn_pre_lv2_torch_list)/len(dcn_pre_lv2_torch_list), 'dcn_pre_lv2_torch')
        print(sum(dcn_pre_lv3_torch_list)/len(dcn_pre_lv3_torch_list), 'dcn_pre_lv3_torch')
        print(sum(dcn_lv0_torch_list)/len(dcn_lv0_torch_list), 'dcn_lv0_torch')
        print(sum(dcn_lv1_torch_list)/len(dcn_lv1_torch_list), 'dcn_lv1_torch')
        print(sum(dcn_lv2_torch_list)/len(dcn_lv2_torch_list), 'dcn_lv2_torch')
        print(sum(dcn_lv3_torch_list)/len(dcn_lv3_torch_list), 'dcn_lv3_torch')
        print(sum(res_lv0_torch_list)/len(res_lv0_torch_list), 'res_lv0_torch')
        print(sum(res_lv1_torch_list)/len(res_lv1_torch_list), 'res_lv1_torch')
        print(sum(res_lv2_torch_list)/len(res_lv2_torch_list), 'res_lv2_torch')
        print(sum(res_lv3_torch_list)/len(res_lv3_torch_list), 'res_lv3_torch')
        print(sum(last_torch_list)/len(last_torch_list), 'last_torch')
        
        a+=sum(flow_lv0_torch_list)/len(flow_lv0_torch_list)
        a+=sum(flow_lv1_torch_list)/len(flow_lv1_torch_list)
        a+=sum(flow_lv2_torch_list)/len(flow_lv2_torch_list)
        a+=sum(flow_lv3_torch_list)/len(flow_lv3_torch_list)
        a+=sum(dcn_pre_lv0_torch_list)/len(dcn_pre_lv0_torch_list)
        a+=sum(dcn_pre_lv1_torch_list)/len(dcn_pre_lv1_torch_list)
        a+=sum(dcn_pre_lv2_torch_list)/len(dcn_pre_lv2_torch_list)
        a+=sum(dcn_pre_lv3_torch_list)/len(dcn_pre_lv3_torch_list)
        a+=sum(dcn_lv0_torch_list)/len(dcn_lv0_torch_list)
        a+=sum(dcn_lv1_torch_list)/len(dcn_lv1_torch_list)
        a+=sum(dcn_lv2_torch_list)/len(dcn_lv2_torch_list)
        a+=sum(dcn_lv3_torch_list)/len(dcn_lv3_torch_list)
        a+=sum(res_lv0_torch_list)/len(res_lv0_torch_list)
        a+=sum(res_lv1_torch_list)/len(res_lv1_torch_list)
        a+=sum(res_lv2_torch_list)/len(res_lv2_torch_list)
        a+=sum(res_lv3_torch_list)/len(res_lv3_torch_list)
        a+=sum(last_torch_list)/len(last_torch_list)
        
        print(a, 'total_torch')

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
    def init_dcn(self):

        self.dcn_offset_lv0.weight.data.zero_()
        self.dcn_offset_lv0.bias.data.zero_()
        self.dcn_mask_lv0.weight.data.zero_()
        self.dcn_mask_lv0.bias.data.zero_()
        self.conv_identify(self.dcn_lv0.weight, self.dcn_lv0.bias)

        self.dcn_offset_lv1.weight.data.zero_()
        self.dcn_offset_lv1.bias.data.zero_()
        self.dcn_mask_lv1.weight.data.zero_()
        self.dcn_mask_lv1.bias.data.zero_()
        self.conv_identify(self.dcn_lv1.weight, self.dcn_lv1.bias)
        
        self.dcn_offset_lv2.weight.data.zero_()
        self.dcn_offset_lv2.bias.data.zero_()
        self.dcn_mask_lv2.weight.data.zero_()
        self.dcn_mask_lv2.bias.data.zero_()
        self.conv_identify(self.dcn_lv2.weight, self.dcn_lv2.bias)
        
        self.dcn_offset_lv3.weight.data.zero_()
        self.dcn_offset_lv3.bias.data.zero_()
        self.dcn_mask_lv3.weight.data.zero_()
        self.dcn_mask_lv3.bias.data.zero_()
        self.conv_identify(self.dcn_lv3.weight, self.dcn_lv3.bias)

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

# class MRCF_simple(nn.Module):

#     def __init__(self, device, mid_channels=16, num_blocks=30, spynet_pretrained=None):

#         super().__init__()

#         self.device = device
#         self.mid_channels = mid_channels
#         self.dg_num = 16
#         self.max_residue_magnitude = 10

#         # optical flow network for feature alignment
#         self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

#         self.dcn_pre = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
#         self.dcn_block = nn.Sequential(
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#         self.dcn_offset = nn.Conv2d(mid_channels, (self.dg_num//16)*2*3*3, 3, 1, 1, bias=True)
#         self.dcn_mask = nn.Conv2d(mid_channels, (self.dg_num//16)*1*3*3, 3, 1, 1, bias=True)
#         self.dcn = DCNv2(mid_channels, mid_channels, 3,
#                          stride=1, padding=1, dilation=1,
#                          deformable_groups=self.dg_num//16)
#         self.init_dcn()

#         # feature extractor
#         self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
#         self.encoder_hr = LTE.LTE_simple_lr(mid_channels)

#         self.conv_tttf = conv3x3(mid_channels * 2, mid_channels)

#         # propagation branches
#         self.forward_resblocks = ResidualBlocksWithInputConv(
#             mid_channels * 2, mid_channels, 1)

#         # upsample
#         # self.upsample = PixelShufflePack(
#             # mid_channels, mid_channels, 2, upsample_kernel=3)
#         # self.upsample = PixelShufflePack(
#         #     mid_channels, mid_channels, 4, upsample_kernel=3)
#         # self.upsample = PixelShufflePack(
#             # mid_channels, mid_channels, 8, upsample_kernel=3)
        
#         # self.upsample_post = PixelShufflePack(
#             # mid_channels, mid_channels, 2, upsample_kernel=3)
#         # self.upsample_post = PixelShufflePack(
#             # mid_channels, mid_channels, 4, upsample_kernel=3)
#         self.upsample_post = PixelShufflePack(
#             mid_channels, mid_channels, 8, upsample_kernel=3)

#         self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
#         self.conv_last = nn.Conv2d(mid_channels, 3, 3, 1, 1)

#         ### 8x settings
#         self.img_upsample_2x = nn.Upsample(
#             scale_factor=2, mode='bilinear', align_corners=False)
#         self.img_upsample_4x = nn.Upsample(
#             scale_factor=4, mode='bilinear', align_corners=False)
#         self.img_upsample_8x = nn.Upsample(
#             scale_factor=8, mode='bilinear', align_corners=False)

#         # activation function
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#     # @profile
#     def compute_flow(self, lrs):
#         """Compute optical flow using SPyNet for feature warping.

#         Note that if the input is an mirror-extended sequence, 'flows_forward'
#         is not needed, since it is equal to 'flows_backward.flip(1)'.

#         Args:
#             lrs (tensor): Input LR images with shape (n, t, c, h, w)

#         Return:
#             tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
#                 flows used for forward-time propagation (current to previous).
#                 'flows_backward' corresponds to the flows used for
#                 backward-time propagation (current to next).
#         """

#         n, t, c, h, w = lrs.size()

#         lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
#         lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

#         # flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
#         flows_backward = None
#         flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

#         return flows_forward, flows_backward

#     # @profile
#     # def forward(self, lrs, fvs, mks):
#     def forward(self, lrs, fvs):
#         """Forward function for BasicVSR.

#         Args:
#             lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

#         Returns:
#             Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
#         """

#         # print(lrs.size(), mks.size())
#         n, t, c, h, w = lrs.size()

#         ### compute optical flow
#         # torch.cuda.reset_max_memory_allocated(0)

#         torch.cuda.synchronize()
#         start.record()
#         flows_forward, flows_backward = self.compute_flow(lrs)
#         # print(torch.cuda.memory_allocated(0), '1')
#         # print(torch.cuda.max_memory_allocated(0), '1')
#         # torch.cuda.reset_max_memory_allocated(0)
#         end.record()
#         torch.cuda.synchronize()
#         print(start.elapsed_time(end) / 1000, 'flow')
        
#         ### forward-time propagation and upsampling
#         outputs = []

#         feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h, w)
#         # feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
#         # feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*4, w*4)
#         # feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*8, w*8)
#         # print(torch.cuda.memory_allocated(0), '2')
#         # print(torch.cuda.max_memory_allocated(0), '2')
#         # torch.cuda.reset_max_memory_allocated(0)

#         ### texture transfer
#         # B, N, C, H, W = mks.size()
#         # mk_lv3 = mks.float()

#         torch.cuda.synchronize()
#         start.record()
#         B, N, C, H, W = lrs.size()
#         lrs_lv0 = lrs.view(B*N, C, H, W)
#         # lrs_lv3 = self.img_upsample_8x(lrs_lv0)
#         # print(torch.cuda.memory_allocated(0), '3')
#         # print(torch.cuda.max_memory_allocated(0), '3')
#         # torch.cuda.reset_max_memory_allocated(0)

#         _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)
#         # print(torch.cuda.memory_allocated(0), '4')
#         # print(torch.cuda.max_memory_allocated(0), '4')
#         # torch.cuda.reset_max_memory_allocated(0)
#         end.record()
#         torch.cuda.synchronize()
#         print(start.elapsed_time(end) / 1000, 'en_lr')

#         # lrs_lv3_view = lrs_lv3.view(B, N, C, H*8, W*8)
#         # mks_float = mks.float()
#         # fvs = (fvs * mks_float + lrs_lv3_view * (1 - mks_float))
#         # print(torch.cuda.memory_allocated(0), '5')
#         # print(torch.cuda.max_memory_allocated(0), '5')
#         # torch.cuda.reset_max_memory_allocated(0)

#         torch.cuda.synchronize()
#         start.record()
#         B, N, C, H, W = fvs.size()
#         # _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)
#         _, _, x_hr_lv3 = self.encoder_hr(fvs.view(B*N, C, H, W), islr=True)
#         # print(torch.cuda.memory_allocated(0), '6')
#         # print(torch.cuda.max_memory_allocated(0), '6')
#         # torch.cuda.reset_max_memory_allocated(0)
#         end.record()
#         torch.cuda.synchronize()
#         print(start.elapsed_time(end) / 1000, 'en_hr')

#         _, C, H, W = x_lr_lv0.size()
#         x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

#         _, C, H, W = x_hr_lv3.size()
#         x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        
#         res_torch_list = []        
#         dcn_pre_torch_list = []
#         dcn_torch_list = []
#         flow_torch_list = []

#         last_torch_list = []

#         for i in range(0, t):
#             lr_cur = lrs[:, i, :, :, :]
#             x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]

#             x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
#             # mk_cur = mks[:, i, :, :, :]

#             feat_prop_lv0 = x_lr_lv0_cur
#             # feat_prop_lv0 = self.upsample(x_lr_lv0_cur)
#             # print(torch.cuda.memory_allocated(0), '14')
#             # print(torch.cuda.max_memory_allocated(0), '14')
#             # torch.cuda.reset_max_memory_allocated(0)

#             if i > 0:  # no warping required for the first timestep
#                 flow = flows_forward[:, i - 1, :, :, :]
                
#                 torch.cuda.synchronize()
#                 start.record()
#                 flow_lv3 = flow
#                 # flow_lv3 = self.img_upsample_2x(flow) * 2.
#                 # flow_lv3 = self.img_upsample_4x(flow) * 4.
#                 # flow_lv3 = self.img_upsample_8x(flow) * 8.

#                 feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
#                 # print(torch.cuda.memory_allocated(0), '7')
#                 # print(torch.cuda.max_memory_allocated(0), '7')
#                 # torch.cuda.reset_max_memory_allocated(0)
#                 end.record()
#                 torch.cuda.synchronize()
#                 flow_torch_list.append(start.elapsed_time(end) / 1000)

#                 torch.cuda.synchronize()
#                 start.record()
#                 feat_prop_lv3_ = torch.cat([feat_prop_lv0, feat_prop_lv3_, flow_lv3], dim=1)
#                 feat_prop_lv3_ = self.dcn_pre(feat_prop_lv3_)
#                 # print(torch.cuda.memory_allocated(0), '8')
#                 # print(torch.cuda.max_memory_allocated(0), '8')
#                 # torch.cuda.reset_max_memory_allocated(0)
    
#                 feat_prop_lv3_ = self.dcn_block(feat_prop_lv3_)
#                 # print(torch.cuda.memory_allocated(0), '9')
#                 # print(torch.cuda.max_memory_allocated(0), '9')
#                 # torch.cuda.reset_max_memory_allocated(0)
    
#                 feat_offset_lv3 = self.dcn_offset(feat_prop_lv3_)
#                 # print(torch.cuda.memory_allocated(0), '10')
#                 # print(torch.cuda.max_memory_allocated(0), '10')
#                 # torch.cuda.reset_max_memory_allocated(0)
    
#                 feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
#                 feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
#                 # print(torch.cuda.memory_allocated(0), '11')
#                 # print(torch.cuda.max_memory_allocated(0), '11')
#                 # torch.cuda.reset_max_memory_allocated(0)
    
#                 feat_mask_lv3 = self.dcn_mask(feat_prop_lv3_)
#                 feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
#                 # print(torch.cuda.memory_allocated(0), '12')
#                 # print(torch.cuda.max_memory_allocated(0), '12')
#                 # torch.cuda.reset_max_memory_allocated(0)
#                 end.record()
#                 torch.cuda.synchronize()
#                 dcn_pre_torch_list.append(start.elapsed_time(end) / 1000)
                
#                 torch.cuda.synchronize()
#                 start.record()    
#                 feat_prop_lv3 = self.dcn(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)
#                 # print(torch.cuda.memory_allocated(0), '13')
#                 # print(torch.cuda.max_memory_allocated(0), '13')
#                 # torch.cuda.reset_max_memory_allocated(0)
#                 end.record()
#                 torch.cuda.synchronize()
#                 dcn_torch_list.append(start.elapsed_time(end) / 1000)

#                 # del flow
#                 # del flow_lv3
#                 # del feat_prop_lv3_
#                 # del feat_mask_lv3
#                 # del feat_offset_lv3
#                 # print(torch.cuda.memory_allocated(0), '13.1')
            
#             torch.cuda.synchronize()
#             start.record()
#             feat_prop_lv3 = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
#             feat_prop_lv3 = self.forward_resblocks(feat_prop_lv3)
#             # print(torch.cuda.memory_allocated(0), '14')
#             # print(torch.cuda.max_memory_allocated(0), '14')
#             # torch.cuda.reset_max_memory_allocated(0)

#             # feat_prop_lv3__ = self.upsample_post(feat_prop_lv3)
#             # feat_prop_lv3_ = torch.cat([feat_prop_lv3__, x_hr_lv3_cur], dim=1)
#             feat_prop_lv3_ = self.upsample_post(feat_prop_lv3)
#             _, _, h, w = x_hr_lv3_cur.size()
#             feat_prop_lv3__ = torch.cat([feat_prop_lv3_[:, :, :h, :w], x_hr_lv3_cur], dim=1)

#             # feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
#             # _, _, h, w = x_hr_lv3_cur.size()
#             # feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :h, :w], x_hr_lv3_cur], dim=1)

#             feat_prop_lv3__ = self.conv_tttf(feat_prop_lv3__)
#             # feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
#             # print(torch.cuda.memory_allocated(0), '15')
#             # print(torch.cuda.max_memory_allocated(0), '15')
#             # torch.cuda.reset_max_memory_allocated(0)

#             # feat_prop_lv3_ = mk_cur.float() * feat_prop_lv3_ + (1 - mk_cur.float()) * feat_prop_lv3__
#             # feat_prop_lv3 = mk_cur.float() * feat_prop_lv3_ + (1 - mk_cur.float()) * feat_prop_lv3
#             feat_prop_lv3_[:, :, :h, :w] = feat_prop_lv3__
#             # feat_prop_lv3[:, :, :h, :w] = feat_prop_lv3_[:, :, :h, :w]

#             end.record()
#             torch.cuda.synchronize()
#             res_torch_list.append(start.elapsed_time(end) / 1000)

#             torch.cuda.synchronize()    
#             start.record()
#             feat_prop_lv3_ = self.lrelu(self.conv_hr(feat_prop_lv3_))
#             # feat_prop_lv3 = self.lrelu(self.conv_hr(feat_prop_lv3))
#             # print(torch.cuda.memory_allocated(0), '16')
#             # print(torch.cuda.max_memory_allocated(0), '16')
#             # torch.cuda.reset_max_memory_allocated(0)

#             out = feat_prop_lv3_
#             # out = feat_prop_lv3

#             out = self.conv_last(out)
#             base = self.img_upsample_8x(lr_cur)
#             out += base
#             outputs.append(out.cpu())
#             end.record()
#             torch.cuda.synchronize()
#             last_torch_list.append(start.elapsed_time(end) / 1000)

#             # print(torch.cuda.memory_allocated(0), '17')
#             # print(torch.cuda.max_memory_allocated(0), '17')
#             # torch.cuda.reset_max_memory_allocated(0)
#             # del out
#             # print(torch.cuda.memory_allocated(0), '18')
#             # del feat_prop_lv3_
#             # print(torch.cuda.memory_allocated(0), '18')
#             # del feat_prop_lv3__
#             # print(torch.cuda.memory_allocated(0), '18')
        
#         a = 0
#         print(sum(flow_torch_list)/len(flow_torch_list), 'flow_torch')
#         print(sum(dcn_pre_torch_list)/len(dcn_pre_torch_list), 'dcn_pre_torch')
#         print(sum(dcn_torch_list)/len(dcn_torch_list), 'dcn_torch')
#         print(sum(res_torch_list)/len(res_torch_list), 'res_torch')
#         print(sum(last_torch_list)/len(last_torch_list), 'last_torch')
        
#         a+=sum(flow_torch_list)/len(flow_torch_list)
#         a+=sum(dcn_pre_torch_list)/len(dcn_pre_torch_list)
#         a+=sum(dcn_torch_list)/len(dcn_torch_list)
#         a+=sum(res_torch_list)/len(res_torch_list)
#         a+=sum(last_torch_list)/len(last_torch_list)
        
#         print(a, 'total_torch')

#         return torch.stack(outputs, dim=1)

#     def init_weights(self, pretrained=None, strict=True):
#         """Init weights for models.

#         Args:
#             pretrained (str, optional): Path for pretrained weights. If given
#                 None, pretrained weights will not be loaded. Defaults: None.
#             strict (boo, optional): Whether strictly load the pretrained model.
#                 Defaults to True.
#         """
#         if isinstance(pretrained, str):
#             # logger = get_root_logger()
#             # load_checkpoint(self, pretrained, strict=strict, logger=logger)
#             model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
#             model_state_dict = self.state_dict()
#             model_state_dict.update(model_state_dict_save)
#             self.load_state_dict(model_state_dict, strict=strict)
#         elif pretrained is not None:
#             raise TypeError(f'"pretrained" must be a str or None. '
#                             f'But received {type(pretrained)}.')
#     def init_dcn(self):

#         self.dcn_offset.weight.data.zero_()
#         self.dcn_offset.bias.data.zero_()
#         self.dcn_mask.weight.data.zero_()
#         self.dcn_mask.bias.data.zero_()
#         self.conv_identify(self.dcn.weight, self.dcn.bias)

#     def conv_identify(self, weight, bias):
#         weight.data.zero_()
#         bias.data.zero_()
#         o, i, h, w = weight.shape
#         y = h//2
#         x = w//2
#         for p in range(i):
#             for q in range(o):
#                 if p == q:
#                     weight.data[q, p, y, x] = 1.0

class MRCF_CRA_x8_v2(nn.Module):

    def __init__(self, device, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.dg_num = 16
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

        self.dcn_pre_lv1 = nn.Conv2d((mid_channels//1)*2+2, mid_channels//1, 3, 1, 1, bias=True)
        self.dcn_pre_lv2 = nn.Conv2d((mid_channels//2)*2+2, mid_channels//2, 3, 1, 1, bias=True)
        self.dcn_pre_lv3 = nn.Conv2d((mid_channels//4)*2+2, mid_channels//4, 3, 1, 1, bias=True)

        self.dcn_block_lv1 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.dcn_offset_lv1 = nn.Conv2d(mid_channels//1, (self.dg_num)*2*3*3, 3, 1, 1, bias=True)
        self.dcn_offset_lv2 = nn.Conv2d(mid_channels//2, 2, 3, 1, 1, bias=True)
        self.dcn_offset_lv3 = nn.Conv2d(mid_channels//4, 2, 3, 1, 1, bias=True)

        self.dcn_mask_lv1 = nn.Conv2d(mid_channels, (self.dg_num)*1*3*3, 3, 1, 1, bias=True)
        
        self.dcn_lv1 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num)
        self.init_dcn()

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_v1(mid_channels)

        self.conv_tttf_lv1 = conv3x3((mid_channels//1)*2, mid_channels//1)
        self.conv_tttf_lv2 = conv3x3((mid_channels//2)*2, mid_channels//2)
        self.conv_tttf_lv3 = conv3x3((mid_channels//4)*2, mid_channels//4)

        # propagation branches
        self.forward_resblocks_lv0 = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        self.forward_resblocks_lv1 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)
        self.forward_resblocks_lv2 = ResidualBlocksWithInputConv(
            (mid_channels // 2) * 2, mid_channels // 2, 1)
        self.forward_resblocks_lv3 = ResidualBlocksWithInputConv(
            (mid_channels // 4) * 2, mid_channels // 4, 1)
        
        # downsample
        self.downsample0 = PixelUnShufflePack(
            mid_channels // 4, mid_channels // 2, 2, downsample_kernel=3)
        self.downsample1 = PixelUnShufflePack(
            mid_channels // 2, mid_channels, 2, downsample_kernel=3)
        # self.downsample2 = PixelUnShufflePack(
            # mid_channels // 4, mid_channels, 2, downsample_kernel=3)

        # upsample
        self.upsample0 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels // 2, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels // 2, mid_channels // 4, 2, upsample_kernel=3)

        self.conv_hr_lv3 = nn.Conv2d(mid_channels // 4, mid_channels // 4, 3, 1, 1)
        self.conv_last_lv3 = nn.Conv2d(mid_channels // 4, 3, 3, 1, 1)

        ### 4x settings
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_downsample_2x = nn.Upsample(
            scale_factor=0.5, mode='bilinear', align_corners=False)
        ### 8x settings
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)
        self.img_downsample_8x = nn.Upsample(
            scale_factor=0.125, mode='bilinear', align_corners=False)

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
    # @profile
    def forward(self, lrs, fvs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        # print(lrs.size(), mks.size())
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        a = time.time()
        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs)
        end.record()
        torch.cuda.synchronize()
        print(time.time() - a, 'flow')
        print(start.elapsed_time(end) / 1000, 'torch')
        
        ### forward-time propagation and upsampling
        outputs = []

        feat_prop_lv0 = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_lv1 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv2 = lrs.new_zeros(n, self.mid_channels // 2, h*4, w*4)
        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels // 4, h*8, w*8)

        ### texture transfer
        # B, N, C, H, W = mks.size()
        # mk_lv3 = mks.float()
        # mk_lv2 = self.img_downsample_2x(mk_lv3.view(B*N, 1, H, W)).view(B, N, 1, H//2, W//2)
        # mk_lv1 = self.img_downsample_2x(mk_lv2.view(B*N, 1, H//2, W//2)).view(B, N, 1, H//4, W//4)
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        a = time.time()
        torch.cuda.synchronize()
        start.record()
        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        # fvs = fvs * mks.float() + lrs_lv3.view(B, N, C, H*8, W*8) * (1 - mks.float())
        B, N, C, H, W = fvs.size()
        x_hr_lv1, x_hr_lv2, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3[:, :, :H, :W]), dim=1), islr=True)
        end.record()
        torch.cuda.synchronize()
        print(time.time() - a, 'encode')
        print(start.elapsed_time(end) / 1000, 'torch')

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv1.size()
        x_hr_lv1 = x_hr_lv1.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv2.size()
        x_hr_lv2 = x_hr_lv2.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)

        res_lv0_list = []
        res_lv1_list = []
        res_lv2_list = []
        res_lv3_list = []        
        
        dcn_pre_lv0_list = []
        dcn_pre_lv1_list = []
        dcn_pre_lv2_list = []
        dcn_pre_lv3_list = []

        dcn_lv0_list = []
        dcn_lv1_list = []
        dcn_lv2_list = []
        dcn_lv3_list = []
        
        flow_lv0_list = []
        flow_lv1_list = []
        flow_lv2_list = []
        flow_lv3_list = []

        last_list = []

        res_lv0_torch_list = []
        res_lv1_torch_list = []
        res_lv2_torch_list = []
        res_lv3_torch_list = []        
        
        dcn_pre_lv0_torch_list = []
        dcn_pre_lv1_torch_list = []
        dcn_pre_lv2_torch_list = []
        dcn_pre_lv3_torch_list = []

        dcn_lv0_torch_list = []
        dcn_lv1_torch_list = []
        dcn_lv2_torch_list = []
        dcn_lv3_torch_list = []
        
        flow_lv0_torch_list = []
        flow_lv1_torch_list = []
        flow_lv2_torch_list = []
        flow_lv3_torch_list = []

        last_torch_list = []

        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]

            x_hr_lv1_cur = x_hr_lv1[:, i, :, :, :]
            x_hr_lv2_cur = x_hr_lv2[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            # mk_cur_lv3 = mk_lv3[:, i, :, :, :]
            # mk_cur_lv2 = mk_lv2[:, i, :, :, :]
            # mk_cur_lv1 = mk_lv1[:, i, :, :, :]

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                # flow_lv0 = flow
                flow_lv1 = self.img_upsample_2x(flow)
                flow_lv2 = self.img_upsample_2x(flow_lv1)
                flow_lv3 = self.img_upsample_2x(flow_lv2)

                feat_prop_lv2 = self.downsample0(feat_prop_lv3)
                feat_prop_lv1 = self.downsample1(feat_prop_lv2)

                # a = time.time()
                # torch.cuda.synchronize()
                # start.record()
                # feat_prop_lv0_ = flow_warp(feat_prop_lv0, flow_lv0.permute(0, 2, 3, 1))
                # end.record()
                # torch.cuda.synchronize()
                # flow_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                # flow_lv0_list.append(time.time() - a)

                feat_prop_lv1_ = flow_warp(feat_prop_lv1, flow_lv1.permute(0, 2, 3, 1))
                end.record()
                torch.cuda.synchronize()
                flow_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                flow_lv1_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                # feat_prop_lv2_ = flow_warp(feat_prop_lv2, flow_lv2.permute(0, 2, 3, 1))
                feat_prop_lv2_ = self.upsample1(feat_prop_lv1_)
                end.record()
                torch.cuda.synchronize()
                flow_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                flow_lv2_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                # feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                feat_prop_lv3_ = self.upsample2(feat_prop_lv2_)
                end.record()
                torch.cuda.synchronize()
                flow_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                flow_lv3_list.append(time.time() - a)

                # a = time.time()
                # torch.cuda.synchronize()
                # start.record()
                # feat_prop_lv0_ = torch.cat([x_lr_lv0_cur, feat_prop_lv0_, flow_lv0], dim=1)
                # feat_prop_lv0_ = self.dcn_pre_lv0(feat_prop_lv0_)
                # feat_prop_lv0_ = self.dcn_block_lv0(feat_prop_lv0_)
                # feat_offset_lv0 = self.dcn_offset_lv0(feat_prop_lv0_)
                # feat_offset_lv0 = self.max_residue_magnitude * torch.tanh(feat_offset_lv0)
                # # feat_offset_lv0 = feat_offset_lv0 + flow_lv0.flip(1).repeat(1, feat_offset_lv0.size(1) // 2, 1, 1)
                # flow_lv0 = torch.cat((flow_lv0[:, 1:2, :, :], flow_lv0[:, 0:1, :, :]), dim=1)
                # flow_lv0 = flow_lv0.repeat(1, feat_offset_lv0.size(1) // 2, 1, 1)
                # feat_offset_lv0 = feat_offset_lv0 + flow_lv0
                # feat_mask_lv0 = self.dcn_mask_lv0(feat_prop_lv0_)
                # feat_mask_lv0 = torch.sigmoid(feat_mask_lv0)
                # end.record()
                # torch.cuda.synchronize()
                # dcn_pre_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                # dcn_pre_lv0_list.append(time.time() - a)

                # a = time.time()
                # torch.cuda.synchronize()
                # start.record()
                # feat_prop_lv0 = self.dcn_lv0(feat_prop_lv0, feat_offset_lv0, feat_mask_lv0)
                # end.record()
                # torch.cuda.synchronize()
                # dcn_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                # dcn_lv0_list.append(time.time() - a)
                
                a = time.time()
                torch.cuda.synchronize()
                start.record()
                x_lr_lv0_cur = self.forward_resblocks_lv0(x_lr_lv0_cur)
                x_lr_lv0_cur = self.lrelu(self.upsample0(x_lr_lv0_cur))
                end.record()
                torch.cuda.synchronize()
                res_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv0_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv1_ = torch.cat([x_lr_lv0_cur, feat_prop_lv1_, flow_lv1], dim=1)
                feat_prop_lv1_ = self.dcn_pre_lv1(feat_prop_lv1_)
                feat_prop_lv1_ = self.dcn_block_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.dcn_offset_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.max_residue_magnitude * torch.tanh(feat_offset_lv1)
                # feat_offset_lv1 = feat_offset_lv1 + flow_lv1.flip(1).repeat(1, feat_offset_lv1.size(1) // 2, 1, 1)
                flow_lv1 = torch.cat((flow_lv1[:, 1:2, :, :], flow_lv1[:, 0:1, :, :]), dim=1)
                flow_lv1 = flow_lv1.repeat(1, feat_offset_lv1.size(1) // 2, 1, 1)
                feat_offset_lv1 = feat_offset_lv1 + flow_lv1
                feat_mask_lv1 = self.dcn_mask_lv1(feat_prop_lv1_)
                feat_mask_lv1 = torch.sigmoid(feat_mask_lv1)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_pre_lv1_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv1 = self.dcn_lv1(feat_prop_lv1, feat_offset_lv1, feat_mask_lv1)
                end.record()
                torch.cuda.synchronize()
                dcn_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_lv1_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv1 = torch.cat([x_lr_lv0_cur, feat_prop_lv1], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                B, C, H, W = x_hr_lv1_cur.size()
                feat_prop_lv1_ = torch.cat([feat_prop_lv1[:, :, :H, :W], x_hr_lv1_cur], dim=1)
                # feat_prop_lv1_ = torch.cat([feat_prop_lv1, x_hr_lv1_cur], dim=1)
                feat_prop_lv1_ = self.conv_tttf_lv1(feat_prop_lv1_)
                # feat_prop_lv1 = mk_cur_lv1 * feat_prop_lv1_ + (1 - mk_cur_lv1) * feat_prop_lv1
                feat_prop_lv1[:, :, :H, :W] = feat_prop_lv1_[:, :, :H, :W]
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))
                end.record()
                torch.cuda.synchronize()
                res_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv1_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv2_ = torch.cat([feat_prop_lv1, feat_prop_lv2_, flow_lv2], dim=1)
                feat_prop_lv2_ = self.dcn_pre_lv2(feat_prop_lv2_)
                # feat_prop_lv2_ = self.dcn_block(feat_prop_lv2_)
                feat_offset_lv2 = self.dcn_offset_lv2(feat_prop_lv2_)
                feat_offset_lv2 = self.max_residue_magnitude * torch.tanh(feat_offset_lv2)
                flow_lv2 = feat_offset_lv2 + flow_lv2
                feat_prop_lv2 = flow_warp(feat_prop_lv2, flow_lv2.permute(0, 2, 3, 1))
                # flow_lv2 = torch.cat((flow_lv2[:, 1:2, :, :], flow_lv2[:, 0:1, :, :]), dim=1)
                # flow_lv2 = flow_lv2.repeat(1, feat_offset_lv2.size(1) // 2, 1, 1)
                # feat_offset_lv2 = feat_offset_lv2 + flow_lv2
                # feat_mask_lv2 = self.dcn_mask_lv2(feat_prop_lv2_)
                # feat_mask_lv2 = torch.sigmoid(feat_mask_lv2)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_pre_lv2_list.append(time.time() - a)

                # a = time.time()
                # torch.cuda.synchronize()
                # start.record()
                # feat_prop_lv2 = self.dcn_lv2(feat_prop_lv2, feat_offset_lv2, feat_mask_lv2)
                # end.record()
                # torch.cuda.synchronize()
                # dcn_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                # dcn_lv2_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                B, C, H, W = x_hr_lv2_cur.size()
                feat_prop_lv2_ = torch.cat([feat_prop_lv2[:, :, :H, :W], x_hr_lv2_cur], dim=1)
                # feat_prop_lv2_ = torch.cat([feat_prop_lv2, x_hr_lv2_cur], dim=1)
                feat_prop_lv2_ = self.conv_tttf_lv2(feat_prop_lv2_)
                # feat_prop_lv2 = mk_cur_lv2 * feat_prop_lv2_ + (1 - mk_cur_lv2) * feat_prop_lv2
                feat_prop_lv2[:, :, :H, :W] = feat_prop_lv2_[:, :, :H, :W]
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))
                end.record()
                torch.cuda.synchronize()
                res_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv2_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3_ = torch.cat([feat_prop_lv2, feat_prop_lv3_, flow_lv3], dim=1)
                feat_prop_lv3_ = self.dcn_pre_lv3(feat_prop_lv3_)
                # feat_prop_lv3_ = self.dcn_block(feat_prop_lv3_)
                feat_offset_lv3 = self.dcn_offset_lv3(feat_prop_lv3_)
                feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                feat_offset_lv3 = feat_offset_lv3 + flow_lv3
                feat_prop_lv3 = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                # flow_lv3 = torch.cat((flow_lv3[:, 1:2, :, :], flow_lv3[:, 0:1, :, :]), dim=1)
                # flow_lv3 = flow_lv3.repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                # feat_offset_lv3 = feat_offset_lv3 + flow_lv3
                # feat_mask_lv3 = self.dcn_mask_lv3(feat_prop_lv3_)
                # feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_pre_lv3_list.append(time.time() - a)

                # a = time.time()
                # torch.cuda.synchronize()
                # start.record()
                # feat_prop_lv3 = self.dcn_lv3(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)
                # end.record()
                # torch.cuda.synchronize()
                # dcn_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                # dcn_lv3_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                B, C, H, W = x_hr_lv3_cur.size()
                feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
                # feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                # feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3
                feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
                end.record()
                torch.cuda.synchronize()
                res_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv3_list.append(time.time() - a)
            
            else:
                a = time.time()
                torch.cuda.synchronize()
                start.record()
                x_lr_lv0_cur = self.forward_resblocks_lv0(x_lr_lv0_cur)
                x_lr_lv0_cur = self.lrelu(self.upsample0(x_lr_lv0_cur))
                end.record()
                torch.cuda.synchronize()
                res_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv0_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv1 = torch.cat([x_lr_lv0_cur, feat_prop_lv1], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                B, C, H, W = x_hr_lv1_cur.size()
                feat_prop_lv1_ = torch.cat([feat_prop_lv1[:, :, :H, :W], x_hr_lv1_cur], dim=1)
                # feat_prop_lv1_ = torch.cat([feat_prop_lv1, x_hr_lv1_cur], dim=1)
                feat_prop_lv1_ = self.conv_tttf_lv1(feat_prop_lv1_)
                # feat_prop_lv1 = mk_cur_lv1 * feat_prop_lv1_ + (1 - mk_cur_lv1) * feat_prop_lv1
                feat_prop_lv1[:, :, :H, :W] = feat_prop_lv1_[:, :, :H, :W]
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))
                end.record()
                torch.cuda.synchronize()
                res_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv1_list.append(time.time() - a)
            
                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                B, C, H, W = x_hr_lv2_cur.size()
                feat_prop_lv2_ = torch.cat([feat_prop_lv2[:, :, :H, :W], x_hr_lv2_cur], dim=1)
                # feat_prop_lv2_ = torch.cat([feat_prop_lv2, x_hr_lv2_cur], dim=1)
                feat_prop_lv2_ = self.conv_tttf_lv2(feat_prop_lv2_)
                # feat_prop_lv2 = mk_cur_lv2 * feat_prop_lv2_ + (1 - mk_cur_lv2) * feat_prop_lv2
                feat_prop_lv2[:, :, :H, :W] = feat_prop_lv2_[:, :, :H, :W]
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))
                end.record()
                torch.cuda.synchronize()
                res_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv2_list.append(time.time() - a)
                
                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                B, C, H, W = x_hr_lv3_cur.size()
                feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
                # feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                # feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3
                feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
                end.record()
                torch.cuda.synchronize()
                res_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv3_list.append(time.time() - a)

            a = time.time()
            torch.cuda.synchronize()
            start.record()
            out_lv3 = feat_prop_lv3
            out_lv3 = self.lrelu(self.conv_hr_lv3(out_lv3))
            out_lv3 = self.conv_last_lv3(out_lv3)
            base_lv3 = self.img_upsample_8x(lr_cur)
            out_lv3 += base_lv3
            outputs.append(out_lv3)
            end.record()
            torch.cuda.synchronize()
            last_torch_list.append(start.elapsed_time(end) / 1000)
            last_list.append(time.time() - a)

        a = 0
        # print(sum(flow_lv0_list)/len(flow_lv0_list), 'flow_lv0')
        print(sum(flow_lv1_list)/len(flow_lv1_list), 'flow_lv1')
        print(sum(flow_lv2_list)/len(flow_lv2_list), 'flow_lv2')
        print(sum(flow_lv3_list)/len(flow_lv3_list), 'flow_lv3')
        # print(sum(dcn_pre_lv0_list)/len(dcn_pre_lv0_list), 'dcn_pre_lv0')
        print(sum(dcn_pre_lv1_list)/len(dcn_pre_lv1_list), 'dcn_pre_lv1')
        print(sum(dcn_pre_lv2_list)/len(dcn_pre_lv2_list), 'dcn_pre_lv2')
        print(sum(dcn_pre_lv3_list)/len(dcn_pre_lv3_list), 'dcn_pre_lv3')
        # print(sum(dcn_lv0_list)/len(dcn_lv0_list), 'dcn_lv0')
        print(sum(dcn_lv1_list)/len(dcn_lv1_list), 'dcn_lv1')
        # print(sum(dcn_lv2_list)/len(dcn_lv2_list), 'dcn_lv2')
        # print(sum(dcn_lv3_list)/len(dcn_lv3_list), 'dcn_lv3')
        print(sum(res_lv0_list)/len(res_lv0_list), 'res_lv0')
        print(sum(res_lv1_list)/len(res_lv1_list), 'res_lv1')
        print(sum(res_lv2_list)/len(res_lv2_list), 'res_lv2')
        print(sum(res_lv3_list)/len(res_lv3_list), 'res_lv3')
        print(sum(last_list)/len(last_list), 'last')
        
        # a+=sum(flow_lv0_list)/len(flow_lv0_list)
        a+=sum(flow_lv1_list)/len(flow_lv1_list)
        a+=sum(flow_lv2_list)/len(flow_lv2_list)
        a+=sum(flow_lv3_list)/len(flow_lv3_list)
        # a+=sum(dcn_pre_lv0_list)/len(dcn_pre_lv0_list)
        a+=sum(dcn_pre_lv1_list)/len(dcn_pre_lv1_list)
        a+=sum(dcn_pre_lv2_list)/len(dcn_pre_lv2_list)
        a+=sum(dcn_pre_lv3_list)/len(dcn_pre_lv3_list)
        # a+=sum(dcn_lv0_list)/len(dcn_lv0_list)
        a+=sum(dcn_lv1_list)/len(dcn_lv1_list)
        # a+=sum(dcn_lv2_list)/len(dcn_lv2_list)
        # a+=sum(dcn_lv3_list)/len(dcn_lv3_list)
        a+=sum(res_lv0_list)/len(res_lv0_list)
        a+=sum(res_lv1_list)/len(res_lv1_list)
        a+=sum(res_lv2_list)/len(res_lv2_list)
        a+=sum(res_lv3_list)/len(res_lv3_list)
        a+=sum(last_list)/len(last_list)
        
        print(a, 'total')

        a = 0
        # print(sum(flow_lv0_torch_list)/len(flow_lv0_torch_list), 'flow_lv0_torch')
        print(sum(flow_lv1_torch_list)/len(flow_lv1_torch_list), 'flow_lv1_torch')
        print(sum(flow_lv2_torch_list)/len(flow_lv2_torch_list), 'flow_lv2_torch')
        print(sum(flow_lv3_torch_list)/len(flow_lv3_torch_list), 'flow_lv3_torch')
        # print(sum(dcn_pre_lv0_torch_list)/len(dcn_pre_lv0_torch_list), 'dcn_pre_lv0_torch')
        print(sum(dcn_pre_lv1_torch_list)/len(dcn_pre_lv1_torch_list), 'dcn_pre_lv1_torch')
        print(sum(dcn_pre_lv2_torch_list)/len(dcn_pre_lv2_torch_list), 'dcn_pre_lv2_torch')
        print(sum(dcn_pre_lv3_torch_list)/len(dcn_pre_lv3_torch_list), 'dcn_pre_lv3_torch')
        # print(sum(dcn_lv0_torch_list)/len(dcn_lv0_torch_list), 'dcn_lv0_torch')
        print(sum(dcn_lv1_torch_list)/len(dcn_lv1_torch_list), 'dcn_lv1_torch')
        # print(sum(dcn_lv2_torch_list)/len(dcn_lv2_torch_list), 'dcn_lv2_torch')
        # print(sum(dcn_lv3_torch_list)/len(dcn_lv3_torch_list), 'dcn_lv3_torch')
        print(sum(res_lv0_torch_list)/len(res_lv0_torch_list), 'res_lv0_torch')
        print(sum(res_lv1_torch_list)/len(res_lv1_torch_list), 'res_lv1_torch')
        print(sum(res_lv2_torch_list)/len(res_lv2_torch_list), 'res_lv2_torch')
        print(sum(res_lv3_torch_list)/len(res_lv3_torch_list), 'res_lv3_torch')
        print(sum(last_torch_list)/len(last_torch_list), 'last_torch')
        
        # a+=sum(flow_lv0_torch_list)/len(flow_lv0_torch_list)
        a+=sum(flow_lv1_torch_list)/len(flow_lv1_torch_list)
        a+=sum(flow_lv2_torch_list)/len(flow_lv2_torch_list)
        a+=sum(flow_lv3_torch_list)/len(flow_lv3_torch_list)
        # a+=sum(dcn_pre_lv0_torch_list)/len(dcn_pre_lv0_torch_list)
        a+=sum(dcn_pre_lv1_torch_list)/len(dcn_pre_lv1_torch_list)
        a+=sum(dcn_pre_lv2_torch_list)/len(dcn_pre_lv2_torch_list)
        a+=sum(dcn_pre_lv3_torch_list)/len(dcn_pre_lv3_torch_list)
        # a+=sum(dcn_lv0_torch_list)/len(dcn_lv0_torch_list)
        a+=sum(dcn_lv1_torch_list)/len(dcn_lv1_torch_list)
        # a+=sum(dcn_lv2_torch_list)/len(dcn_lv2_torch_list)
        # a+=sum(dcn_lv3_torch_list)/len(dcn_lv3_torch_list)
        a+=sum(res_lv0_torch_list)/len(res_lv0_torch_list)
        a+=sum(res_lv1_torch_list)/len(res_lv1_torch_list)
        a+=sum(res_lv2_torch_list)/len(res_lv2_torch_list)
        a+=sum(res_lv3_torch_list)/len(res_lv3_torch_list)
        a+=sum(last_torch_list)/len(last_torch_list)
        
        print(a, 'total_torch')

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
    def init_dcn(self):
        
        self.dcn_offset_lv1.weight.data.zero_()
        self.dcn_offset_lv1.bias.data.zero_()
        self.dcn_mask_lv1.weight.data.zero_()
        self.dcn_mask_lv1.bias.data.zero_()
        self.conv_identify(self.dcn_lv1.weight, self.dcn_lv1.bias)

        self.dcn_offset_lv2.weight.data.zero_()
        self.dcn_offset_lv2.bias.data.zero_()

        self.dcn_offset_lv3.weight.data.zero_()
        self.dcn_offset_lv3.bias.data.zero_()

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

class MRCF_CRA_x8_v3(nn.Module):

    def __init__(self, device, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.dg_num = 16
        self.dk = 1
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

        self.dcn_pre_lv0 = nn.Conv2d((mid_channels//1)*2+2, mid_channels//1, 3, 1, 1, bias=True)
        self.dcn_pre_lv1 = nn.Conv2d((mid_channels//1)*2+2, mid_channels//1, 3, 1, 1, bias=True)
        self.dcn_pre_lv2 = nn.Conv2d((mid_channels//2)*2+2, mid_channels//2, 3, 1, 1, bias=True)
        self.dcn_pre_lv3 = nn.Conv2d((mid_channels//4)*2+2, mid_channels//4, 3, 1, 1, bias=True)

        self.dcn_block_lv0 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.dcn_offset_lv0 = nn.Conv2d(mid_channels//1, (self.dg_num)*2*self.dk*self.dk, 3, 1, 1, bias=True)
        self.dcn_offset_lv1 = nn.Conv2d(mid_channels//1, 2, 3, 1, 1, bias=True)
        self.dcn_offset_lv2 = nn.Conv2d(mid_channels//2, 2, 3, 1, 1, bias=True)
        self.dcn_offset_lv3 = nn.Conv2d(mid_channels//4, 2, 3, 1, 1, bias=True)

        self.dcn_mask_lv0 = nn.Conv2d(mid_channels, (self.dg_num)*1*3*3, 3, 1, 1, bias=True)
        
        self.dcn_lv0 = DCNv2(mid_channels, mid_channels, self.dk,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num)
        self.init_dcn()

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_v1(mid_channels)

        self.conv_tttf_lv1 = conv3x3((mid_channels//1)*2, mid_channels//1)
        self.conv_tttf_lv2 = conv3x3((mid_channels//2)*2, mid_channels//2)
        self.conv_tttf_lv3 = conv3x3((mid_channels//4)*2, mid_channels//4)

        # propagation branches
        self.forward_resblocks_lv0 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)
        self.forward_resblocks_lv1 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)
        self.forward_resblocks_lv2 = ResidualBlocksWithInputConv(
            (mid_channels // 2) * 2, mid_channels // 2, 1)
        self.forward_resblocks_lv3 = ResidualBlocksWithInputConv(
            (mid_channels // 4) * 2, mid_channels // 4, 1)
        
        # downsample
        self.downsample0 = PixelUnShufflePack(
            mid_channels // 4, mid_channels // 2, 2, downsample_kernel=3)
        self.downsample1 = PixelUnShufflePack(
            mid_channels // 2, mid_channels, 2, downsample_kernel=3)
        self.downsample2 = PixelUnShufflePack(
            mid_channels, mid_channels, 2, downsample_kernel=3)

        # upsample
        self.upsample0 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels // 2, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels // 2, mid_channels // 4, 2, upsample_kernel=3)

        self.conv_hr_lv3 = nn.Conv2d(mid_channels // 4, mid_channels // 4, 3, 1, 1)
        self.conv_last_lv3 = nn.Conv2d(mid_channels // 4, 3, 3, 1, 1)

        ### 4x settings
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_downsample_2x = nn.Upsample(
            scale_factor=0.5, mode='bilinear', align_corners=False)
        ### 8x settings
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)
        self.img_downsample_8x = nn.Upsample(
            scale_factor=0.125, mode='bilinear', align_corners=False)

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
    # @profile
    def forward(self, lrs, fvs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        # print(lrs.size(), mks.size())
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        a = time.time()
        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs)
        end.record()
        torch.cuda.synchronize()
        print(time.time() - a, 'flow')
        print(start.elapsed_time(end) / 1000, 'torch')
        
        ### forward-time propagation and upsampling
        outputs = []

        feat_prop_lv0 = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_lv1 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv2 = lrs.new_zeros(n, self.mid_channels // 2, h*4, w*4)
        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels // 4, h*8, w*8)

        ### texture transfer
        # B, N, C, H, W = mks.size()
        # mk_lv3 = mks.float()
        # mk_lv2 = self.img_downsample_2x(mk_lv3.view(B*N, 1, H, W)).view(B, N, 1, H//2, W//2)
        # mk_lv1 = self.img_downsample_2x(mk_lv2.view(B*N, 1, H//2, W//2)).view(B, N, 1, H//4, W//4)
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        a = time.time()
        torch.cuda.synchronize()
        start.record()
        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        # fvs = fvs * mks.float() + lrs_lv3.view(B, N, C, H*8, W*8) * (1 - mks.float())
        B, N, C, H, W = fvs.size()
        x_hr_lv1, x_hr_lv2, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3[:, :, :H, :W]), dim=1), islr=True)
        end.record()
        torch.cuda.synchronize()
        print(time.time() - a, 'encode')
        print(start.elapsed_time(end) / 1000, 'torch')

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv1.size()
        x_hr_lv1 = x_hr_lv1.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv2.size()
        x_hr_lv2 = x_hr_lv2.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)

        res_lv0_list = []
        res_lv1_list = []
        res_lv2_list = []
        res_lv3_list = []        
        
        dcn_pre_lv0_list = []
        dcn_pre_lv1_list = []
        dcn_pre_lv2_list = []
        dcn_pre_lv3_list = []

        dcn_lv0_list = []
        dcn_lv1_list = []
        dcn_lv2_list = []
        dcn_lv3_list = []
        
        flow_lv0_list = []
        flow_lv1_list = []
        flow_lv2_list = []
        flow_lv3_list = []

        last_list = []

        res_lv0_torch_list = []
        res_lv1_torch_list = []
        res_lv2_torch_list = []
        res_lv3_torch_list = []        
        
        dcn_pre_lv0_torch_list = []
        dcn_pre_lv1_torch_list = []
        dcn_pre_lv2_torch_list = []
        dcn_pre_lv3_torch_list = []

        dcn_lv0_torch_list = []
        dcn_lv1_torch_list = []
        dcn_lv2_torch_list = []
        dcn_lv3_torch_list = []
        
        flow_lv0_torch_list = []
        flow_lv1_torch_list = []
        flow_lv2_torch_list = []
        flow_lv3_torch_list = []

        last_torch_list = []

        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]

            x_hr_lv1_cur = x_hr_lv1[:, i, :, :, :]
            x_hr_lv2_cur = x_hr_lv2[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            # mk_cur_lv3 = mk_lv3[:, i, :, :, :]
            # mk_cur_lv2 = mk_lv2[:, i, :, :, :]
            # mk_cur_lv1 = mk_lv1[:, i, :, :, :]

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]


                flow_lv0 = flow
                flow_lv1 = self.img_upsample_2x(flow_lv0)
                flow_lv2 = self.img_upsample_2x(flow_lv1)
                flow_lv3 = self.img_upsample_2x(flow_lv2)

                feat_prop_lv2 = self.downsample0(feat_prop_lv3)
                feat_prop_lv1 = self.downsample1(feat_prop_lv2)
                feat_prop_lv0 = self.downsample2(feat_prop_lv1)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv0_ = flow_warp(feat_prop_lv0, flow_lv0.permute(0, 2, 3, 1))
                end.record()
                torch.cuda.synchronize()
                flow_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                flow_lv0_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                # feat_prop_lv1_ = flow_warp(feat_prop_lv1, flow_lv1.permute(0, 2, 3, 1))
                feat_prop_lv1_ = self.upsample0(feat_prop_lv0_)
                end.record()
                torch.cuda.synchronize()
                flow_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                flow_lv1_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                # feat_prop_lv2_ = flow_warp(feat_prop_lv2, flow_lv2.permute(0, 2, 3, 1))
                feat_prop_lv2_ = self.upsample1(feat_prop_lv1_)
                end.record()
                torch.cuda.synchronize()
                flow_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                flow_lv2_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                # feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                feat_prop_lv3_ = self.upsample2(feat_prop_lv2_)
                end.record()
                torch.cuda.synchronize()
                flow_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                flow_lv3_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv0_ = torch.cat([x_lr_lv0_cur, feat_prop_lv0_, flow_lv0], dim=1)
                feat_prop_lv0_ = self.dcn_pre_lv0(feat_prop_lv0_)
                feat_prop_lv0_ = self.dcn_block_lv0(feat_prop_lv0_)
                feat_offset_lv0 = self.dcn_offset_lv0(feat_prop_lv0_)
                feat_offset_lv0 = self.max_residue_magnitude * torch.tanh(feat_offset_lv0)
                # feat_offset_lv0 = feat_offset_lv0 + flow_lv0.flip(1).repeat(1, feat_offset_lv0.size(1) // 2, 1, 1)
                flow_lv0 = torch.cat((flow_lv0[:, 1:2, :, :], flow_lv0[:, 0:1, :, :]), dim=1)
                flow_lv0 = flow_lv0.repeat(1, feat_offset_lv0.size(1) // 2, 1, 1)
                feat_offset_lv0 = feat_offset_lv0 + flow_lv0
                feat_mask_lv0 = self.dcn_mask_lv0(feat_prop_lv0_)
                feat_mask_lv0 = torch.sigmoid(feat_mask_lv0)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_pre_lv0_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv0 = self.dcn_lv0(feat_prop_lv0, feat_offset_lv0, feat_mask_lv0)
                end.record()
                torch.cuda.synchronize()
                dcn_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_lv0_list.append(time.time() - a)
                
                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv0 = torch.cat([x_lr_lv0_cur, feat_prop_lv0], dim=1)
                feat_prop_lv0 = self.forward_resblocks_lv0(feat_prop_lv0)
                feat_prop_lv0 = self.lrelu(self.upsample0(feat_prop_lv0))
                end.record()
                torch.cuda.synchronize()
                res_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv0_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv1_ = torch.cat([feat_prop_lv0, feat_prop_lv1_, flow_lv1], dim=1)
                feat_prop_lv1_ = self.dcn_pre_lv1(feat_prop_lv1_)
                # feat_prop_lv1_ = self.dcn_block_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.dcn_offset_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.max_residue_magnitude * torch.tanh(feat_offset_lv1)
                # feat_offset_lv1 = feat_offset_lv1 + flow_lv1.flip(1).repeat(1, feat_offset_lv1.size(1) // 2, 1, 1)
                # flow_lv1 = torch.cat((flow_lv1[:, 1:2, :, :], flow_lv1[:, 0:1, :, :]), dim=1)
                # flow_lv1 = flow_lv1.repeat(1, feat_offset_lv1.size(1) // 2, 1, 1)
                # feat_offset_lv1 = feat_offset_lv1 + flow_lv1
                flow_lv1 = feat_offset_lv1 + flow_lv1
                feat_prop_lv1 = flow_warp(feat_prop_lv1, flow_lv1.permute(0, 2, 3, 1))
                # feat_mask_lv1 = self.dcn_mask_lv1(feat_prop_lv1_)
                # feat_mask_lv1 = torch.sigmoid(feat_mask_lv1)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_pre_lv1_list.append(time.time() - a)

                # a = time.time()
                # torch.cuda.synchronize()
                # start.record()
                # feat_prop_lv1 = self.dcn_lv1(feat_prop_lv1, feat_offset_lv1, feat_mask_lv1)
                # end.record()
                # torch.cuda.synchronize()
                # dcn_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                # dcn_lv1_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv1 = torch.cat([feat_prop_lv0, feat_prop_lv1], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                B, C, H, W = x_hr_lv1_cur.size()
                feat_prop_lv1_ = torch.cat([feat_prop_lv1[:, :, :H, :W], x_hr_lv1_cur], dim=1)
                # feat_prop_lv1_ = torch.cat([feat_prop_lv1, x_hr_lv1_cur], dim=1)
                feat_prop_lv1_ = self.conv_tttf_lv1(feat_prop_lv1_)
                # feat_prop_lv1 = mk_cur_lv1 * feat_prop_lv1_ + (1 - mk_cur_lv1) * feat_prop_lv1
                feat_prop_lv1[:, :, :H, :W] = feat_prop_lv1_[:, :, :H, :W]
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))
                end.record()
                torch.cuda.synchronize()
                res_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv1_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv2_ = torch.cat([feat_prop_lv1, feat_prop_lv2_, flow_lv2], dim=1)
                feat_prop_lv2_ = self.dcn_pre_lv2(feat_prop_lv2_)
                # feat_prop_lv2_ = self.dcn_block(feat_prop_lv2_)
                feat_offset_lv2 = self.dcn_offset_lv2(feat_prop_lv2_)
                feat_offset_lv2 = self.max_residue_magnitude * torch.tanh(feat_offset_lv2)
                flow_lv2 = feat_offset_lv2 + flow_lv2
                feat_prop_lv2 = flow_warp(feat_prop_lv2, flow_lv2.permute(0, 2, 3, 1))
                # flow_lv2 = torch.cat((flow_lv2[:, 1:2, :, :], flow_lv2[:, 0:1, :, :]), dim=1)
                # flow_lv2 = flow_lv2.repeat(1, feat_offset_lv2.size(1) // 2, 1, 1)
                # feat_offset_lv2 = feat_offset_lv2 + flow_lv2
                # feat_mask_lv2 = self.dcn_mask_lv2(feat_prop_lv2_)
                # feat_mask_lv2 = torch.sigmoid(feat_mask_lv2)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_pre_lv2_list.append(time.time() - a)

                # a = time.time()
                # torch.cuda.synchronize()
                # start.record()
                # feat_prop_lv2 = self.dcn_lv2(feat_prop_lv2, feat_offset_lv2, feat_mask_lv2)
                # end.record()
                # torch.cuda.synchronize()
                # dcn_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                # dcn_lv2_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                B, C, H, W = x_hr_lv2_cur.size()
                feat_prop_lv2_ = torch.cat([feat_prop_lv2[:, :, :H, :W], x_hr_lv2_cur], dim=1)
                # feat_prop_lv2_ = torch.cat([feat_prop_lv2, x_hr_lv2_cur], dim=1)
                feat_prop_lv2_ = self.conv_tttf_lv2(feat_prop_lv2_)
                # feat_prop_lv2 = mk_cur_lv2 * feat_prop_lv2_ + (1 - mk_cur_lv2) * feat_prop_lv2
                feat_prop_lv2[:, :, :H, :W] = feat_prop_lv2_[:, :, :H, :W]
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))
                end.record()
                torch.cuda.synchronize()
                res_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv2_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3_ = torch.cat([feat_prop_lv2, feat_prop_lv3_, flow_lv3], dim=1)
                feat_prop_lv3_ = self.dcn_pre_lv3(feat_prop_lv3_)
                # feat_prop_lv3_ = self.dcn_block(feat_prop_lv3_)
                feat_offset_lv3 = self.dcn_offset_lv3(feat_prop_lv3_)
                feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                feat_offset_lv3 = feat_offset_lv3 + flow_lv3
                feat_prop_lv3 = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                # flow_lv3 = torch.cat((flow_lv3[:, 1:2, :, :], flow_lv3[:, 0:1, :, :]), dim=1)
                # flow_lv3 = flow_lv3.repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                # feat_offset_lv3 = feat_offset_lv3 + flow_lv3
                # feat_mask_lv3 = self.dcn_mask_lv3(feat_prop_lv3_)
                # feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                dcn_pre_lv3_list.append(time.time() - a)

                # a = time.time()
                # torch.cuda.synchronize()
                # start.record()
                # feat_prop_lv3 = self.dcn_lv3(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)
                # end.record()
                # torch.cuda.synchronize()
                # dcn_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                # dcn_lv3_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                B, C, H, W = x_hr_lv3_cur.size()
                feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
                # feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                # feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3
                feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
                end.record()
                torch.cuda.synchronize()
                res_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv3_list.append(time.time() - a)
            
            else:
                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv0 = torch.cat([x_lr_lv0_cur, feat_prop_lv0], dim=1)
                feat_prop_lv0 = self.forward_resblocks_lv0(feat_prop_lv0)
                feat_prop_lv0 = self.lrelu(self.upsample0(feat_prop_lv0))
                end.record()
                torch.cuda.synchronize()
                res_lv0_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv0_list.append(time.time() - a)

                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv1 = torch.cat([feat_prop_lv0, feat_prop_lv1], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                B, C, H, W = x_hr_lv1_cur.size()
                feat_prop_lv1_ = torch.cat([feat_prop_lv1[:, :, :H, :W], x_hr_lv1_cur], dim=1)
                # feat_prop_lv1_ = torch.cat([feat_prop_lv1, x_hr_lv1_cur], dim=1)
                feat_prop_lv1_ = self.conv_tttf_lv1(feat_prop_lv1_)
                # feat_prop_lv1 = mk_cur_lv1 * feat_prop_lv1_ + (1 - mk_cur_lv1) * feat_prop_lv1
                feat_prop_lv1[:, :, :H, :W] = feat_prop_lv1_[:, :, :H, :W]
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))
                end.record()
                torch.cuda.synchronize()
                res_lv1_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv1_list.append(time.time() - a)
            
                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                B, C, H, W = x_hr_lv2_cur.size()
                feat_prop_lv2_ = torch.cat([feat_prop_lv2[:, :, :H, :W], x_hr_lv2_cur], dim=1)
                # feat_prop_lv2_ = torch.cat([feat_prop_lv2, x_hr_lv2_cur], dim=1)
                feat_prop_lv2_ = self.conv_tttf_lv2(feat_prop_lv2_)
                # feat_prop_lv2 = mk_cur_lv2 * feat_prop_lv2_ + (1 - mk_cur_lv2) * feat_prop_lv2
                feat_prop_lv2[:, :, :H, :W] = feat_prop_lv2_[:, :, :H, :W]
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))
                end.record()
                torch.cuda.synchronize()
                res_lv2_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv2_list.append(time.time() - a)
                
                a = time.time()
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                B, C, H, W = x_hr_lv3_cur.size()
                feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
                # feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                # feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3
                feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
                end.record()
                torch.cuda.synchronize()
                res_lv3_torch_list.append(start.elapsed_time(end) / 1000)
                res_lv3_list.append(time.time() - a)

            a = time.time()
            torch.cuda.synchronize()
            start.record()
            out_lv3 = feat_prop_lv3
            out_lv3 = self.lrelu(self.conv_hr_lv3(out_lv3))
            out_lv3 = self.conv_last_lv3(out_lv3)
            base_lv3 = self.img_upsample_8x(lr_cur)
            out_lv3 += base_lv3
            outputs.append(out_lv3)
            end.record()
            torch.cuda.synchronize()
            last_torch_list.append(start.elapsed_time(end) / 1000)
            last_list.append(time.time() - a)

        a = 0
        print(sum(flow_lv0_list)/len(flow_lv0_list), 'flow_lv0')
        print(sum(flow_lv1_list)/len(flow_lv1_list), 'flow_lv1')
        print(sum(flow_lv2_list)/len(flow_lv2_list), 'flow_lv2')
        print(sum(flow_lv3_list)/len(flow_lv3_list), 'flow_lv3')
        print(sum(dcn_pre_lv0_list)/len(dcn_pre_lv0_list), 'dcn_pre_lv0')
        print(sum(dcn_pre_lv1_list)/len(dcn_pre_lv1_list), 'dcn_pre_lv1')
        print(sum(dcn_pre_lv2_list)/len(dcn_pre_lv2_list), 'dcn_pre_lv2')
        print(sum(dcn_pre_lv3_list)/len(dcn_pre_lv3_list), 'dcn_pre_lv3')
        print(sum(dcn_lv0_list)/len(dcn_lv0_list), 'dcn_lv0')
        # print(sum(dcn_lv1_list)/len(dcn_lv1_list), 'dcn_lv1')
        # print(sum(dcn_lv2_list)/len(dcn_lv2_list), 'dcn_lv2')
        # print(sum(dcn_lv3_list)/len(dcn_lv3_list), 'dcn_lv3')
        print(sum(res_lv0_list)/len(res_lv0_list), 'res_lv0')
        print(sum(res_lv1_list)/len(res_lv1_list), 'res_lv1')
        print(sum(res_lv2_list)/len(res_lv2_list), 'res_lv2')
        print(sum(res_lv3_list)/len(res_lv3_list), 'res_lv3')
        print(sum(last_list)/len(last_list), 'last')
        
        a+=sum(flow_lv0_list)/len(flow_lv0_list)
        a+=sum(flow_lv1_list)/len(flow_lv1_list)
        a+=sum(flow_lv2_list)/len(flow_lv2_list)
        a+=sum(flow_lv3_list)/len(flow_lv3_list)
        a+=sum(dcn_pre_lv0_list)/len(dcn_pre_lv0_list)
        a+=sum(dcn_pre_lv1_list)/len(dcn_pre_lv1_list)
        a+=sum(dcn_pre_lv2_list)/len(dcn_pre_lv2_list)
        a+=sum(dcn_pre_lv3_list)/len(dcn_pre_lv3_list)
        a+=sum(dcn_lv0_list)/len(dcn_lv0_list)
        # a+=sum(dcn_lv1_list)/len(dcn_lv1_list)
        # a+=sum(dcn_lv2_list)/len(dcn_lv2_list)
        # a+=sum(dcn_lv3_list)/len(dcn_lv3_list)
        a+=sum(res_lv0_list)/len(res_lv0_list)
        a+=sum(res_lv1_list)/len(res_lv1_list)
        a+=sum(res_lv2_list)/len(res_lv2_list)
        a+=sum(res_lv3_list)/len(res_lv3_list)
        a+=sum(last_list)/len(last_list)
        
        print(a, 'total')

        a = 0
        print(sum(flow_lv0_torch_list)/len(flow_lv0_torch_list), 'flow_lv0_torch')
        print(sum(flow_lv1_torch_list)/len(flow_lv1_torch_list), 'flow_lv1_torch')
        print(sum(flow_lv2_torch_list)/len(flow_lv2_torch_list), 'flow_lv2_torch')
        print(sum(flow_lv3_torch_list)/len(flow_lv3_torch_list), 'flow_lv3_torch')
        print(sum(dcn_pre_lv0_torch_list)/len(dcn_pre_lv0_torch_list), 'dcn_pre_lv0_torch')
        print(sum(dcn_pre_lv1_torch_list)/len(dcn_pre_lv1_torch_list), 'dcn_pre_lv1_torch')
        print(sum(dcn_pre_lv2_torch_list)/len(dcn_pre_lv2_torch_list), 'dcn_pre_lv2_torch')
        print(sum(dcn_pre_lv3_torch_list)/len(dcn_pre_lv3_torch_list), 'dcn_pre_lv3_torch')
        print(sum(dcn_lv0_torch_list)/len(dcn_lv0_torch_list), 'dcn_lv0_torch')
        # print(sum(dcn_lv1_torch_list)/len(dcn_lv1_torch_list), 'dcn_lv1_torch')
        # print(sum(dcn_lv2_torch_list)/len(dcn_lv2_torch_list), 'dcn_lv2_torch')
        # print(sum(dcn_lv3_torch_list)/len(dcn_lv3_torch_list), 'dcn_lv3_torch')
        print(sum(res_lv0_torch_list)/len(res_lv0_torch_list), 'res_lv0_torch')
        print(sum(res_lv1_torch_list)/len(res_lv1_torch_list), 'res_lv1_torch')
        print(sum(res_lv2_torch_list)/len(res_lv2_torch_list), 'res_lv2_torch')
        print(sum(res_lv3_torch_list)/len(res_lv3_torch_list), 'res_lv3_torch')
        print(sum(last_torch_list)/len(last_torch_list), 'last_torch')
        
        a+=sum(flow_lv0_torch_list)/len(flow_lv0_torch_list)
        a+=sum(flow_lv1_torch_list)/len(flow_lv1_torch_list)
        a+=sum(flow_lv2_torch_list)/len(flow_lv2_torch_list)
        a+=sum(flow_lv3_torch_list)/len(flow_lv3_torch_list)
        a+=sum(dcn_pre_lv0_torch_list)/len(dcn_pre_lv0_torch_list)
        a+=sum(dcn_pre_lv1_torch_list)/len(dcn_pre_lv1_torch_list)
        a+=sum(dcn_pre_lv2_torch_list)/len(dcn_pre_lv2_torch_list)
        a+=sum(dcn_pre_lv3_torch_list)/len(dcn_pre_lv3_torch_list)
        a+=sum(dcn_lv0_torch_list)/len(dcn_lv0_torch_list)
        # a+=sum(dcn_lv1_torch_list)/len(dcn_lv1_torch_list)
        # a+=sum(dcn_lv2_torch_list)/len(dcn_lv2_torch_list)
        # a+=sum(dcn_lv3_torch_list)/len(dcn_lv3_torch_list)
        a+=sum(res_lv0_torch_list)/len(res_lv0_torch_list)
        a+=sum(res_lv1_torch_list)/len(res_lv1_torch_list)
        a+=sum(res_lv2_torch_list)/len(res_lv2_torch_list)
        a+=sum(res_lv3_torch_list)/len(res_lv3_torch_list)
        a+=sum(last_torch_list)/len(last_torch_list)
        
        print(a, 'total_torch')

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
    def init_dcn(self):
        
        self.dcn_offset_lv0.weight.data.zero_()
        self.dcn_offset_lv0.bias.data.zero_()
        self.dcn_mask_lv0.weight.data.zero_()
        self.dcn_mask_lv0.bias.data.zero_()
        self.conv_identify(self.dcn_lv0.weight, self.dcn_lv0.bias)

        self.dcn_offset_lv1.weight.data.zero_()
        self.dcn_offset_lv1.bias.data.zero_()

        self.dcn_offset_lv2.weight.data.zero_()
        self.dcn_offset_lv2.bias.data.zero_()

        self.dcn_offset_lv3.weight.data.zero_()
        self.dcn_offset_lv3.bias.data.zero_()

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

class MRCF_CRA_x8_v1(nn.Module):

    def __init__(self, device, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.dg_num = 16
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

        self.dcn_pre_lv1 = nn.Conv2d((mid_channels//1)*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv2 = nn.Conv2d((mid_channels//2)*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_pre_lv3 = nn.Conv2d((mid_channels//4)*2+2, mid_channels, 3, 1, 1, bias=True)

        self.dcn_block_lv1 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.dcn_offset_lv1 = nn.Conv2d(mid_channels, (self.dg_num)*2*3*3, 3, 1, 1, bias=True)
        self.dcn_offset_lv2 = nn.Conv2d(mid_channels, 2, 3, 1, 1, bias=True)
        self.dcn_offset_lv3 = nn.Conv2d(mid_channels, 2, 3, 1, 1, bias=True)

        self.dcn_mask_lv1 = nn.Conv2d(mid_channels, (self.dg_num)*1*3*3, 3, 1, 1, bias=True)
        
        self.dcn_lv1 = DCNv2(mid_channels, mid_channels, 3,
                        stride=1, padding=1, dilation=1,
                        deformable_groups=self.dg_num)
        self.init_dcn()

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_v1(mid_channels)

        self.conv_tttf_lv1 = conv3x3((mid_channels//1)*2, mid_channels//1)
        self.conv_tttf_lv2 = conv3x3((mid_channels//2)*2, mid_channels//2)
        self.conv_tttf_lv3 = conv3x3((mid_channels//4)*2, mid_channels//4)

        # propagation branches
        self.forward_resblocks_lv0 = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        self.forward_resblocks_lv1 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)
        self.forward_resblocks_lv2 = ResidualBlocksWithInputConv(
            (mid_channels // 2) * 2, mid_channels // 2, 1)
        self.forward_resblocks_lv3 = ResidualBlocksWithInputConv(
            (mid_channels // 4) * 2, mid_channels // 4, 1)
        
        # downsample
        self.downsample0 = PixelUnShufflePack(
            mid_channels // 4, mid_channels // 2, 2, downsample_kernel=3)
        self.downsample1 = PixelUnShufflePack(
            mid_channels // 2, mid_channels, 2, downsample_kernel=3)
        # self.downsample2 = PixelUnShufflePack(
            # mid_channels // 4, mid_channels, 2, downsample_kernel=3)

        # upsample
        self.upsample0 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels // 2, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels // 2, mid_channels // 4, 2, upsample_kernel=3)

        self.conv_hr_lv3 = nn.Conv2d(mid_channels // 4, mid_channels // 4, 3, 1, 1)
        self.conv_last_lv3 = nn.Conv2d(mid_channels // 4, 3, 3, 1, 1)

        ### 4x settings
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_downsample_2x = nn.Upsample(
            scale_factor=0.5, mode='bilinear', align_corners=False)
        ### 8x settings
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)
        self.img_downsample_8x = nn.Upsample(
            scale_factor=0.125, mode='bilinear', align_corners=False)

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
    # @profile
    def forward(self, lrs, fvs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        # print(lrs.size(), mks.size())
        n, t, c, h, w = lrs.size()

        rg_h = 1080
        rg_w = 1920
        ### compute optical flow
        torch.cuda.synchronize()
        start.record()
        H, W = rg_h//8, rg_w//8
        flows_forward, flows_backward = self.compute_flow(lrs[:, :, :, :H, :W])
        end.record()
        torch.cuda.synchronize()
        print((start.elapsed_time(end) / 1000 / t), 'flow_torch')
        ### forward-time propagation and upsampling
        outputs = []

        feat_prop_lv0 = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_lv1 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv2 = lrs.new_zeros(n, self.mid_channels // 2, h*4, w*4)
        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels // 4, h*8, w*8)

        ### texture transfer
        # B, N, C, H, W = mks.size()
        # mk_lv3 = mks.float()
        # mk_lv2 = self.img_downsample_2x(mk_lv3.view(B*N, 1, H, W)).view(B, N, 1, H//2, W//2)
        # mk_lv1 = self.img_downsample_2x(mk_lv2.view(B*N, 1, H//2, W//2)).view(B, N, 1, H//4, W//4)
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        # fvs = fvs * mks.float() + lrs_lv3.view(B, N, C, H*8, W*8) * (1 - mks.float())
        B, N, C, H, W = fvs.size()
        x_hr_lv1, x_hr_lv2, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3[:, :, :H, :W]), dim=1), islr=True)
        # x_hr_lv1, x_hr_lv2, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv1.size()
        x_hr_lv1 = x_hr_lv1.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv2.size()
        x_hr_lv2 = x_hr_lv2.contiguous().view(B, N, C, H, W)
        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)

        flow_list = []

        torch.cuda.synchronize()
        start.record()

        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]

            x_hr_lv1_cur = x_hr_lv1[:, i, :, :, :]
            x_hr_lv2_cur = x_hr_lv2[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            # mk_cur_lv3 = mk_lv3[:, i, :, :, :]
            # mk_cur_lv2 = mk_lv2[:, i, :, :, :]
            # mk_cur_lv1 = mk_lv1[:, i, :, :, :]

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]

                flow_lv1 = self.img_upsample_2x(flow)
                flow_lv2 = self.img_upsample_2x(flow_lv1)
                flow_lv3 = self.img_upsample_2x(flow_lv2)

                H, W = rg_h, rg_w
                feat_prop_lv3 = feat_prop_lv3[:, :, :H, :W]
                feat_prop_lv2 = self.downsample0(feat_prop_lv3)
                feat_prop_lv1 = self.downsample1(feat_prop_lv2)

                # a = time.time()
                # torch.cuda.synchronize()
                # start.record()
                feat_prop_lv1_ = flow_warp(feat_prop_lv1, flow_lv1.permute(0, 2, 3, 1))
                feat_prop_lv2_ = flow_warp(feat_prop_lv2, flow_lv2.permute(0, 2, 3, 1))
                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                # end.record()
                # torch.cuda.synchronize()
                # flow_list.append(start.elapsed_time(end) / 1000)

                x_lr_lv0_cur = self.forward_resblocks_lv0(x_lr_lv0_cur)
                x_lr_lv0_cur = self.lrelu(self.upsample0(x_lr_lv0_cur))

                H, W = rg_h//4, rg_w//4
                feat_prop_lv1_ = torch.cat([x_lr_lv0_cur[:, :, :H, :W], feat_prop_lv1_, flow_lv1], dim=1)
                # feat_prop_lv1_ = torch.cat([x_lr_lv0_cur, feat_prop_lv1_, flow_lv1], dim=1)
                feat_prop_lv1_ = self.dcn_pre_lv1(feat_prop_lv1_)
                feat_prop_lv1_ = self.dcn_block_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.dcn_offset_lv1(feat_prop_lv1_)
                feat_offset_lv1 = self.max_residue_magnitude * torch.tanh(feat_offset_lv1)
                flow_lv1 = torch.cat((flow_lv1[:, 1:2, :, :], flow_lv1[:, 0:1, :, :]), dim=1)
                flow_lv1 = flow_lv1.repeat(1, feat_offset_lv1.size(1) // 2, 1, 1)
                feat_offset_lv1 = feat_offset_lv1 + flow_lv1
                feat_mask_lv1 = self.dcn_mask_lv1(feat_prop_lv1_)
                feat_mask_lv1 = torch.sigmoid(feat_mask_lv1)

                feat_prop_lv1 = self.dcn_lv1(feat_prop_lv1, feat_offset_lv1, feat_mask_lv1)

                feat_prop_lv1 = torch.cat([x_lr_lv0_cur, x_lr_lv0_cur], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                # feat_prop_lv1 = x_lr_lv0_cur
                # feat_prop_lv1[:, :, :H, :W] = feat_prop_lv1_[:, :, :H, :W]
                # feat_prop_lv1 = torch.cat([x_lr_lv0_cur, feat_prop_lv1], dim=1)
                # feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)

                B, C, H, W = x_hr_lv1_cur.size()
                feat_prop_lv1_ = torch.cat([feat_prop_lv1[:, :, :H, :W], x_hr_lv1_cur], dim=1)
                # feat_prop_lv1_ = torch.cat([feat_prop_lv1, x_hr_lv1_cur], dim=1)
                feat_prop_lv1_ = self.conv_tttf_lv1(feat_prop_lv1_)
                # feat_prop_lv1 = mk_cur_lv1 * feat_prop_lv1_ + (1 - mk_cur_lv1) * feat_prop_lv1
                feat_prop_lv1[:, :, :H, :W] = feat_prop_lv1_[:, :, :H, :W]
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))

                H, W = rg_h//2, rg_w//2
                feat_prop_lv2_ = torch.cat([feat_prop_lv1[:, :, :H, :W], feat_prop_lv2_, flow_lv2], dim=1)
                # feat_prop_lv2_ = torch.cat([feat_prop_lv1, feat_prop_lv2_, flow_lv2], dim=1)
                feat_prop_lv2_ = self.dcn_pre_lv2(feat_prop_lv2_)
                feat_offset_lv2 = self.dcn_offset_lv2(feat_prop_lv2_)
                feat_offset_lv2 = self.max_residue_magnitude * torch.tanh(feat_offset_lv2)
                flow_lv2 = feat_offset_lv2 + flow_lv2
                feat_prop_lv2 = flow_warp(feat_prop_lv2, flow_lv2.permute(0, 2, 3, 1))

                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv1], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                # feat_prop_lv2 = feat_prop_lv1
                # feat_prop_lv2[:, :, :H, :W] = feat_prop_lv2_[:, :, :H, :W]
                # feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                # feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)

                B, C, H, W = x_hr_lv2_cur.size()
                feat_prop_lv2_ = torch.cat([feat_prop_lv2[:, :, :H, :W], x_hr_lv2_cur], dim=1)
                # feat_prop_lv2_ = torch.cat([feat_prop_lv2, x_hr_lv2_cur], dim=1)
                feat_prop_lv2_ = self.conv_tttf_lv2(feat_prop_lv2_)
                # feat_prop_lv2 = mk_cur_lv2 * feat_prop_lv2_ + (1 - mk_cur_lv2) * feat_prop_lv2
                feat_prop_lv2[:, :, :H, :W] = feat_prop_lv2_[:, :, :H, :W]
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))

                H, W = rg_h, rg_w
                feat_prop_lv3_ = torch.cat([feat_prop_lv2[:, :, :H, :W], feat_prop_lv3_, flow_lv3], dim=1)
                # feat_prop_lv3_ = torch.cat([feat_prop_lv2, feat_prop_lv3_, flow_lv3], dim=1)
                feat_prop_lv3_ = self.dcn_pre_lv3(feat_prop_lv3_)
                feat_offset_lv3 = self.dcn_offset_lv3(feat_prop_lv3_)
                feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                feat_offset_lv3 = feat_offset_lv3 + flow_lv3
                feat_prop_lv3 = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))

                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv2], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                # feat_prop_lv3 = feat_prop_lv2
                # feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
                # feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                # feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)

                B, C, H, W = x_hr_lv3_cur.size()
                feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
                # feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                # feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3
                feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
            
            else:
                x_lr_lv0_cur = self.forward_resblocks_lv0(x_lr_lv0_cur)
                x_lr_lv0_cur = self.lrelu(self.upsample0(x_lr_lv0_cur))

                feat_prop_lv1 = torch.cat([x_lr_lv0_cur, feat_prop_lv1], dim=1)
                feat_prop_lv1 = self.forward_resblocks_lv1(feat_prop_lv1)
                B, C, H, W = x_hr_lv1_cur.size()
                feat_prop_lv1_ = torch.cat([feat_prop_lv1[:, :, :H, :W], x_hr_lv1_cur], dim=1)
                # feat_prop_lv1_ = torch.cat([feat_prop_lv1, x_hr_lv1_cur], dim=1)
                feat_prop_lv1_ = self.conv_tttf_lv1(feat_prop_lv1_)
                # feat_prop_lv1 = mk_cur_lv1 * feat_prop_lv1_ + (1 - mk_cur_lv1) * feat_prop_lv1
                feat_prop_lv1[:, :, :H, :W] = feat_prop_lv1_[:, :, :H, :W]
                feat_prop_lv1 = self.lrelu(self.upsample1(feat_prop_lv1))
            
                feat_prop_lv2 = torch.cat([feat_prop_lv1, feat_prop_lv2], dim=1)
                feat_prop_lv2 = self.forward_resblocks_lv2(feat_prop_lv2)
                B, C, H, W = x_hr_lv2_cur.size()
                feat_prop_lv2_ = torch.cat([feat_prop_lv2[:, :, :H, :W], x_hr_lv2_cur], dim=1)
                # feat_prop_lv2_ = torch.cat([feat_prop_lv2, x_hr_lv2_cur], dim=1)
                feat_prop_lv2_ = self.conv_tttf_lv2(feat_prop_lv2_)
                # feat_prop_lv2 = mk_cur_lv2 * feat_prop_lv2_ + (1 - mk_cur_lv2) * feat_prop_lv2
                feat_prop_lv2[:, :, :H, :W] = feat_prop_lv2_[:, :, :H, :W]
                feat_prop_lv2 = self.lrelu(self.upsample2(feat_prop_lv2))
                
                feat_prop_lv3 = torch.cat([feat_prop_lv2, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_lv3(feat_prop_lv3)
                B, C, H, W = x_hr_lv3_cur.size()
                feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
                # feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
                feat_prop_lv3_ = self.conv_tttf_lv3(feat_prop_lv3_)
                # feat_prop_lv3 = mk_cur_lv3 * feat_prop_lv3_ + (1 - mk_cur_lv3) * feat_prop_lv3
                feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]

            out_lv3 = feat_prop_lv3
            out_lv3 = self.lrelu(self.conv_hr_lv3(out_lv3))
            out_lv3 = self.conv_last_lv3(out_lv3)
            base_lv3 = self.img_upsample_8x(lr_cur)
            out_lv3 += base_lv3
            outputs.append(out_lv3)

        end.record()
        torch.cuda.synchronize()
        flow_list.append(start.elapsed_time(end) / 1000 / t)

        print(sum(flow_list)/len(flow_list), 'other_torch')

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
    def init_dcn(self):
        
        self.dcn_offset_lv1.weight.data.zero_()
        self.dcn_offset_lv1.bias.data.zero_()
        self.dcn_mask_lv1.weight.data.zero_()
        self.dcn_mask_lv1.bias.data.zero_()
        self.conv_identify(self.dcn_lv1.weight, self.dcn_lv1.bias)

        self.dcn_offset_lv2.weight.data.zero_()
        self.dcn_offset_lv2.bias.data.zero_()

        self.dcn_offset_lv3.weight.data.zero_()
        self.dcn_offset_lv3.bias.data.zero_()

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

class MRCF_simple_v4(nn.Module):

    def __init__(self, device, mid_channels=16, y_only=False, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels//8
        self.dg_num = 1
        self.dk = 3
        self.max_residue_magnitude = 80

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

        self.dcn_pre = nn.Conv2d(self.last_channels*2+2, self.last_channels, 3, 1, 1, bias=True)
        self.dcn_block = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.last_channels, self.last_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_offset = nn.Conv2d(self.last_channels, (self.dg_num)*2, 3, 1, 1, bias=True)
        self.dcn_mask = nn.Conv2d(self.last_channels, (self.dg_num)*1, 3, 1, 1, bias=True)
        # self.dcn_offset = nn.Conv2d(self.last_channels, (self.dg_num)*2*self.dk*self.dk, 3, 1, 1, bias=True)
        # self.dcn_mask = nn.Conv2d(self.last_channels, (self.dg_num)*1*self.dk*self.dk, 3, 1, 1, bias=True)
        self.dcn = DCNv2(self.last_channels, self.last_channels, self.dk,
                         stride=1, padding=(self.dk-1)//2, dilation=1,
                         deformable_groups=self.dg_num)
        self.init_dcn()

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_single(self.last_channels)

        self.conv_tttf = conv3x3((self.last_channels) * 2, self.last_channels)

        # propagation branches
        self.forward_resblocks = ResidualBlocksWithInputConv(
            self.last_channels * 2, self.last_channels, 1)
        
        self.upsample = PixelShufflePack(
            mid_channels, self.last_channels, 8, upsample_kernel=3)
        
        self.conv_hr = nn.Conv2d(self.last_channels, self.last_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(self.last_channels, 3, 3, 1, 1)

        ### 8x settings
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_upsample_4x = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # @profile
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

    def forward(self, lrs, fvs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        # print(lrs.size(), mks.size())
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end) / 1000, 'flow')
        
        ### forward-time propagation and upsampling
        outputs = []

        feat_prop_lv3 = lrs.new_zeros(n, self.last_channels, h*8, w*8)

        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        lrs_lv3_view = lrs_lv3.view(B, N, C, H*8, W*8)

        B, N, C, H, W = fvs.size()
        _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3[:, :, :H, :W]), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        
        res_torch_list = []        
        dcn_pre_torch_list = []
        dcn_torch_list = []
        flow_torch_list = []

        last_torch_list = []
        
        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]

            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]

            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_8x(flow) * 8.
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                end.record()
                torch.cuda.synchronize()
                flow_torch_list.append(start.elapsed_time(end) / 1000)
                
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3_ = torch.cat([feat_prop_lv0, feat_prop_lv3_, flow_lv3], dim=1)
                feat_prop_lv3_ = self.dcn_pre(feat_prop_lv3_)
                feat_prop_lv3_ = self.dcn_block(feat_prop_lv3_)
                feat_offset_lv3 = self.dcn_offset(feat_prop_lv3_)
                feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                # feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                B, C, H, W = feat_offset_lv3.size()
                feat_offset_lv3 = feat_offset_lv3.view(B, 2, C//2, H, W)
                feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).unsqueeze(2).repeat(1, 1, C//2, 1, 1)
                feat_offset_lv3 = feat_offset_lv3.repeat(1, 9, 1, 1, 1).view(B, C*9, H, W)
                feat_mask_lv3 = self.dcn_mask(feat_prop_lv3_)
                feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
                feat_mask_lv3 = feat_mask_lv3.repeat(1, 9, 1, 1)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_torch_list.append(start.elapsed_time(end) / 1000)
                
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = self.dcn(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)
                end.record()
                torch.cuda.synchronize()
                dcn_torch_list.append(start.elapsed_time(end) / 1000)

                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks(feat_prop_lv3)
                end.record()
                torch.cuda.synchronize()
                res_torch_list.append(start.elapsed_time(end) / 1000)

                # feat_prop_lv3_ = torch.cat([feat_prop_lv0, feat_prop_lv3, flow_lv3], dim=1)
                # feat_prop_lv3_ = self.dcn_pre(feat_prop_lv3_)
                # feat_prop_lv3_ = self.dcn_block(feat_prop_lv3_)
                # feat_offset_lv3 = self.dcn_offset(feat_prop_lv3_)
                # feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                # # feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                # B, C, H, W = feat_offset_lv3.size()
                # feat_offset_lv3 = feat_offset_lv3.view(B, 2, C//2, H, W)
                # feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).unsqueeze(2).repeat(1, 1, C//2, 1, 1)
                # feat_offset_lv3 = feat_offset_lv3.repeat(1, 9, 1, 1, 1).view(B, C*9, H, W)
                # feat_mask_lv3 = self.dcn_mask(feat_prop_lv3_)
                # feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
                # feat_mask_lv3 = feat_mask_lv3.repeat(1, 9, 1, 1)
                # feat_prop_lv3 = self.dcn(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)

                # feat_prop_lv3 = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                # feat_prop_lv3 = self.forward_resblocks(feat_prop_lv3)
            else:
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks(feat_prop_lv3)
                end.record()
                torch.cuda.synchronize()
                res_torch_list.append(start.elapsed_time(end) / 1000)
                
                # feat_prop_lv3 = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                # feat_prop_lv3 = self.forward_resblocks(feat_prop_lv3)

            torch.cuda.synchronize()
            start.record()
            B, C, H, W = x_hr_lv3_cur.size()
            feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)

            out = feat_prop_lv3
            out = self.conv_last(out)
            base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)
            end.record()
            torch.cuda.synchronize()
            last_torch_list.append(start.elapsed_time(end) / 1000)
        
        a = 0
        print(sum(flow_torch_list)/len(flow_torch_list), 'flow_torch')
        print(sum(dcn_pre_torch_list)/len(dcn_pre_torch_list), 'dcn_pre_torch')
        print(sum(dcn_torch_list)/len(dcn_torch_list), 'dcn_torch')
        print(sum(res_torch_list)/len(res_torch_list), 'res_torch')
        print(sum(last_torch_list)/len(last_torch_list), 'last_torch')
        
        a+=sum(flow_torch_list)/len(flow_torch_list)
        a+=sum(dcn_pre_torch_list)/len(dcn_pre_torch_list)
        a+=sum(dcn_torch_list)/len(dcn_torch_list)
        a+=sum(res_torch_list)/len(res_torch_list)
        a+=sum(last_torch_list)/len(last_torch_list)
        
        print(a, 'total_torch')

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

class MRCF_simple(nn.Module):

    def __init__(self, device, mid_channels=16, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels//8
        self.dg_num = 8
        self.dk = 3
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)
        # self.spynet = FNet(in_nc=3)

        self.dcn_pre = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_block = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # self.dcn_offset = nn.Conv2d(mid_channels, (self.dg_num)*2*self.dk*self.dk, 3, 1, 1, bias=True)
        self.dcn_offset = nn.Conv2d(mid_channels, (self.dg_num)*2, 3, 1, 1, bias=True)
        # self.dcn_mask = nn.Conv2d(mid_channels, (self.dg_num)*1*self.dk*self.dk, 3, 1, 1, bias=True)
        self.dcn_mask = nn.Conv2d(mid_channels, (self.dg_num)*1, 3, 1, 1, bias=True)
        self.dcn = DCNv2(mid_channels, mid_channels, self.dk,
                         stride=1, padding=(self.dk-1)//2, dilation=1,
                         deformable_groups=self.dg_num)
        self.init_dcn()


        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_lr(self.last_channels)

        self.conv_tttf = conv3x3((self.last_channels) * 2, self.last_channels)

        # propagation branches
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)
        
        # downsample
        # self.downsample = PixelUnShufflePack(
            # mid_channels // 4, mid_channels, 2, downsample_kernel=3)
        # self.downsample = PixelUnShufflePack(
            # mid_channels, mid_channels, 4, downsample_kernel=3)
        self.downsample = PixelUnShufflePack(
            self.last_channels, mid_channels, 4, downsample_kernel=3)
        # self.downsample = PixelUnShufflePack(
            # mid_channels // 4, mid_channels, 8, downsample_kernel=3)

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        # self.upsample = PixelShufflePack(
        #     mid_channels, mid_channels, 4, upsample_kernel=3)
        # self.upsample = PixelShufflePack(
            # mid_channels, mid_channels, 8, upsample_kernel=3)
        
        # self.upsample_post = PixelShufflePack(
            # mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)
        # self.upsample_post = PixelShufflePack(
            # mid_channels, mid_channels, 8, upsample_kernel=3)

        self.conv_hr = nn.Conv2d(self.last_channels, self.last_channels, 3, 1, 1)
        # self.conv_hr = nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(self.last_channels, 3, 3, 1, 1)

        ### 8x settings
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_upsample_4x = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # @profile
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

    # @profile
    # def forward(self, lrs, fvs, mks):
    def forward(self, lrs, fvs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        # print(lrs.size(), mks.size())
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        # torch.cuda.reset_max_memory_allocated(0)

        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs)
        # print(torch.cuda.memory_allocated(0), '1')
        # print(torch.cuda.max_memory_allocated(0), '1')
        # torch.cuda.reset_max_memory_allocated(0)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end) / 1000, 'flow')
        
        ### forward-time propagation and upsampling
        outputs = []

        # feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        # feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*4, w*4)
        # feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*8, w*8)
        # print(torch.cuda.memory_allocated(0), '2')
        # print(torch.cuda.max_memory_allocated(0), '2')
        # torch.cuda.reset_max_memory_allocated(0)

        ### texture transfer
        # B, N, C, H, W = mks.size()
        # mk_lv3 = mks.float()

        torch.cuda.synchronize()
        start.record()
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        # lrs_lv3 = self.img_upsample_8x(lrs_lv0)
        # print(torch.cuda.memory_allocated(0), '3')
        # print(torch.cuda.max_memory_allocated(0), '3')
        # torch.cuda.reset_max_memory_allocated(0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)
        # print(torch.cuda.memory_allocated(0), '4')
        # print(torch.cuda.max_memory_allocated(0), '4')
        # torch.cuda.reset_max_memory_allocated(0)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end) / 1000, 'en_lr')

        # lrs_lv3_view = lrs_lv3.view(B, N, C, H*8, W*8)
        # mks_float = mks.float()
        # fvs = (fvs * mks_float + lrs_lv3_view * (1 - mks_float))
        # print(torch.cuda.memory_allocated(0), '5')
        # print(torch.cuda.max_memory_allocated(0), '5')
        # torch.cuda.reset_max_memory_allocated(0)

        torch.cuda.synchronize()
        start.record()
        B, N, C, H, W = fvs.size()
        # _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)
        _, _, x_hr_lv3 = self.encoder_hr(fvs.view(B*N, C, H, W), islr=True)
        # print(torch.cuda.memory_allocated(0), '6')
        # print(torch.cuda.max_memory_allocated(0), '6')
        # torch.cuda.reset_max_memory_allocated(0)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end) / 1000, 'en_hr')

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        
        res_torch_list = []        
        dcn_pre_torch_list = []
        dcn_torch_list = []
        flow_torch_list = []

        last_torch_list = []

        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]

            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            # mk_cur = mks[:, i, :, :, :]

            # feat_prop_lv0 = x_lr_lv0_cur
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)
            # print(torch.cuda.memory_allocated(0), '14')
            # print(torch.cuda.max_memory_allocated(0), '14')
            # torch.cuda.reset_max_memory_allocated(0)

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]
                
                torch.cuda.synchronize()
                start.record()
                # flow_lv3 = flow
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                # flow_lv3 = self.img_upsample_4x(flow) * 4.
                # flow_lv3 = self.img_upsample_8x(flow) * 8.

                feat_prop_lv3 = self.downsample(feat_prop_lv3)

                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                end.record()
                torch.cuda.synchronize()
                flow_torch_list.append(start.elapsed_time(end) / 1000)

                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3_ = torch.cat([feat_prop_lv0, feat_prop_lv3_, flow_lv3], dim=1)
                feat_prop_lv3_ = self.dcn_pre(feat_prop_lv3_)
                feat_prop_lv3_ = self.dcn_block(feat_prop_lv3_)
                feat_offset_lv3 = self.dcn_offset(feat_prop_lv3_)
                feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                B, C, H, W = feat_offset_lv3.size()
                feat_offset_lv3 = feat_offset_lv3.view(B, 2, C//2, H, W)
                feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).unsqueeze(2).repeat(1, 1, C//2, 1, 1)
                feat_offset_lv3 = feat_offset_lv3.repeat(1, 9, 1, 1, 1).view(B, C*9, H, W)
                # feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                # feat_offset_lv3 = feat_offset_lv3 + torch.cat((flow_lv3[:,1:2,:,:], flow_lv3[:,0:1,:,:]), dim=1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                feat_mask_lv3 = self.dcn_mask(feat_prop_lv3_)
                feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
                feat_mask_lv3 = feat_mask_lv3.repeat(1, 9, 1, 1)
                # feat_prop_lv3 = self.dcn(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_torch_list.append(start.elapsed_time(end) / 1000)
                
                torch.cuda.synchronize()
                start.record()
                # feat_prop_lv3_ = torch.cat([feat_prop_lv0, feat_prop_lv3, flow_lv3], dim=1)
                # feat_prop_lv3_ = self.dcn_pre(feat_prop_lv3_)
                # feat_prop_lv3_ = self.dcn_block(feat_prop_lv3_)
                # feat_offset_lv3 = self.dcn_offset(feat_prop_lv3_)
                # feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                # # B, C, H, W = feat_offset_lv3.size()
                # # feat_offset_lv3 = feat_offset_lv3.view(B, 2, C//2, H, W)
                # # feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).unsqueeze(2).repeat(1, 1, C//2, 1, 1)
                # # feat_offset_lv3 = feat_offset_lv3.repeat(1, 9, 1, 1, 1).view(B, C*9, H, W)
                # feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                # # feat_offset_lv3 = feat_offset_lv3 + torch.cat((flow_lv3[:,1:2,:,:], flow_lv3[:,0:1,:,:]), dim=1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                # feat_mask_lv3 = self.dcn_mask(feat_prop_lv3_)
                # feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
                # # feat_mask_lv3 = feat_mask_lv3.repeat(1, 9, 1, 1)
                feat_prop_lv3 = self.dcn(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)
                end.record()
                torch.cuda.synchronize()
                dcn_torch_list.append(start.elapsed_time(end) / 1000)

                # del flow
                # del flow_lv3
                # del feat_prop_lv3_
                # del feat_mask_lv3
                # del feat_offset_lv3
                # print(torch.cuda.memory_allocated(0), '13.1')
            
            torch.cuda.synchronize()
            start.record()
            feat_prop_lv3 = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
            feat_prop_lv3 = self.forward_resblocks(feat_prop_lv3)
            # print(torch.cuda.memory_allocated(0), '14')
            # print(torch.cuda.max_memory_allocated(0), '14')
            # torch.cuda.reset_max_memory_allocated(0)

            # feat_prop_lv3__ = self.upsample_post(feat_prop_lv3)
            # feat_prop_lv3_ = torch.cat([feat_prop_lv3__, x_hr_lv3_cur], dim=1)
            feat_prop_lv3 = self.upsample_post(feat_prop_lv3)
            _, _, h, w = x_hr_lv3_cur.size()
            feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :h, :w], x_hr_lv3_cur], dim=1)

            # feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
            # _, _, h, w = x_hr_lv3_cur.size()
            # feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :h, :w], x_hr_lv3_cur], dim=1)

            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            # feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            # print(torch.cuda.memory_allocated(0), '15')
            # print(torch.cuda.max_memory_allocated(0), '15')
            # torch.cuda.reset_max_memory_allocated(0)

            # feat_prop_lv3_ = mk_cur.float() * feat_prop_lv3_ + (1 - mk_cur.float()) * feat_prop_lv3__
            # feat_prop_lv3 = mk_cur.float() * feat_prop_lv3_ + (1 - mk_cur.float()) * feat_prop_lv3
            feat_prop_lv3[:, :, :h, :w] = feat_prop_lv3_
            # feat_prop_lv3[:, :, :h, :w] = feat_prop_lv3_[:, :, :h, :w]

            end.record()
            torch.cuda.synchronize()
            res_torch_list.append(start.elapsed_time(end) / 1000)


            torch.cuda.synchronize()
            start.record()
            # feat_prop_lv3 = self.lrelu(self.conv_hr(feat_prop_lv3))
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            # print(torch.cuda.memory_allocated(0), '16')
            # print(torch.cuda.max_memory_allocated(0), '16')
            # torch.cuda.reset_max_memory_allocated(0)

            out = feat_prop_lv3
            # out = feat_prop_lv3
            out = self.conv_last(out)
            base = self.img_upsample_8x(lr_cur)
            out += base

            outputs.append(out.cpu())
            # outputs.append(out)
            end.record()
            torch.cuda.synchronize()
            last_torch_list.append(start.elapsed_time(end) / 1000)

            # print(torch.cuda.memory_allocated(0), '17')
            # print(torch.cuda.max_memory_allocated(0), '17')
            # torch.cuda.reset_max_memory_allocated(0)
            # del out
            # print(torch.cuda.memory_allocated(0), '18')
            # del feat_prop_lv3_
            # print(torch.cuda.memory_allocated(0), '18')
            # del feat_prop_lv3__
            # print(torch.cuda.memory_allocated(0), '18')
        
        a = 0
        print(sum(flow_torch_list)/len(flow_torch_list), 'flow_torch')
        print(sum(dcn_pre_torch_list)/len(dcn_pre_torch_list), 'dcn_pre_torch')
        print(sum(dcn_torch_list)/len(dcn_torch_list), 'dcn_torch')
        print(sum(res_torch_list)/len(res_torch_list), 'res_torch')
        print(sum(last_torch_list)/len(last_torch_list), 'last_torch')
        
        a+=sum(flow_torch_list)/len(flow_torch_list)
        a+=sum(dcn_pre_torch_list)/len(dcn_pre_torch_list)
        a+=sum(dcn_torch_list)/len(dcn_torch_list)
        a+=sum(res_torch_list)/len(res_torch_list)
        a+=sum(last_torch_list)/len(last_torch_list)
        
        print(a, 'total_torch')

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

class MRCF_simple_duf(nn.Module):

    def __init__(self, device, mid_channels=16, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels // 8
        self.dg_num = 16
        self.dk = 3
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

        self.dcn_pre = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_block = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_offset = nn.Conv2d(mid_channels, (self.dg_num)*2*self.dk*self.dk, 3, 1, 1, bias=True)
        self.dcn_mask = nn.Conv2d(mid_channels, (self.dg_num)*1*self.dk*self.dk, 3, 1, 1, bias=True)
        self.dcn = DCNv2(mid_channels, mid_channels, self.dk,
                         stride=1, padding=(self.dk-1)//2, dilation=1,
                         deformable_groups=self.dg_num)
        self.init_dcn()


        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_lr(self.last_channels)

        # self.conv_tttf = conv3x3((mid_channels) * 2, mid_channels)
        self.conv_tttf = conv3x3(3 + self.last_channels, 3)

        # propagation branches
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels * 2, self.last_channels, 1)

        # downsample
        self.downsample = PixelUnShufflePack(
            3, mid_channels, 4, downsample_kernel=3)

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)

        self.conv_hr = nn.Conv2d(self.last_channels, self.last_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(self.last_channels, 3, 3, 1, 1)

        ### 8x settings
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_upsample_4x = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # generate a local expansion filter, similar to im2col
        self.filter_size = (3, 3)
        self.filter_prod = int(np.prod(self.filter_size))
        expansion_filter = torch.eye(self.filter_prod).view(self.filter_prod, 1, *self.filter_size)  # (kh*kw, 1, kh, kw)
        self.expansion_filter = expansion_filter.repeat(3, 1, 1, 1)  # repeat for all the 3 channels
        self.conv_x = nn.Conv2d(self.last_channels, 3 * 4 * 4, 3, 1, 1)
        self.conv_f = nn.Conv2d(self.last_channels, 1 * self.filter_prod * 4 * 4, 3, 1, 1)


    # @profile
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

        flows_backward = None
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    # @profile
    # def forward(self, lrs, fvs, mks):
    def forward(self, lrs, fvs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lrs.size()

        ### compute optical flow
        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end) / 1000, 'flow')
        
        ### forward-time propagation and upsampling
        outputs = []

        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)

        torch.cuda.synchronize()
        start.record()
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end) / 1000, 'en_lr')


        torch.cuda.synchronize()
        start.record()
        B, N, C, H, W = fvs.size()
        # _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)
        _, _, x_hr_lv3 = self.encoder_hr(fvs.view(B*N, C, H, W), islr=True)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end) / 1000, 'en_hr')

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        
        res_torch_list = []        
        dcn_pre_torch_list = []
        dcn_torch_list = []
        flow_torch_list = []

        last_torch_list = []

        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]

            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]

            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]
                
                torch.cuda.synchronize()
                start.record()
                flow_lv3 = self.img_upsample_2x(flow) * 2.

                feat_prop_lv3 = self.downsample(feat_prop_lv3)

                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                end.record()
                torch.cuda.synchronize()
                flow_torch_list.append(start.elapsed_time(end) / 1000)

                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3_ = torch.cat([feat_prop_lv0, feat_prop_lv3_, flow_lv3], dim=1)
                feat_prop_lv3_ = self.dcn_pre(feat_prop_lv3_)
                feat_prop_lv3_ = self.dcn_block(feat_prop_lv3_)
                feat_offset_lv3 = self.dcn_offset(feat_prop_lv3_)
                feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                feat_mask_lv3 = self.dcn_mask(feat_prop_lv3_)
                feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_torch_list.append(start.elapsed_time(end) / 1000)
                
                torch.cuda.synchronize()
                start.record()    
                feat_prop_lv3 = self.dcn(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)
                end.record()
                torch.cuda.synchronize()
                dcn_torch_list.append(start.elapsed_time(end) / 1000)

            torch.cuda.synchronize()
            start.record()
            feat_prop_lv3 = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
            feat_prop_lv3 = self.forward_resblocks(feat_prop_lv3)

            B, _, h, w = feat_prop_lv3.size()
            x_base = self.img_upsample_2x(lr_cur)
            # feat_prop_lv3_x = self.conv_x(self.lrelu(feat_prop_lv3))
            feat_prop_lv3_f = self.conv_f(self.lrelu(feat_prop_lv3))
            feat_prop_lv3_f = F.softmax(feat_prop_lv3_f.view(B, self.filter_prod, 4**2, h, w), dim=1)
            n, filter_prod, upsampling_square, h, w = feat_prop_lv3_f.size()
            kh, kw = self.filter_size
            expanded_input = F.conv2d(
                x_base,
                self.expansion_filter.to(x_base),
                padding=(kh // 2, kw // 2),
                groups=3)  # (n, 3*filter_prod, h, w)
            expanded_input = expanded_input.view(n, 3, filter_prod, h, w).permute(0, 3, 4, 1, 2)  # (n, h, w, 3, filter_prod)
            feat_prop_lv3_f = feat_prop_lv3_f.permute(0, 3, 4, 1, 2)  # (n, h, w, filter_prod, upsampling_square]
            feat_prop_lv3 = torch.matmul(expanded_input, feat_prop_lv3_f)  # (n, h, w, 3, upsampling_square)
            feat_prop_lv3 = feat_prop_lv3.permute(0, 3, 4, 1, 2).view(n, 3 * upsampling_square, h, w)
            # feat_prop_lv3 += feat_prop_lv3_x
            feat_prop_lv3 = F.pixel_shuffle(feat_prop_lv3, 4)

            _, _, h, w = x_hr_lv3_cur.size()
            feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :h, :w], x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3[:, :, :h, :w] = feat_prop_lv3_

            end.record()
            torch.cuda.synchronize()
            res_torch_list.append(start.elapsed_time(end) / 1000)

            torch.cuda.synchronize()
            start.record()
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            out = feat_prop_lv3
            base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out.cpu())
            end.record()
            torch.cuda.synchronize()
            last_torch_list.append(start.elapsed_time(end) / 1000)

        a = 0
        print(sum(flow_torch_list)/len(flow_torch_list), 'flow_torch')
        print(sum(dcn_pre_torch_list)/len(dcn_pre_torch_list), 'dcn_pre_torch')
        print(sum(dcn_torch_list)/len(dcn_torch_list), 'dcn_torch')
        print(sum(res_torch_list)/len(res_torch_list), 'res_torch')
        print(sum(last_torch_list)/len(last_torch_list), 'last_torch')
        
        a+=sum(flow_torch_list)/len(flow_torch_list)
        a+=sum(dcn_pre_torch_list)/len(dcn_pre_torch_list)
        a+=sum(dcn_torch_list)/len(dcn_torch_list)
        a+=sum(res_torch_list)/len(res_torch_list)
        a+=sum(last_torch_list)/len(last_torch_list)
        
        print(a, 'total_torch')

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

class MRCF_simple_dcn2(nn.Module):

    def __init__(self, device, mid_channels=16, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels
        self.dg_num = 16
        self.dk = 3
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

        self.dcn_pre_0 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_block_0 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_offset_0 = nn.Conv2d(mid_channels, (self.dg_num)*2*self.dk*self.dk, 3, 1, 1, bias=True)
        # self.dcn_offset = nn.Conv2d(mid_channels, (self.dg_num)*2, 3, 1, 1, bias=True)
        self.dcn_mask_0 = nn.Conv2d(mid_channels, (self.dg_num)*1*self.dk*self.dk, 3, 1, 1, bias=True)
        # self.dcn_mask = nn.Conv2d(mid_channels, (self.dg_num)*1, 3, 1, 1, bias=True)
        self.dcn_0 = DCNv2(mid_channels, mid_channels, self.dk,
                         stride=1, padding=(self.dk-1)//2, dilation=1,
                         deformable_groups=self.dg_num)
        self.dcn_pre_1 = nn.Conv2d(mid_channels*2+2, mid_channels, 3, 1, 1, bias=True)
        self.dcn_block_1 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dcn_offset_1 = nn.Conv2d(mid_channels, (self.dg_num)*2*self.dk*self.dk, 3, 1, 1, bias=True)
        # self.dcn_offset = nn.Conv2d(mid_channels, (self.dg_num)*2, 3, 1, 1, bias=True)
        self.dcn_mask_1 = nn.Conv2d(mid_channels, (self.dg_num)*1*self.dk*self.dk, 3, 1, 1, bias=True)
        # self.dcn_mask = nn.Conv2d(mid_channels, (self.dg_num)*1, 3, 1, 1, bias=True)
        self.dcn_1 = DCNv2(mid_channels, mid_channels, self.dk,
                         stride=1, padding=(self.dk-1)//2, dilation=1,
                         deformable_groups=self.dg_num)
        self.init_dcn()


        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_lr(self.last_channels)

        self.conv_tttf = conv3x3((self.last_channels) * 2, self.last_channels)

        # propagation branches
        self.forward_resblocks_0 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv(
            mid_channels * 2, mid_channels, 1)
        
        # downsample
        # self.downsample = PixelUnShufflePack(
            # mid_channels // 4, mid_channels, 2, downsample_kernel=3)
        # self.downsample = PixelUnShufflePack(
            # mid_channels, mid_channels, 4, downsample_kernel=3)
        self.downsample = PixelUnShufflePack(
            self.last_channels, mid_channels, 4, downsample_kernel=3)
        # self.downsample = PixelUnShufflePack(
            # mid_channels // 4, mid_channels, 8, downsample_kernel=3)

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        # self.upsample = PixelShufflePack(
        #     mid_channels, mid_channels, 4, upsample_kernel=3)
        # self.upsample = PixelShufflePack(
            # mid_channels, mid_channels, 8, upsample_kernel=3)
        
        # self.upsample_post = PixelShufflePack(
            # mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)
        # self.upsample_post = PixelShufflePack(
            # mid_channels, mid_channels, 8, upsample_kernel=3)

        self.conv_hr = nn.Conv2d(self.last_channels, self.last_channels, 3, 1, 1)
        # self.conv_hr = nn.Conv2d(mid_channels, self.last_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(self.last_channels, 3, 3, 1, 1)

        ### 8x settings
        self.img_upsample_2x = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.img_upsample_4x = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_upsample_8x = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # @profile
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

    # @profile
    # def forward(self, lrs, fvs, mks):
    def forward(self, lrs, fvs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        # print(lrs.size(), mks.size())
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        # torch.cuda.reset_max_memory_allocated(0)

        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs)
        # print(torch.cuda.memory_allocated(0), '1')
        # print(torch.cuda.max_memory_allocated(0), '1')
        # torch.cuda.reset_max_memory_allocated(0)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end) / 1000, 'flow')
        
        ### forward-time propagation and upsampling
        outputs = []

        # feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        # feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*4, w*4)
        # feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*8, w*8)
        # print(torch.cuda.memory_allocated(0), '2')
        # print(torch.cuda.max_memory_allocated(0), '2')
        # torch.cuda.reset_max_memory_allocated(0)

        ### texture transfer
        # B, N, C, H, W = mks.size()
        # mk_lv3 = mks.float()

        torch.cuda.synchronize()
        start.record()
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        # lrs_lv3 = self.img_upsample_8x(lrs_lv0)
        # print(torch.cuda.memory_allocated(0), '3')
        # print(torch.cuda.max_memory_allocated(0), '3')
        # torch.cuda.reset_max_memory_allocated(0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)
        # print(torch.cuda.memory_allocated(0), '4')
        # print(torch.cuda.max_memory_allocated(0), '4')
        # torch.cuda.reset_max_memory_allocated(0)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end) / 1000, 'en_lr')

        # lrs_lv3_view = lrs_lv3.view(B, N, C, H*8, W*8)
        # mks_float = mks.float()
        # fvs = (fvs * mks_float + lrs_lv3_view * (1 - mks_float))
        # print(torch.cuda.memory_allocated(0), '5')
        # print(torch.cuda.max_memory_allocated(0), '5')
        # torch.cuda.reset_max_memory_allocated(0)

        torch.cuda.synchronize()
        start.record()
        B, N, C, H, W = fvs.size()
        # _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), lrs_lv3), dim=1), islr=True)
        _, _, x_hr_lv3 = self.encoder_hr(fvs.view(B*N, C, H, W), islr=True)
        # print(torch.cuda.memory_allocated(0), '6')
        # print(torch.cuda.max_memory_allocated(0), '6')
        # torch.cuda.reset_max_memory_allocated(0)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end) / 1000, 'en_hr')

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        
        res_torch_list = []        
        dcn_pre_torch_list = []
        dcn_torch_list = []
        flow_torch_list = []

        last_torch_list = []

        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]

            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            # mk_cur = mks[:, i, :, :, :]

            # feat_prop_lv0 = x_lr_lv0_cur
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)

            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]
                
                torch.cuda.synchronize()
                start.record()
                # flow_lv3 = flow
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                # flow_lv3 = self.img_upsample_4x(flow) * 4.
                # flow_lv3 = self.img_upsample_8x(flow) * 8.

                feat_prop_lv3 = self.downsample(feat_prop_lv3)

                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                end.record()
                torch.cuda.synchronize()
                flow_torch_list.append(start.elapsed_time(end) / 1000)

                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3_ = torch.cat([feat_prop_lv0, feat_prop_lv3_, flow_lv3], dim=1)
                feat_prop_lv3_ = self.dcn_pre_0(feat_prop_lv3_)
                feat_prop_lv3_ = self.dcn_block_0(feat_prop_lv3_)
                feat_offset_lv3 = self.dcn_offset_0(feat_prop_lv3_)
                feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                # B, C, H, W = feat_offset_lv3.size()
                # feat_offset_lv3 = feat_offset_lv3.view(B, 2, C//2, H, W)
                # feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).unsqueeze(2).repeat(1, 1, C//2, 1, 1)
                # feat_offset_lv3 = feat_offset_lv3.repeat(1, 9, 1, 1, 1).view(B, C*9, H, W)
                feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                # feat_offset_lv3 = feat_offset_lv3 + torch.cat((flow_lv3[:,1:2,:,:], flow_lv3[:,0:1,:,:]), dim=1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                feat_mask_lv3 = self.dcn_mask_0(feat_prop_lv3_)
                feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
                # feat_mask_lv3 = feat_mask_lv3.repeat(1, 9, 1, 1)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_torch_list.append(start.elapsed_time(end) / 1000)

                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = self.dcn_0(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)
                end.record()
                torch.cuda.synchronize()
                dcn_torch_list.append(start.elapsed_time(end) / 1000)

                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_0(feat_prop_lv3)
                end.record()
                torch.cuda.synchronize()
                res_torch_list.append(start.elapsed_time(end) / 1000)
                
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3_ = torch.cat([feat_prop_lv0, feat_prop_lv3, flow_lv3], dim=1)
                feat_prop_lv3_ = self.dcn_pre_1(feat_prop_lv3_)
                feat_prop_lv3_ = self.dcn_block_1(feat_prop_lv3_)
                feat_offset_lv3 = self.dcn_offset_1(feat_prop_lv3_)
                feat_offset_lv3 = self.max_residue_magnitude * torch.tanh(feat_offset_lv3)
                # # B, C, H, W = feat_offset_lv3.size()
                # # feat_offset_lv3 = feat_offset_lv3.view(B, 2, C//2, H, W)
                # # feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).unsqueeze(2).repeat(1, 1, C//2, 1, 1)
                # # feat_offset_lv3 = feat_offset_lv3.repeat(1, 9, 1, 1, 1).view(B, C*9, H, W)
                feat_offset_lv3 = feat_offset_lv3 + flow_lv3.flip(1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                # # feat_offset_lv3 = feat_offset_lv3 + torch.cat((flow_lv3[:,1:2,:,:], flow_lv3[:,0:1,:,:]), dim=1).repeat(1, feat_offset_lv3.size(1) // 2, 1, 1)
                feat_mask_lv3 = self.dcn_mask_1(feat_prop_lv3_)
                feat_mask_lv3 = torch.sigmoid(feat_mask_lv3)
                # feat_mask_lv3 = feat_mask_lv3.repeat(1, 9, 1, 1)
                end.record()
                torch.cuda.synchronize()
                dcn_pre_torch_list.append(start.elapsed_time(end) / 1000)
                
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = self.dcn_1(feat_prop_lv3, feat_offset_lv3, feat_mask_lv3)
                end.record()
                torch.cuda.synchronize()
                dcn_torch_list.append(start.elapsed_time(end) / 1000)
                
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_1(feat_prop_lv3)
                end.record()
                torch.cuda.synchronize()
                res_torch_list.append(start.elapsed_time(end) / 1000)

            else:
                torch.cuda.synchronize()
                start.record()
                feat_prop_lv3 = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_0(feat_prop_lv3)

                feat_prop_lv3 = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_1(feat_prop_lv3)
                end.record()
                torch.cuda.synchronize()
                res_torch_list.append(start.elapsed_time(end) / 1000)

            torch.cuda.synchronize()
            start.record()
            # feat_prop_lv3__ = self.upsample_post(feat_prop_lv3)
            # feat_prop_lv3_ = torch.cat([feat_prop_lv3__, x_hr_lv3_cur], dim=1)
            feat_prop_lv3 = self.upsample_post(feat_prop_lv3)
            _, _, h, w = x_hr_lv3_cur.size()
            feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :h, :w], x_hr_lv3_cur], dim=1)

            # feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
            # _, _, h, w = x_hr_lv3_cur.size()
            # feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :h, :w], x_hr_lv3_cur], dim=1)

            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)

            # feat_prop_lv3_ = mk_cur.float() * feat_prop_lv3_ + (1 - mk_cur.float()) * feat_prop_lv3__
            # feat_prop_lv3 = mk_cur.float() * feat_prop_lv3_ + (1 - mk_cur.float()) * feat_prop_lv3
            feat_prop_lv3[:, :, :h, :w] = feat_prop_lv3_
            # feat_prop_lv3[:, :, :h, :w] = feat_prop_lv3_[:, :, :h, :w]

            # feat_prop_lv3 = self.lrelu(self.conv_hr(feat_prop_lv3))
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)

            out = feat_prop_lv3
            # out = feat_prop_lv3
            out = self.conv_last(out)
            base = self.img_upsample_8x(lr_cur)
            out += base

            outputs.append(out.cpu())
            # outputs.append(out)
            end.record()
            torch.cuda.synchronize()
            last_torch_list.append(start.elapsed_time(end) / 1000)

        a = 0
        print(sum(flow_torch_list)/len(flow_torch_list), 'flow_torch')
        print(sum(dcn_pre_torch_list)/len(dcn_pre_torch_list), 'dcn_pre_torch')
        print(sum(dcn_torch_list)/len(dcn_torch_list), 'dcn_torch')
        print(sum(res_torch_list)/len(res_torch_list), 'res_torch')
        print(sum(last_torch_list)/len(last_torch_list), 'last_torch')
        
        a+=sum(flow_torch_list)/len(flow_torch_list)
        a+=sum(dcn_pre_torch_list)/len(dcn_pre_torch_list)
        a+=sum(dcn_torch_list)/len(dcn_torch_list)
        a+=sum(res_torch_list)/len(res_torch_list)
        a+=sum(last_torch_list)/len(last_torch_list)
        
        print(a, 'total_torch')

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
    def init_dcn(self):

        self.dcn_offset_0.weight.data.zero_()
        self.dcn_offset_0.bias.data.zero_()
        self.dcn_mask_0.weight.data.zero_()
        self.dcn_mask_0.bias.data.zero_()
        self.conv_identify(self.dcn_0.weight, self.dcn_0.bias)

        self.dcn_offset_1.weight.data.zero_()
        self.dcn_offset_1.bias.data.zero_()
        self.dcn_mask_1.weight.data.zero_()
        self.dcn_mask_1.bias.data.zero_()
        self.conv_identify(self.dcn_1.weight, self.dcn_1.bias)

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

class MRCF_simple_v1_dcn2_v4_kai(nn.Module):

    def __init__(self, device, mid_channels=16, y_only=False, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels // 1
        self.dg_num = 16
        self.dk = 1
        self.y_only = y_only
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk)

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_single(self.last_channels)

        self.conv_tttf = conv3x3(self.last_channels * 2, self.last_channels)
        # self.conv_tttf = conv3x3(mid_channels * 2, mid_channels)

        # propagation branches
        self.forward_resblocks_0 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        
        # downsample
        self.downsample = PixelUnShufflePack(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)

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

    # @profile
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

                feat_prop_lv3 = self.downsample(feat_prop_lv3)

                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                feat_prop_lv3 = self.dcn_0(feat_prop_lv0, feat_prop_lv3, feat_prop_lv3_, flow_lv3)
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)

                feat_prop_lv3 = self.dcn_1(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3, flow_lv3)
                
                feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_1(feat_prop_lv3)
            else:
                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)

                feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_1(feat_prop_lv3)
            
            feat_prop_lv3 = self.lrelu(self.upsample_post(feat_prop_lv3))
            feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3 = mk_cur.float() * feat_prop_lv3_ + (1 - mk_cur.float()) * feat_prop_lv3
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            out = feat_prop_lv3
            
            out = self.conv_last(out)
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

class MRCF_simple_v1_dcn2_v4_kai(nn.Module):

    def __init__(self, device, mid_channels=16, y_only=False, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels // 1
        self.dg_num = 16
        self.dk = 1
        self.y_only = y_only
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk)

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_single(self.last_channels)

        self.conv_tttf = conv3x3(self.last_channels * 2, self.last_channels)
        # self.conv_tttf = conv3x3(mid_channels * 2, mid_channels)

        # propagation branches
        self.forward_resblocks_0 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        
        # downsample
        self.downsample = PixelUnShufflePack(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)

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

    # @profile
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

                feat_prop_lv3 = self.downsample(feat_prop_lv3)

                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                feat_prop_lv3 = self.dcn_0(feat_prop_lv0, feat_prop_lv3, feat_prop_lv3_, flow_lv3)
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)

                feat_prop_lv3 = self.dcn_1(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3, flow_lv3)
                
                feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_1(feat_prop_lv3)
            else:
                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)

                feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_1(feat_prop_lv3)
            
            feat_prop_lv3 = self.lrelu(self.upsample_post(feat_prop_lv3))
            feat_prop_lv3_ = torch.cat([feat_prop_lv3, x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3 = mk_cur.float() * feat_prop_lv3_ + (1 - mk_cur.float()) * feat_prop_lv3
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            out = feat_prop_lv3
            
            out = self.conv_last(out)
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

class MRCF_simple_v1_dcn2_v4_pcd(nn.Module):

    def __init__(self, device, mid_channels=16, y_only=False, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels // 1
        self.dg_num = 16
        self.dk = 1
        self.y_only = y_only
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)

        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk)

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_single(self.last_channels)

        self.conv_tttf = conv3x3(self.last_channels * 2, self.last_channels)
        # self.conv_tttf = conv3x3(mid_channels * 2, mid_channels)

        # propagation branches
        self.forward_resblocks_0 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv(
            mid_channels*2, mid_channels, 1)
        
        # downsample
        self.downsample = PixelUnShufflePack(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)

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

    # @profile
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

                feat_prop_lv3 = self.downsample(feat_prop_lv3)

                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                feat_prop_lv3 = self.dcn_0(feat_prop_lv0, feat_prop_lv3, feat_prop_lv3_, flow_lv3)
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)

                feat_prop_lv3 = self.dcn_1(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3, flow_lv3)
                
                feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_1(feat_prop_lv3)
            else:
                feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)

                feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                feat_prop_lv3 = self.forward_resblocks_1(feat_prop_lv3)
            
            feat_prop_lv3 = self.lrelu(self.upsample_post(feat_prop_lv3))
            B, C, H, W = x_hr_lv3_cur.size()
            feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            out = feat_prop_lv3
            
            out = self.conv_last(out)
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

class MRCF_simple_v1_dcn2_v4_kai(nn.Module):

    def __init__(self, device, mid_channels=16, y_only=False, spynet_pretrained=None):

        super().__init__()

        self.device = device
        self.mid_channels = mid_channels
        self.last_channels = mid_channels // 8
        self.dg_num = 8
        self.dk = 3
        self.y_only = y_only
        self.max_residue_magnitude = 10

        # optical flow network for feature alignment
        # self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)
        self.spynet = FNet(in_nc=3)
        # self.spynet.load_state_dict(torch.load(spynet_pretrained))

        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, repeat=True)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, repeat=True, pre_offset=True, interpolate='none')
        self.dcn_2 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, repeat=True, pre_offset=True, interpolate='none')
        self.dcn_3 = DCN_module(self.last_channels, 1,  self.dk, self.max_residue_magnitude, repeat=True, pre_offset=True, interpolate='pixelshuffle')
        # self.dcn_3 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=True, interpolate='none')

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_single(self.last_channels)

        self.conv_tttf = conv3x3(self.last_channels * 2, self.last_channels)

        #### propagation branches
        self.forward_resblocks_0_ = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        self.forward_resblocks_1_ = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        self.forward_resblocks_2_ = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        self.forward_resblocks_3_ = ResidualBlocksWithInputConv(
            self.last_channels, self.last_channels, 1)

        self.forward_resblocks_0 = ResidualBlocksWithInputConv(
            mid_channels*3, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv(
            mid_channels*3, mid_channels, 1)
        self.forward_resblocks_2 = ResidualBlocksWithInputConv(
            mid_channels*3, mid_channels, 1)
        self.forward_resblocks_3 = ResidualBlocksWithInputConv(
            self.last_channels*3, self.last_channels, 1)

        # self.forward_resblocks_0 = ResidualBlocksWithInputConv(
        #     mid_channels*2, mid_channels, 1)
        # self.forward_resblocks_1 = ResidualBlocksWithInputConv(
        #     mid_channels*2, mid_channels, 1)
        # self.forward_resblocks_2 = ResidualBlocksWithInputConv(
        #     mid_channels*2, mid_channels, 1)
        # self.forward_resblocks_3 = ResidualBlocksWithInputConv(
        #     self.last_channels*2, self.last_channels, 1)

        # downsample
        self.downsample = PixelUnShufflePack_v2(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)

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

    def forward(self, lrs, fvs, warp_size=(1080, 1920)):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        flow_list = []
        enc_list = []
        dcn_list = []
        res_list = []
        last_list = []
        WP_h, WP_w = warp_size
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs[:, :, :, :WP_h // 8, :WP_w // 8])
        end.record()
        torch.cuda.synchronize()
        flow_list.append(start.elapsed_time(end) / 1000)
        
        ### forward-time propagation and upsampling
        outputs = []

        # feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        # feat_prop_lv3_0 = lrs.new_zeros(n, self.last_channels, h*8, w*8)

        torch.cuda.synchronize()
        start.record()
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        lrs_lv3_view = lrs_lv3.view(B, N, C, H*8, W*8)

        B, N, C, H, W = fvs.size()
        _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), fvs.view(B*N, C, H, W)), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        end.record()
        torch.cuda.synchronize()
        enc_list.append(start.elapsed_time(end) / 1000)
        
        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)

            if i > 0:  # no warping required for the first timestep
                torch.cuda.synchronize()
                start.record()
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                flow_lv0 = self.img_upsample_8x(flow) * 8.

                feat_prop_lv3_0 = feat_prop_lv3
                feat_prop_lv3 = self.downsample(feat_prop_lv3)
                feat_prop_lv3_ = flow_warp(feat_prop_lv3, flow_lv3.permute(0, 2, 3, 1))
                feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))

                #### v15
                # L0      
                feat_prop_lv3_a, offset = self.dcn_0(feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4],
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_,
                                                     flow_lv3)

                feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a, feat_prop_lv3_], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_, feat_prop_lv0)
                
                # L1
                feat_prop_lv3_a, offset = self.dcn_1(feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4],
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_,
                                                     flow_lv3, offset)
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a, feat_prop_lv3_], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_, feat_prop_lv0)

                # L2
                feat_prop_lv3_a, offset = self.dcn_2(feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4],
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_, 
                                                     flow_lv3, offset)
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a, feat_prop_lv3_], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_, feat_prop_lv0)

                # L3
                feat_prop_lv0 = self.lrelu(self.upsample_post(feat_prop_lv0))
                feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0[:, :, :WP_h, :WP_w],
                                                feat_prop_lv3_0,
                                                feat_prop_lv3_0_,
                                                flow_lv0, offset)
                # feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3_, flow_lv3, offset)
                
                feat_prop_lv3 = torch.cat([feat_prop_lv0[:, :, :WP_h, :WP_w], feat_prop_lv3_a, feat_prop_lv3_0_], dim=1)
                # feat_prop_lv3 = torch.cat([feat_prop_lv0[:, :, :WP_h, :WP_w], feat_prop_lv3_a], dim=1)
                # feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_a, feat_prop_lv3_], dim=1)
                # feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_a], dim=1)
                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3, feat_prop_lv0)
                end.record()
                torch.cuda.synchronize()
                dcn_list.append(start.elapsed_time(end) / 1000)
            else:
                torch.cuda.synchronize()
                start.record()
                #### v15
                # L0
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3, feat_prop_lv3], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3], dim=1)
                # feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_)
                feat_prop_lv0 = self.forward_resblocks_0_(feat_prop_lv0)
                
                # L1
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0, feat_prop_lv3, feat_prop_lv3], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                # feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_)
                feat_prop_lv0 = self.forward_resblocks_1_(feat_prop_lv0)

                # L2
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3], dim=1)
                # feat_prop_lv0_ = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                # feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_)
                feat_prop_lv0 = self.forward_resblocks_2_(feat_prop_lv0)

                # L3
                feat_prop_lv0 = self.lrelu(self.upsample_post(feat_prop_lv0))
                # feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3, feat_prop_lv3], dim=1)
                # feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3], dim=1)
                # feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_0, feat_prop_lv3_0], dim=1)
                # feat_prop_lv3 = torch.cat([feat_prop_lv0_, feat_prop_lv3_0], dim=1)
                # feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3)
                feat_prop_lv3 = self.forward_resblocks_3_(feat_prop_lv0)
                end.record()
                torch.cuda.synchronize()
                res_list.append(start.elapsed_time(end) / 1000)
            
            # feat_prop_lv3 = self.lrelu(self.upsample_post(feat_prop_lv3))
            torch.cuda.synchronize()
            start.record()
            B, C, H, W = x_hr_lv3_cur.size()
            feat_prop_lv3_ = torch.cat([feat_prop_lv0[:, :, :H, :W], x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv0[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
            feat_prop_lv3 = self.lrelu(feat_prop_lv0)
            out = feat_prop_lv3
            feat_prop_lv3 = feat_prop_lv3[:, :, :WP_h, :WP_w]

            out = self.conv_last(out)
            base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)
            end.record()
            torch.cuda.synchronize()
            last_list.append(start.elapsed_time(end) / 1000)

        print(sum(flow_list)/len(flow_list), 'flow')
        print(sum(enc_list)/len(enc_list), 'enc')
        print(sum(dcn_list)/len(dcn_list), 'dcn')
        print(sum(res_list)/len(res_list), 'res')
        print(sum(last_list)/len(last_list), 'last')
        print(sum(flow_list)/len(flow_list) +
              sum(enc_list)/len(enc_list) + 
              sum(dcn_list)/len(dcn_list) + 
              sum(last_list)/len(last_list), 'total')

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

class MRCF_simple_v13(nn.Module):

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
        # self.spynet.load_state_dict(torch.load(spynet_pretrained))

        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        self.dcn_2 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        if self.hr_dcn:
            self.dcn_3 = DCN_module(self.last_channels, 1, self.dk, self.max_residue_magnitude, repeat=True, pre_offset=self.offset_prop, interpolate='pixelshuffle')
        else:
            self.dcn_3 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        # self.dcn_3 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=True, interpolate='none')

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_single(self.last_channels)

        self.conv_tttf = conv3x3(self.last_channels * 2, self.last_channels)

        #### propagation branches
        self.forward_resblocks_0_ = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        self.forward_resblocks_1_ = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        self.forward_resblocks_2_ = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        if self.hr_dcn:
            self.forward_resblocks_3_ = ResidualBlocksWithInputConv(
                self.last_channels, self.last_channels, 1)
        else:
            self.forward_resblocks_3_ = ResidualBlocksWithInputConv(
                mid_channels, mid_channels, 1)

        self.forward_resblocks_0 = ResidualBlocksWithInputConv_v2(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv_v2(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_2 = ResidualBlocksWithInputConv_v2(
            mid_channels*2, mid_channels, 1)
        if self.hr_dcn:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv_v2(
                self.last_channels*2, self.last_channels, 1)
        else:
            self.forward_resblocks_3 = ResidualBlocksWithInputConv_v2(
                mid_channels*2, mid_channels, 1)

        # downsample
        self.downsample = PixelUnShufflePack_v2(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)

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

    def forward(self, lrs, fvs, warp_size=(1080, 1920)):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        flow_list = []
        enc_list = []
        dcn_list = []
        res_list = []
        last_list = []
        WP_h, WP_w = warp_size
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs[:, :, :, :WP_h // 8, :WP_w // 8])
        end.record()
        torch.cuda.synchronize()
        flow_list.append(start.elapsed_time(end) / 1000)
        
        ### forward-time propagation and upsampling
        outputs = []

        torch.cuda.synchronize()
        start.record()
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)


        B, N, C, H, W = fvs.size()
        _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), fvs.view(B*N, C, H, W)), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        end.record()
        torch.cuda.synchronize()
        enc_list.append(start.elapsed_time(end) / 1000)
        
        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)

            if i > 0:  # no warping required for the first timestep
                torch.cuda.synchronize()
                start.record()
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                flow_lv0 = self.img_upsample_8x(flow) * 8.

                feat_prop_lv3_0 = feat_prop_lv3
                feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))
                feat_prop_lv3_ = self.downsample(feat_prop_lv3_0_)
                feat_prop_lv3 = self.downsample(feat_prop_lv3)

                #### v15
                # L0      
                feat_prop_lv3_a, offset = self.dcn_0(feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4],
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_,
                                                     flow_lv3)
                if not self.offset_prop:
                    offset = None

                feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_, feat_prop_lv0)
                
                # L1
                feat_prop_lv3_a, offset = self.dcn_1(feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4],
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_,
                                                     flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_, feat_prop_lv0)

                # L2
                feat_prop_lv3_a, offset = self.dcn_2(feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4],
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_, 
                                                     flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_, feat_prop_lv0)

                # L3
                if self.hr_dcn:
                    feat_prop_lv0 = self.lrelu(self.upsample_post(feat_prop_lv0_))
                    feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0[:, :, :WP_h, :WP_w],
                                                    feat_prop_lv3_0,
                                                    feat_prop_lv3_0_,
                                                    flow_lv0, offset)
                    feat_prop_lv3 = torch.cat([feat_prop_lv0[:, :, :WP_h, :WP_w], feat_prop_lv3_a], dim=1)
                else:
                    feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0[:, :, :WP_h, :WP_w],
                                                    feat_prop_lv3,
                                                    feat_prop_lv3_,
                                                    flow_lv3, offset)
                    feat_prop_lv3 = torch.cat([feat_prop_lv0[:, :, :WP_h, :WP_w], feat_prop_lv3_a], dim=1)

                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3, feat_prop_lv0)
                end.record()
                torch.cuda.synchronize()
                dcn_list.append(start.elapsed_time(end) / 1000)
            else:
                torch.cuda.synchronize()
                start.record()
                #### v15
                # L0
                feat_prop_lv0 = self.forward_resblocks_0_(feat_prop_lv0)
                
                # L1
                feat_prop_lv0 = self.forward_resblocks_1_(feat_prop_lv0)

                # L2
                feat_prop_lv0 = self.forward_resblocks_2_(feat_prop_lv0)

                # L3
                if self.hr_dcn:
                    feat_prop_lv0 = self.lrelu(self.upsample_post(feat_prop_lv0))
                feat_prop_lv3 = self.forward_resblocks_3_(feat_prop_lv0)
                end.record()
                torch.cuda.synchronize()
                res_list.append(start.elapsed_time(end) / 1000)
            
            if not self.hr_dcn:
                feat_prop_lv3 = self.lrelu(self.upsample_post(feat_prop_lv3))
            torch.cuda.synchronize()
            start.record()
            B, C, H, W = x_hr_lv3_cur.size()
            feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            out = feat_prop_lv3
            feat_prop_lv3 = feat_prop_lv3[:, :, :WP_h, :WP_w]

            out = self.conv_last(out)
            base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)
            end.record()
            torch.cuda.synchronize()
            last_list.append(start.elapsed_time(end) / 1000)

        print(sum(flow_list)/len(flow_list), 'flow')
        print(sum(enc_list)/len(enc_list), 'enc')
        print(sum(dcn_list)/len(dcn_list), 'dcn')
        print(sum(res_list)/len(res_list), 'res')
        print(sum(last_list)/len(last_list), 'last')
        print(sum(flow_list)/len(flow_list) +
              sum(enc_list)/len(enc_list) + 
              sum(dcn_list)/len(dcn_list) + 
              sum(last_list)/len(last_list), 'total')

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
            model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
            model_state_dict = self.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.load_state_dict(model_state_dict, strict=strict)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class MRCF_simple_v13_nodcn(nn.Module):

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
        # self.spynet.load_state_dict(torch.load(spynet_pretrained))

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

        #### propagation branches
        self.forward_resblocks_0_ = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        self.forward_resblocks_1_ = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        self.forward_resblocks_2_ = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        self.forward_resblocks_3_ = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)

        self.forward_resblocks_0 = ResidualBlocksWithInputConv_v2(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv_v2(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_2 = ResidualBlocksWithInputConv_v2(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_3 = ResidualBlocksWithInputConv_v2(
            mid_channels*2, mid_channels, 1)

        # downsample
        self.downsample = PixelUnShufflePack_v2(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)

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

    def forward(self, lrs, fvs, warp_size=(1080, 1920)):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        flow_list = []
        enc_list = []
        dcn_list = []
        res_list = []
        last_list = []
        WP_h, WP_w = warp_size
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs[:, :, :, :WP_h // 8, :WP_w // 8])
        end.record()
        torch.cuda.synchronize()
        flow_list.append(start.elapsed_time(end) / 1000)
        
        ### forward-time propagation and upsampling
        outputs = []

        torch.cuda.synchronize()
        start.record()
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)


        B, N, C, H, W = fvs.size()
        _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), fvs.view(B*N, C, H, W)), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        end.record()
        torch.cuda.synchronize()
        enc_list.append(start.elapsed_time(end) / 1000)
        
        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)

            if i > 0:  # no warping required for the first timestep
                torch.cuda.synchronize()
                start.record()
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                flow_lv0 = self.img_upsample_8x(flow) * 8.

                feat_prop_lv3_0 = feat_prop_lv3
                feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))
                feat_prop_lv3_ = self.downsample(feat_prop_lv3_0_)
                feat_prop_lv3 = self.downsample(feat_prop_lv3)

                #### v13
                # L0
                feat_prop_lv3_a = self.dcn_0(torch.cat((feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_, flow_lv3), dim=1))
                if not self.offset_prop:
                    offset = None

                feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_, feat_prop_lv0)
                
                # L1
                feat_prop_lv3_a = self.dcn_1(torch.cat((feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_, flow_lv3), dim=1))
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_, feat_prop_lv0)

                # L2
                feat_prop_lv3_a = self.dcn_2(torch.cat((feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_, flow_lv3), dim=1))
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_, feat_prop_lv0)

                # L3
                feat_prop_lv3_a = self.dcn_3(torch.cat((feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_, flow_lv3), dim=1))
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a], dim=1)
                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv0_, feat_prop_lv0)
                end.record()
                torch.cuda.synchronize()
                dcn_list.append(start.elapsed_time(end) / 1000)
            else:
                torch.cuda.synchronize()
                start.record()
                #### v15
                # L0
                feat_prop_lv0 = self.forward_resblocks_0_(feat_prop_lv0)
                
                # L1
                feat_prop_lv0 = self.forward_resblocks_1_(feat_prop_lv0)

                # L2
                feat_prop_lv0 = self.forward_resblocks_2_(feat_prop_lv0)

                # L3
                feat_prop_lv3 = self.forward_resblocks_3_(feat_prop_lv0)
                end.record()
                torch.cuda.synchronize()
                res_list.append(start.elapsed_time(end) / 1000)
            
            feat_prop_lv3 = self.lrelu(self.upsample_post(feat_prop_lv3))
            torch.cuda.synchronize()
            start.record()
            B, C, H, W = x_hr_lv3_cur.size()
            feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            out = feat_prop_lv3
            feat_prop_lv3 = feat_prop_lv3[:, :, :WP_h, :WP_w]

            out = self.conv_last(out)
            base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)
            end.record()
            torch.cuda.synchronize()
            last_list.append(start.elapsed_time(end) / 1000)

        print(sum(flow_list)/len(flow_list), 'flow')
        print(sum(enc_list)/len(enc_list), 'enc')
        print(sum(dcn_list)/len(dcn_list), 'dcn')
        print(sum(res_list)/len(res_list), 'res')
        print(sum(last_list)/len(last_list), 'last')
        print(sum(flow_list)/len(flow_list) +
              sum(enc_list)/len(enc_list) + 
              sum(dcn_list)/len(dcn_list) + 
              sum(last_list)/len(last_list), 'total')

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
            model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
            model_state_dict = self.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.load_state_dict(model_state_dict, strict=strict)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class MRCF_simple_v15(nn.Module):

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
        # self.spynet.load_state_dict(torch.load(spynet_pretrained))

        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        self.dcn_2 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        if self.hr_dcn:
            self.dcn_3 = DCN_module(self.last_channels, 1, self.dk, self.max_residue_magnitude, repeat=True, pre_offset=self.offset_prop, interpolate='pixelshuffle')
        else:
            self.dcn_3 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        # self.dcn_3 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=True, interpolate='none')

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_single(self.last_channels)

        self.conv_tttf = conv3x3(self.last_channels * 2, self.last_channels)

        #### propagation branches
        self.forward_resblocks_0_ = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        self.forward_resblocks_1_ = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        self.forward_resblocks_2_ = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, 1)
        if self.hr_dcn:
            self.forward_resblocks_3_ = ResidualBlocksWithInputConv(
                self.last_channels, self.last_channels, 1)
        else:
            self.forward_resblocks_3_ = ResidualBlocksWithInputConv(
                mid_channels, mid_channels, 1)

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

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            mid_channels, self.last_channels, 4, upsample_kernel=3)

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

    def forward(self, lrs, fvs, warp_size=(1080, 1920)):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        flow_list = []
        enc_list = []
        dcn_list = []
        res_list = []
        last_list = []
        WP_h, WP_w = warp_size
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs[:, :, :, :WP_h // 8, :WP_w // 8])
        end.record()
        torch.cuda.synchronize()
        flow_list.append(start.elapsed_time(end) / 1000)
        
        ### forward-time propagation and upsampling
        outputs = []

        torch.cuda.synchronize()
        start.record()
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)


        B, N, C, H, W = fvs.size()
        _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), fvs.view(B*N, C, H, W)), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        end.record()
        torch.cuda.synchronize()
        enc_list.append(start.elapsed_time(end) / 1000)
        
        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)

            if i > 0:  # no warping required for the first timestep
                torch.cuda.synchronize()
                start.record()
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                flow_lv0 = self.img_upsample_8x(flow) * 8.

                feat_prop_lv3_0 = feat_prop_lv3
                feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))
                feat_prop_lv3_ = self.downsample(feat_prop_lv3_0_)
                feat_prop_lv3 = self.downsample(feat_prop_lv3)

                #### v15
                # L0      
                feat_prop_lv3_a, offset = self.dcn_0(feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4],
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_,
                                                     flow_lv3)
                if not self.offset_prop:
                    offset = None

                feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a, feat_prop_lv3_], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_, feat_prop_lv0)
                
                # L1
                feat_prop_lv3_a, offset = self.dcn_1(feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4],
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_,
                                                     flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a, feat_prop_lv3_], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_, feat_prop_lv0)

                # L2
                feat_prop_lv3_a, offset = self.dcn_2(feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4],
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_, 
                                                     flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_prop_lv3_a, feat_prop_lv3_], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_, feat_prop_lv0)

                # L3
                if self.hr_dcn:
                    feat_prop_lv0 = self.lrelu(self.upsample_post(feat_prop_lv0_))
                    feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0[:, :, :WP_h, :WP_w],
                                                    feat_prop_lv3_0,
                                                    feat_prop_lv3_0_,
                                                    flow_lv0, offset)
                    feat_prop_lv3 = torch.cat([feat_prop_lv0[:, :, :WP_h, :WP_w], feat_prop_lv3_a, feat_prop_lv3_0_], dim=1)
                else:
                    feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0[:, :, :WP_h, :WP_w],
                                                    feat_prop_lv3,
                                                    feat_prop_lv3_,
                                                    flow_lv3, offset)
                    feat_prop_lv3 = torch.cat([feat_prop_lv0[:, :, :WP_h, :WP_w], feat_prop_lv3_a, feat_prop_lv3_], dim=1)

                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3, feat_prop_lv0)
                end.record()
                torch.cuda.synchronize()
                dcn_list.append(start.elapsed_time(end) / 1000)
            else:
                torch.cuda.synchronize()
                start.record()
                #### v15
                # L0
                feat_prop_lv0 = self.forward_resblocks_0_(feat_prop_lv0)
                
                # L1
                feat_prop_lv0 = self.forward_resblocks_1_(feat_prop_lv0)

                # L2
                feat_prop_lv0 = self.forward_resblocks_2_(feat_prop_lv0)

                # L3
                if self.hr_dcn:
                    feat_prop_lv0 = self.lrelu(self.upsample_post(feat_prop_lv0))
                feat_prop_lv3 = self.forward_resblocks_3_(feat_prop_lv0)
                end.record()
                torch.cuda.synchronize()
                res_list.append(start.elapsed_time(end) / 1000)
            
            if not self.hr_dcn:
                feat_prop_lv3 = self.lrelu(self.upsample_post(feat_prop_lv3))
            torch.cuda.synchronize()
            start.record()
            B, C, H, W = x_hr_lv3_cur.size()
            feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            out = feat_prop_lv3
            feat_prop_lv3 = feat_prop_lv3[:, :, :WP_h, :WP_w]

            out = self.conv_last(out)
            base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)
            end.record()
            torch.cuda.synchronize()
            last_list.append(start.elapsed_time(end) / 1000)

        print(sum(flow_list)/len(flow_list), 'flow')
        print(sum(enc_list)/len(enc_list), 'enc')
        print(sum(dcn_list)/len(dcn_list), 'dcn')
        print(sum(res_list)/len(res_list), 'res')
        print(sum(last_list)/len(last_list), 'last')
        print(sum(flow_list)/len(flow_list) +
              sum(enc_list)/len(enc_list) + 
              sum(dcn_list)/len(dcn_list) + 
              sum(last_list)/len(last_list), 'total')

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
            model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
            model_state_dict = self.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.load_state_dict(model_state_dict, strict=strict)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class MRCF_simple_v18(nn.Module):

    def __init__(self, device, mid_channels=16, y_only=False, hr_dcn=True, offset_prop=True, split_ratio=3, spynet_pretrained=None):

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
        self.split_ratio = split_ratio

        # optical flow network for feature alignment
        # self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)
        self.spynet = FNet(in_nc=3)
        # self.spynet.load_state_dict(torch.load(spynet_pretrained))

        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        self.dcn_2 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        self.dcn_3 = DCN_module(self.last_channels, 1, self.dk, self.max_residue_magnitude, repeat=True, pre_offset=self.offset_prop, interpolate='pixelshuffle')

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)
        self.encoder_hr = LTE.LTE_simple_hr_single(self.last_channels)

        self.conv_tttf = conv3x3(self.last_channels * 2, self.last_channels)

        #### propagation branches
        self.forward_resblocks_0_ = ResidualBlocksWithInputConv(
            (mid_channels * self.split_ratio) // 4, mid_channels, 1)
        self.forward_resblocks_1_ = ResidualBlocksWithInputConv(
            (mid_channels * self.split_ratio) // 4, mid_channels, 1)
        self.forward_resblocks_2_ = ResidualBlocksWithInputConv(
            (mid_channels * self.split_ratio) // 4, mid_channels, 1)
        self.forward_resblocks_3_ = ResidualBlocksWithInputConv(
            self.last_channels, self.last_channels, 1)

        self.forward_resblocks_0 = ResidualBlocksWithInputConv_v2(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv_v2(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_2 = ResidualBlocksWithInputConv_v2(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_3 = ResidualBlocksWithInputConv_v2(
            self.last_channels*2, self.last_channels, 1)

        # downsample
        self.downsample = PixelUnShufflePack_v2(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, (mid_channels * self.split_ratio) // 4, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            (mid_channels * self.split_ratio) // 4, self.last_channels, 4, upsample_kernel=3)

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

    def forward(self, lrs, fvs, warp_size=(1080, 1920)):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        flow_list = []
        enc_list = []
        dcn_list = []
        res_list = []
        last_list = []
        WP_h, WP_w = warp_size
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs[:, :, :, :WP_h // 8, :WP_w // 8])
        end.record()
        torch.cuda.synchronize()
        flow_list.append(start.elapsed_time(end) / 1000)
        
        ### forward-time propagation and upsampling
        outputs = []

        torch.cuda.synchronize()
        start.record()
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        B, N, C, H, W = fvs.size()
        _, _, x_hr_lv3 = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), fvs.view(B*N, C, H, W)), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        end.record()
        torch.cuda.synchronize()
        enc_list.append(start.elapsed_time(end) / 1000)
        
        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)

            if i > 0:  # no warping required for the first timestep
                torch.cuda.synchronize()
                start.record()
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                flow_lv0 = self.img_upsample_8x(flow) * 8.

                feat_prop_lv3_0 = feat_prop_lv3
                feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))
                feat_prop_lv3_ = self.downsample(feat_prop_lv3_0_)
                feat_prop_lv3 = self.downsample(feat_prop_lv3)
                feat_mix = torch.cat((feat_lv0,
                                      feat_lv1,
                                      feat_lv2), dim=1)
                
                feat_mix = flow_warp(feat_mix, flow_lv3.permute(0, 2, 3, 1))
                
                feat_mix = torch.chunk(feat_mix, 3, dim=1)
                feat_lv0 = feat_mix[0]
                feat_lv1 = feat_mix[1]
                feat_lv2 = feat_mix[2]

                #### v15
                # L0
                feat_temp = torch.cat((feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_lv0), dim=1)
                feat_prop_lv3_a, offset = self.dcn_0(feat_temp,
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_,
                                                     flow_lv3)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_temp, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_, feat_temp)
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1) # 5 -> 4
                feat_lv0 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)[:, :, :WP_h // 4, :WP_w // 4]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)
                
                # L1
                feat_temp = torch.cat((feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_lv1), dim=1)
                feat_prop_lv3_a, offset = self.dcn_1(feat_temp,
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_,
                                                     flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_temp, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_, feat_temp)
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1) # 5 -> 4
                feat_lv1 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)[:, :, :WP_h // 4, :WP_w // 4]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)

                # L2
                feat_temp = torch.cat((feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_lv2), dim=1)
                feat_prop_lv3_a, offset = self.dcn_2(feat_temp,
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_, 
                                                     flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_temp, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_, feat_temp)
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1) # 5 -> 4
                feat_lv2 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)[:, :, :WP_h // 4, :WP_w // 4]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)

                # L3
                feat_prop_lv0 = self.lrelu(self.upsample_post(feat_prop_lv0))
                feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0[:, :, :WP_h, :WP_w],
                                                feat_prop_lv3_0,
                                                feat_prop_lv3_0_,
                                                flow_lv0, offset)
                feat_prop_lv3 = torch.cat([feat_prop_lv0[:, :, :WP_h, :WP_w], feat_prop_lv3_a], dim=1)

                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3, feat_prop_lv0)
                end.record()
                torch.cuda.synchronize()
                dcn_list.append(start.elapsed_time(end) / 1000)
            else:
                torch.cuda.synchronize()
                start.record()
                #### v15
                # L0
                feat_prop_lv0 = self.forward_resblocks_0_(feat_prop_lv0)
                feat_prop_lv0 = torch.chunk(feat_prop_lv0, 4, dim=1)
                feat_lv0 = torch.cat(feat_prop_lv0[self.split_ratio:4], dim=1)[:, :, :WP_h // 4, :WP_w // 4]
                feat_prop_lv0 = torch.cat(feat_prop_lv0[:self.split_ratio], dim=1)
                
                # L1
                feat_prop_lv0 = self.forward_resblocks_1_(feat_prop_lv0)
                feat_prop_lv0 = torch.chunk(feat_prop_lv0, 4, dim=1)
                feat_lv1 = torch.cat(feat_prop_lv0[self.split_ratio:4], dim=1)[:, :, :WP_h // 4, :WP_w // 4]
                feat_prop_lv0 = torch.cat(feat_prop_lv0[:self.split_ratio], dim=1)

                # L2
                feat_prop_lv0 = self.forward_resblocks_2_(feat_prop_lv0)
                feat_prop_lv0 = torch.chunk(feat_prop_lv0, 4, dim=1)
                feat_lv2 = torch.cat(feat_prop_lv0[self.split_ratio:4], dim=1)[:, :, :WP_h // 4, :WP_w // 4]
                feat_prop_lv0 = torch.cat(feat_prop_lv0[:self.split_ratio], dim=1)

                # L3
                feat_prop_lv0 = self.lrelu(self.upsample_post(feat_prop_lv0))
                feat_prop_lv3 = self.forward_resblocks_3_(feat_prop_lv0)
                end.record()
                torch.cuda.synchronize()
                res_list.append(start.elapsed_time(end) / 1000)
            
            torch.cuda.synchronize()
            start.record()
            B, C, H, W = x_hr_lv3_cur.size()
            feat_prop_lv3_ = torch.cat([feat_prop_lv3[:, :, :H, :W], x_hr_lv3_cur], dim=1)
            feat_prop_lv3_ = self.conv_tttf(feat_prop_lv3_)
            feat_prop_lv3[:, :, :H, :W] = feat_prop_lv3_[:, :, :H, :W]
            feat_prop_lv3 = self.lrelu(feat_prop_lv3)
            out = feat_prop_lv3
            feat_prop_lv3 = feat_prop_lv3[:, :, :WP_h, :WP_w]

            out = self.conv_last(out)
            base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)
            end.record()
            torch.cuda.synchronize()
            last_list.append(start.elapsed_time(end) / 1000)

        print(sum(flow_list)/len(flow_list), 'flow')
        print(sum(enc_list)/len(enc_list), 'enc')
        print(sum(dcn_list)/len(dcn_list), 'dcn')
        print(sum(res_list)/len(res_list), 'res')
        print(sum(last_list)/len(last_list), 'last')
        print(sum(flow_list)/len(flow_list) +
              sum(enc_list)/len(enc_list) + 
              sum(dcn_list)/len(dcn_list) + 
              sum(last_list)/len(last_list), 'total')

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
            model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
            model_state_dict = self.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.load_state_dict(model_state_dict, strict=strict)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class MRCF_simple_v18_nofv(nn.Module):

    def __init__(self, device, mid_channels=16, y_only=False, hr_dcn=True, offset_prop=True, split_ratio=3, spynet_pretrained=None):

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
        self.split_ratio = split_ratio

        # optical flow network for feature alignment
        # self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)
        self.spynet = FNet(in_nc=3)
        # self.spynet.load_state_dict(torch.load(spynet_pretrained))

        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        self.dcn_2 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        self.dcn_3 = DCN_module(self.last_channels, 1, self.dk, self.max_residue_magnitude, repeat=True, pre_offset=self.offset_prop, interpolate='pixelshuffle')

        # feature extractor
        self.encoder_lr = LTE.LTE_simple_lr(mid_channels)

        #### propagation branches
        self.forward_resblocks_0_ = ResidualBlocksWithInputConv(
            (mid_channels * self.split_ratio) // 4, mid_channels, 1)
        self.forward_resblocks_1_ = ResidualBlocksWithInputConv(
            (mid_channels * self.split_ratio) // 4, mid_channels, 1)
        self.forward_resblocks_2_ = ResidualBlocksWithInputConv(
            (mid_channels * self.split_ratio) // 4, mid_channels, 1)
        self.forward_resblocks_3_ = ResidualBlocksWithInputConv(
            self.last_channels, self.last_channels, 1)

        self.forward_resblocks_0 = ResidualBlocksWithInputConv_v2(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_1 = ResidualBlocksWithInputConv_v2(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_2 = ResidualBlocksWithInputConv_v2(
            mid_channels*2, mid_channels, 1)
        self.forward_resblocks_3 = ResidualBlocksWithInputConv_v2(
            self.last_channels*2, self.last_channels, 1)

        # downsample
        self.downsample = PixelUnShufflePack_v2(
            self.last_channels, mid_channels, 4, downsample_kernel=3)

        # upsample
        self.upsample = PixelShufflePack(
            mid_channels, (mid_channels * self.split_ratio) // 4, 2, upsample_kernel=3)
        self.upsample_post = PixelShufflePack(
            (mid_channels * self.split_ratio) // 4, self.last_channels, 4, upsample_kernel=3)

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

    def forward(self, lrs, fvs, warp_size=(1080, 1920)):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        flow_list = []
        enc_list = []
        dcn_list = []
        res_list = []
        last_list = []
        WP_h, WP_w = warp_size
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs[:, :, :, :WP_h // 8, :WP_w // 8])
        end.record()
        torch.cuda.synchronize()
        flow_list.append(start.elapsed_time(end) / 1000)
        
        ### forward-time propagation and upsampling
        outputs = []

        torch.cuda.synchronize()
        start.record()
        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        end.record()
        torch.cuda.synchronize()
        enc_list.append(start.elapsed_time(end) / 1000)
        
        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)

            if i > 0:  # no warping required for the first timestep
                torch.cuda.synchronize()
                start.record()
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                flow_lv0 = self.img_upsample_8x(flow) * 8.

                feat_prop_lv3_0 = feat_prop_lv3
                feat_prop_lv3_0_ = flow_warp(feat_prop_lv3_0, flow_lv0.permute(0, 2, 3, 1))
                feat_prop_lv3_ = self.downsample(feat_prop_lv3_0_)
                feat_prop_lv3 = self.downsample(feat_prop_lv3)
                feat_mix = torch.cat((feat_lv0,
                                      feat_lv1,
                                      feat_lv2), dim=1)
                
                feat_mix = flow_warp(feat_mix, flow_lv3.permute(0, 2, 3, 1))
                
                feat_mix = torch.chunk(feat_mix, 3, dim=1)
                feat_lv0 = feat_mix[0]
                feat_lv1 = feat_mix[1]
                feat_lv2 = feat_mix[2]

                #### v15
                # L0
                feat_temp = torch.cat((feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_lv0), dim=1)
                feat_prop_lv3_a, offset = self.dcn_0(feat_temp,
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_,
                                                     flow_lv3)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_temp, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_0(feat_prop_lv0_, feat_temp)
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1) # 5 -> 4
                feat_lv0 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)[:, :, :WP_h // 4, :WP_w // 4]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)
                
                # L1
                feat_temp = torch.cat((feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_lv1), dim=1)
                feat_prop_lv3_a, offset = self.dcn_1(feat_temp,
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_,
                                                     flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_temp, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_1(feat_prop_lv0_, feat_temp)
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1) # 5 -> 4
                feat_lv1 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)[:, :, :WP_h // 4, :WP_w // 4]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)

                # L2
                feat_temp = torch.cat((feat_prop_lv0[:, :, :WP_h // 4, :WP_w // 4], feat_lv2), dim=1)
                feat_prop_lv3_a, offset = self.dcn_2(feat_temp,
                                                     feat_prop_lv3,
                                                     feat_prop_lv3_, 
                                                     flow_lv3, offset)
                if not self.offset_prop:
                    offset = None
                
                feat_prop_lv0_ = torch.cat([feat_temp, feat_prop_lv3_a], dim=1)
                feat_prop_lv0_ = self.forward_resblocks_2(feat_prop_lv0_, feat_temp)
                feat_prop_lv0_ = torch.chunk(feat_prop_lv0_, 4, dim=1) # 5 -> 4
                feat_lv2 = torch.cat(feat_prop_lv0_[self.split_ratio:4], dim=1)[:, :, :WP_h // 4, :WP_w // 4]
                # feat_prop_lv0_ = torch.cat(feat_prop_lv0_[:self.split_ratio], dim=1)

                # L3
                feat_prop_lv0 = self.lrelu(self.upsample_post(feat_prop_lv0))
                feat_prop_lv3_a, _ = self.dcn_3(feat_prop_lv0[:, :, :WP_h, :WP_w],
                                                feat_prop_lv3_0,
                                                feat_prop_lv3_0_,
                                                flow_lv0, offset)
                feat_prop_lv3 = torch.cat([feat_prop_lv0[:, :, :WP_h, :WP_w], feat_prop_lv3_a], dim=1)

                feat_prop_lv3 = self.forward_resblocks_3(feat_prop_lv3, feat_prop_lv0)
                end.record()
                torch.cuda.synchronize()
                dcn_list.append(start.elapsed_time(end) / 1000)
            else:
                torch.cuda.synchronize()
                start.record()
                #### v15
                # L0
                feat_prop_lv0 = self.forward_resblocks_0_(feat_prop_lv0)
                feat_prop_lv0 = torch.chunk(feat_prop_lv0, 4, dim=1)
                feat_lv0 = torch.cat(feat_prop_lv0[self.split_ratio:4], dim=1)[:, :, :WP_h // 4, :WP_w // 4]
                feat_prop_lv0 = torch.cat(feat_prop_lv0[:self.split_ratio], dim=1)
                
                # L1
                feat_prop_lv0 = self.forward_resblocks_1_(feat_prop_lv0)
                feat_prop_lv0 = torch.chunk(feat_prop_lv0, 4, dim=1)
                feat_lv1 = torch.cat(feat_prop_lv0[self.split_ratio:4], dim=1)[:, :, :WP_h // 4, :WP_w // 4]
                feat_prop_lv0 = torch.cat(feat_prop_lv0[:self.split_ratio], dim=1)

                # L2
                feat_prop_lv0 = self.forward_resblocks_2_(feat_prop_lv0)
                feat_prop_lv0 = torch.chunk(feat_prop_lv0, 4, dim=1)
                feat_lv2 = torch.cat(feat_prop_lv0[self.split_ratio:4], dim=1)[:, :, :WP_h // 4, :WP_w // 4]
                feat_prop_lv0 = torch.cat(feat_prop_lv0[:self.split_ratio], dim=1)

                # L3
                feat_prop_lv0 = self.lrelu(self.upsample_post(feat_prop_lv0))
                feat_prop_lv3 = self.forward_resblocks_3_(feat_prop_lv0)
                end.record()
                torch.cuda.synchronize()
                res_list.append(start.elapsed_time(end) / 1000)
            
            torch.cuda.synchronize()
            start.record()
            out = feat_prop_lv3
            feat_prop_lv3 = feat_prop_lv3[:, :, :WP_h, :WP_w]

            out = self.conv_last(out)
            base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)
            end.record()
            torch.cuda.synchronize()
            last_list.append(start.elapsed_time(end) / 1000)

        print(sum(flow_list)/len(flow_list), 'flow')
        print(sum(enc_list)/len(enc_list), 'enc')
        print(sum(dcn_list)/len(dcn_list), 'dcn')
        print(sum(res_list)/len(res_list), 'res')
        print(sum(last_list)/len(last_list), 'last')
        print(sum(flow_list)/len(flow_list) +
              sum(enc_list)/len(enc_list) + 
              sum(dcn_list)/len(dcn_list) + 
              sum(last_list)/len(last_list), 'total')

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
            model_state_dict_save = {k:v for k,v in torch.load(pretrained, map_location=self.device).items()}
            model_state_dict = self.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.load_state_dict(model_state_dict, strict=strict)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class MRCF_simple_v0(nn.Module):
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

        self.pre_lrs = None
        self.feat_prop_0 = None
        self.feat_prop_1 = None
        self.feat_prop_2 = None
        self.feat_prop_3 = None

        # optical flow network for feature alignment
        # self.spynet = SPyNet(pretrained=spynet_pretrained, device=device)
        self.spynet = FNet(in_nc=3)
        self.spynet.load_state_dict(torch.load(spynet_pretrained))
        
        self.dcn_0 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude)
        self.dcn_1 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
        self.dcn_2 = DCN_module(mid_channels, self.dg_num, self.dk, self.max_residue_magnitude, pre_offset=self.offset_prop, interpolate='none')
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
        t += 1
        if torch.is_tensor(self.pre_lrs):
            lrs = torch.cat((self.pre_lrs[:, -1, :, :, :].unsqueeze(1), lrs), dim=1)
            self.pre_lrs = lrs[:, 1:, :, :, :].clone()
        else:
            self.pre_lrs = lrs.clone()
            lrs = torch.cat((lrs[:, -1, :, :, :].unsqueeze(1), lrs), dim=1)

        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        # flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_backward = None
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, fvs, warp_size=(1080, 1920)):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        flow_list = []
        enc_list = []
        dcn_list = []
        res_list = []
        last_list = []
        WP_h, WP_w = warp_size
        n, t, c, h, w = lrs.size()

        ### compute optical flow
        torch.cuda.synchronize()
        start.record()
        flows_forward, flows_backward = self.compute_flow(lrs[:, :, :, :WP_h // 8, :WP_w // 8])
        end.record()
        torch.cuda.synchronize()
        flow_list.append(start.elapsed_time(end) / 1000)
        
        ### forward-time propagation and upsampling
        outputs = []

        torch.cuda.synchronize()
        start.record()
        feat_prop_lv3 = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        feat_prop_lv3_0 = lrs.new_zeros(n, self.last_channels, h*8, w*8)

        B, N, C, H, W = lrs.size()
        lrs_lv0 = lrs.view(B*N, C, H, W)
        lrs_lv3 = self.img_upsample_8x(lrs_lv0)

        _, _, x_lr_lv0 = self.encoder_lr(lrs_lv0, islr=True)

        lrs_lv3_view = lrs_lv3.view(B, N, C, H*8, W*8)

        B, N, C, H, W = fvs.size()
        x_hr_lv3, _, _ = self.encoder_hr(torch.cat((fvs.view(B*N, C, H, W), fvs.view(B*N, C, H, W)), dim=1), islr=True)

        _, C, H, W = x_lr_lv0.size()
        x_lr_lv0 = x_lr_lv0.contiguous().view(B, N, C, H, W)

        _, C, H, W = x_hr_lv3.size()
        x_hr_lv3 = x_hr_lv3.contiguous().view(B, N, C, H, W)
        end.record()
        torch.cuda.synchronize()
        enc_list.append(start.elapsed_time(end) / 1000)

        for i in range(0, t):
            lr_cur = lrs[:, i, :, :, :]
            x_lr_lv0_cur = x_lr_lv0[:, i, :, :, :]
            x_hr_lv3_cur = x_hr_lv3[:, i, :, :, :]
            feat_prop_lv0 = self.upsample(x_lr_lv0_cur)
            feat_prop_lv0_ = torch.cat([feat_prop_lv0[:, :, :H, :W], x_hr_lv3_cur], dim=1)
            feat_prop_lv0_ = self.conv_tttf(feat_prop_lv0_)

            if i > 0:  # no warping required for the first timestep
                torch.cuda.synchronize()
                start.record()
                flow = flows_forward[:, i - 1, :, :, :]
                
                flow_lv3 = self.img_upsample_2x(flow) * 2.
                if self.hr_dcn:
                    flow_lv0 = self.img_upsample_8x(flow) * 8.

                feat_prop_0 = self.feat_prop_0
                feat_prop_1 = self.feat_prop_1
                feat_prop_2 = self.feat_prop_2
                feat_prop_3 = self.feat_prop_3

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
                feat_prop_3, _ = self.dcn_3(feat_prop_2, feat_prop_3, feat_prop_3_, flow_lv3, offset)
                
                feat_prop_3 = torch.cat([feat_prop_2, feat_prop_3], dim=1)
                feat_prop_3 = self.forward_resblocks_3(feat_prop_3)
                end.record()
                torch.cuda.synchronize()
                dcn_list.append(start.elapsed_time(end) / 1000)
            else:
                torch.cuda.synchronize()
                start.record()
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
                feat_prop_3 = torch.cat([feat_prop_2, feat_prop_lv3], dim=1)
                feat_prop_3 = self.forward_resblocks_3(feat_prop_3)
                end.record()
                torch.cuda.synchronize()
                res_list.append(start.elapsed_time(end) / 1000)
            
            torch.cuda.synchronize()
            start.record()
            feat_prop_lv3 = self.lrelu(self.upsample_post(feat_prop_3))
            out = feat_prop_lv3
            self.feat_prop_0 = feat_prop_0
            self.feat_prop_1 = feat_prop_1
            self.feat_prop_2 = feat_prop_2
            self.feat_prop_3 = feat_prop_3

            out = self.conv_last(out)
            base = self.img_upsample_8x(lr_cur)
            out += base
            outputs.append(out)
            end.record()
            torch.cuda.synchronize()
            last_list.append(start.elapsed_time(end) / 1000)

        print(sum(flow_list)/len(flow_list), 'flow')
        print(sum(enc_list)/len(enc_list), 'enc')
        print(sum(dcn_list)/len(dcn_list), 'dcn')
        print(sum(res_list)/len(res_list), 'res')
        print(sum(last_list)/len(last_list), 'last')
        print(sum(flow_list)/len(flow_list) +
              sum(enc_list)/len(enc_list) + 
              sum(dcn_list)/len(dcn_list) + 
              sum(last_list)/len(last_list), 'total')

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
    
    def clear_states(self):
        self.pre_lrs = None
        self.feat_prop_0 = None
        self.feat_prop_1 = None
        self.feat_prop_2 = None
        self.feat_prop_3 = None