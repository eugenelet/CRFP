import torch
import torch.nn as nn
import torch.nn.functional as F

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

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''

        return pixel_unshuffle(input, self.downscale_factor)

class LTE_simple_lr(torch.nn.Module):
    def __init__(self, mid_channels):
        super(LTE_simple_lr, self).__init__()
        
        ### use vgg19 weights to initialize
        self.slice1 = torch.nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x, islr=False):
        x = self.slice1(x)
        # x_lv3 = x
        # x_lv2 = x
        # x_lv1 = x
        return None, None, x

class LTE_simple_hr(torch.nn.Module):
    def __init__(self, mid_channels):
        super(LTE_simple_hr, self).__init__()
        
        ### use vgg19 weights to initialize
        self.slice1 = torch.nn.Sequential(
            nn.Conv2d(6, mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.slice2 = torch.nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.slice3 = torch.nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.conv_lv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_lv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_lv3 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, islr=False):
        if islr:
            x = self.slice1(x)
            x_lv3 = self.lrelu(self.conv_lv3(x))
            x = self.slice2(x)
            x_lv2 = self.lrelu(self.conv_lv2(x))
            x = self.slice3(x)
            x_lv1 = self.lrelu(self.conv_lv1(x))
        else:
            x_lv3 = x
            x = self.slice2(x)
            x_lv2 = x
            x = self.slice3(x)
            x_lv1 = x
        return x_lv1, x_lv2, x_lv3

class LTE_simple_hr_single(torch.nn.Module):
    def __init__(self, mid_channels):
        super(LTE_simple_hr_single, self).__init__()
        
        ### use vgg19 weights to initialize
        self.slice1 = torch.nn.Sequential(
            nn.Conv2d(6, mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x, islr=False):
        x = self.slice1(x)
        # x_lv3 = x
        # x_lv2 = x
        # x_lv1 = x
        return None, None, x
        
class LTE_simple_hr_ps(torch.nn.Module):
    def __init__(self, mid_channels):
        super(LTE_simple_hr_ps, self).__init__()
        
        ### use vgg19 weights to initialize
        self.slice1 = torch.nn.Sequential(
            nn.Conv2d(6, mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.slice2 = torch.nn.Sequential(
            PixelUnshuffle(4),
            nn.Conv2d(mid_channels*16, mid_channels*4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels*4, mid_channels*4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.slice3 = torch.nn.Sequential(
            nn.Conv2d(mid_channels*4, mid_channels*4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels*4, mid_channels*4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.slice4 = torch.nn.Sequential(
            nn.Conv2d(mid_channels*4, mid_channels*4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels*4, mid_channels*4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.conv_lv0 = nn.Conv2d(mid_channels*4, mid_channels*4, 3, 1, 1)
        self.conv_lv1 = nn.Conv2d(mid_channels*4, mid_channels*4, 3, 1, 1)
        self.conv_lv2 = nn.Conv2d(mid_channels*4, mid_channels*4, 3, 1, 1)
        self.conv_lv3 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.slice1(x)
        x_lv3 = self.lrelu(self.conv_lv3(x))
        x = self.slice2(x)
        x_lv2 = self.lrelu(self.conv_lv2(x))
        x = self.slice3(x)
        x_lv1 = self.lrelu(self.conv_lv1(x))
        x = self.slice4(x)
        x_lv0 = self.lrelu(self.conv_lv0(x))

        return x_lv0, x_lv1, x_lv2, x_lv3

class LTE_simple_hr_v1(torch.nn.Module):
    def __init__(self, mid_channels):
        super(LTE_simple_hr_v1, self).__init__()
        
        ### use vgg19 weights to initialize
        self.slice1 = torch.nn.Sequential(
            nn.Conv2d(6, mid_channels//4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels//4, mid_channels//4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.slice2 = torch.nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(mid_channels//4, mid_channels//2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels//2, mid_channels//2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.slice3 = torch.nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(mid_channels//2, mid_channels//1, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels//1, mid_channels//1, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.conv_lv3 = nn.Conv2d(mid_channels//4, mid_channels//4, 3, 1, 1)
        self.conv_lv2 = nn.Conv2d(mid_channels//2, mid_channels//2, 3, 1, 1)
        self.conv_lv1 = nn.Conv2d(mid_channels//1, mid_channels//1, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, islr=False):
        if islr:
            x = self.slice1(x)
            x_lv3 = self.lrelu(self.conv_lv3(x))
            x = self.slice2(x)
            x_lv2 = self.lrelu(self.conv_lv2(x))
            x = self.slice3(x)
            x_lv1 = self.lrelu(self.conv_lv1(x))
        else:
            x_lv3 = x
            x = self.slice2(x)
            x_lv2 = x
            x = self.slice3(x)
            x_lv1 = x
        return x_lv1, x_lv2, x_lv3

class LTE_simple_hr_x8(torch.nn.Module):
    def __init__(self, mid_channels):
        super(LTE_simple_hr_x8, self).__init__()
        
        ### use vgg19 weights to initialize
        self.slice1 = torch.nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.slice2 = torch.nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.slice3 = torch.nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.slice4 = torch.nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.conv_lv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_lv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_lv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_lv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x, islr=False):
        if islr:
            x = self.slice1(x)
            x_lv3 = self.lrelu(self.conv_lv3(x))
            x = self.slice2(x)
            x_lv2 = self.lrelu(self.conv_lv2(x))
            x = self.slice3(x)
            x_lv1 = self.lrelu(self.conv_lv1(x))
            x = self.slice4(x)
            x_lv0 = self.lrelu(self.conv_lv0(x))
        else:
            x_lv3 = x
            x = self.slice2(x)
            x_lv2 = x
            x = self.slice3(x)
            x_lv1 = x
            x = self.slice4(x)
            x_lv0 = x
        return x_lv0, x_lv1, x_lv2, x_lv3
