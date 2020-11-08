import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class NN3Dby2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation= nn.LeakyReLU(0.1), bn=True):
        super().__init__()

        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #self.layer = nn.utils.spectral_norm(self.layer)
        
        self.bn = nn.BatchNorm2d(out_channels, affine=False) if bn else False        
        self.activation = activation if activation else False

    def forward(self, xs):
        xs = torch.unbind(xs, dim=2) # [B, C, L, H, W]
        # Unbind the video data to a tuple of frames
        
        if(self.bn):
          xs = torch.stack([self.bn(self.layer(x)) for x in xs], dim=2)
        else:
          xs = torch.stack([self.layer(x) for x in xs], dim=2)
        
        if self.activation:
          xs = self.activation(xs)

        return xs


class NN3Dby2DTSM(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()

        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels, affine=False) if bn else False 
        self.activation = activation

        from model.tsm_utils import LearnableTSM
        
        LGTSM_kargs = {
            "shift_ratio": 0.5,
            "shift_groups": 2,
            "shift_width": 1,
            "fixed": True
            }

        self.learnableTSM = LearnableTSM(**LGTSM_kargs)

    def forward(self, xs):
        B, C, L, H, W = xs.shape

        # Learnable temporal shift (via 3x1x1 conv)
        xs_tsm = self.learnableTSM(xs).transpose(1, 2).contiguous()

        out = self.layer(xs_tsm.view(B * L, C, H, W))
        _, C_, H_, W_ = out.shape
        return out.view(B, L, C_, H_, W_).transpose(1, 2)
