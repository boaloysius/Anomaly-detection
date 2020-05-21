# Based on https://github.com/avalonstrel/GatedConvolution_pytorch/
import logging
import torch.nn as nn
import torch
import numpy as np
from model.Generator import Generator


class AVID(nn.Module):
    def __init__(self, opts=[], nc_in=5, nc_out=3, d_s_args={}, d_t_args={}):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.generator = Generator(5, 3, 64)

        #################
        # Discriminator #
        #################
        '''
        if 'spatial_discriminator' not in opts or opts['spatial_discriminator']:
            self.spatial_discriminator = SNTemporalPatchGANDiscriminator(
                nc_in=5, conv_type='2d', **self.d_s_args
            )
        if 'temporal_discriminator' not in opts or opts['temporal_discriminator']:
            self.temporal_discriminator = SNTemporalPatchGANDiscriminator(
                nc_in=5, **self.d_t_args
            )
        '''

    def forward(self, imgs, model='G'):
        if model == 'G':
            output = self.generator(imgs)
            return output
        '''
        elif model == 'D_t':
            guidances = torch.full_like(masks, 0.) if guidances is None else guidances
            input_imgs = torch.cat([imgs, masks, guidances], dim=2)
            output = self.temporal_discriminator(input_imgs)
        elif model == 'D_s':
            guidances = torch.full_like(masks, 0.) if guidances is None else guidances
            input_imgs = torch.cat([imgs, masks, guidances], dim=2)
            # merge temporal dimension to batch dimension
            in_shape = list(input_imgs.shape)
            input_imgs = input_imgs.view([in_shape[0] * in_shape[1]] + in_shape[2:])
            output = self.spatial_discriminator(input_imgs)
            # split batch and temporal dimension
            output = output.view(in_shape[0], in_shape[1], -1)
        else:
            raise ValueError(f'forwarding model should be "G", "D_t", or "D_s", but got {model}')
        return output
        '''