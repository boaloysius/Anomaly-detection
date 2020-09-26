from model.blocks import NN3Dby2D, NN3Dby2DTSM
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys

class DownSampleModule_224(nn.Module):
    def __init__(self, nc_in, nf):
        super().__init__()

        #Input 3-64
        self.conv1 = NN3Dby2D(nc_in, nf * 1, kernel_size=(3, 3), stride=1,padding=1)
        self.conv2 = NN3Dby2D(nf * 1, nf * 1, kernel_size=(3, 3), stride=1,padding=1)

        # Downsample 64-128
        self.conv3 = NN3Dby2D(nf * 1, nf * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = NN3Dby2D(nf * 2, nf * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        # Downsample 128-256
        self.conv5 = NN3Dby2D(nf * 2, nf * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv6 = NN3Dby2D(nf * 4, nf * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Downsample 256-512
        self.conv7 = NN3Dby2D(nf * 4, nf * 8, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv8 = NN3Dby2D(nf * 8, nf * 8, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Downsample 512-1024
        self.conv9 = NN3Dby2D(nf * 8, nf * 16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv10 = NN3Dby2D(nf * 16, nf * 16, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Downsample 1024-2048
        self.conv11 = NN3Dby2D(nf * 16, nf * 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv12 = NN3Dby2D(nf * 32, nf * 32, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def interpolate(self, c, sf):
        return F.interpolate(c, scale_factor=(1, sf, sf))

    def forward(self, inp):
        saved = []
        out = inp
        out = self.conv1(out)
        out = self.conv2(out)
        saved.append(out)

        out = self.conv3(self.interpolate(out,1/2))
        out = self.conv4(out)
        saved.append(out)
        #'''
        out = self.conv5(self.interpolate(out,1/2))
        out = self.conv6(out)
        saved.append(out)

        out = self.conv7(self.interpolate(out,1/2))
        out = self.conv8(out)
        saved.append(out)

        out = self.conv9(self.interpolate(out,1/2))
        out = self.conv10(out)
        saved.append(out)

        out = self.conv11(self.interpolate(out,1/2))
        out = self.conv12(out)
        #'''
        return (out, saved)


class UpSampleModule_224(nn.Module):
    def __init__(self, nc_out, nf):
        super().__init__()
        # Upsample 2048-1024
        self.conv1 = NN3Dby2D(nf*48, nf*16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = NN3Dby2D(nf*16, nf*8, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Upsample 1024-512
        self.conv3 = NN3Dby2D(nf*16, nf*8, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = NN3Dby2D(nf*8, nf*4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Upsample 512-256
        self.conv5 = NN3Dby2D(nf*8, nf*4, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6  = NN3Dby2D(nf*4, nf*2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Upsample 256-128 
        self.conv7 = NN3Dby2D(nf*4, nf*2, kernel_size=(3, 3), stride=1, padding=1)
        self.conv8 = NN3Dby2D(nf*2, nf*1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Upsample 128-64
        self.conv9 = NN3Dby2D(nf*2, nf*1, kernel_size=(3, 3), stride=1, padding=1)
        self.conv10 = NN3Dby2D(nf*1, nf*1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Output
        self.conv11 = NN3Dby2D(nf*1, nf*1, kernel_size=(3, 3), stride=1, padding=1)
        self.conv12 = NN3Dby2D(nf*1, nc_out, kernel_size=(3, 3), stride=(1, 1), padding=1, bn=False, activation=nn.Tanh(), dropout=False)

    def interpolate(self, c, sf):
        return F.interpolate(c, scale_factor=(1, sf, sf))
    
    #def upsample()

    def concat(self, x_down, x_up):
      x_up_split = torch.unbind(x_up, dim=2) # [B, C, L, H, W]
      x_down_split = torch.unbind(x_down, dim=2) # [B, C, L, H, W]

      #print(x_up_split[0].shape, x_down_split[0].shape)
      xs = [torch.cat([x_up_split[i], x_down_split[i]], dim=1) for i in range(len(x_up_split))]
      xs = torch.stack([x for x in xs], dim=2)
      return xs

    def forward(self, inp):
        out = inp[0]
        saved = inp[1]
        #'''
        out = self.conv1(self.concat(saved[-1] ,self.interpolate(out,2)))
        out = self.conv2(out)
        out = self.conv3(self.concat(saved[-2],self.interpolate(out,2)))
        out = self.conv4(out)
        out = self.conv5(self.concat(saved[-3],self.interpolate(out,2)))
        out = self.conv6(out)
        out = self.conv7(self.concat(saved[-4],self.interpolate(out,2)))
        out = self.conv8(out)
        #'''
        out = self.conv9(self.concat(saved[-5],self.interpolate(out,2)))
        out = self.conv10(out)
        out = self.conv11(out)
        out = self.conv12(out)
        return out


class DownSampleModule_112(nn.Module):
    def __init__(self, nc_in, nf):
        super().__init__()

        #Input 3-64
        self.conv1 = NN3Dby2D(nc_in, nf * 1, kernel_size=(3, 3), stride=1,padding=1)
        self.conv2 = NN3Dby2D(nf * 1, nf * 1, kernel_size=(3, 3), stride=1,padding=1)

        # Downsample 64-128
        self.conv3 = NN3Dby2D(nf * 1, nf * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = NN3Dby2D(nf * 2, nf * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        # Downsample 128-256
        self.conv5 = NN3Dby2D(nf * 2, nf * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv6 = NN3Dby2D(nf * 4, nf * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Downsample 256-512
        self.conv7 = NN3Dby2D(nf * 4, nf * 8, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv8 = NN3Dby2D(nf * 8, nf * 8, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Downsample 512-1024
        self.conv9 = NN3Dby2D(nf * 8, nf * 16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv10 = NN3Dby2D(nf * 16, nf * 16, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Downsample 1024-2048
        self.conv11 = NN3Dby2D(nf * 16, nf * 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv12 = NN3Dby2D(nf * 32, nf * 32, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def interpolate(self, c, sf):
        return F.interpolate(c, scale_factor=(1, sf, sf))

    def forward(self, inp):
        saved = []
        out = inp
        out = self.conv1(out)
        out = self.conv2(out)
        saved.append(out)

        out = self.conv3(self.interpolate(out,1/2))
        out = self.conv4(out)
        saved.append(out)
        #'''
        out = self.conv5(self.interpolate(out,1/2))
        out = self.conv6(out)
        saved.append(out)

        out = self.conv7(self.interpolate(out,1/2))
        out = self.conv8(out)
        saved.append(out)

        out = self.conv9(self.interpolate(out,1/2))
        out = self.conv10(out)
        #saved.append(out)

        #out = self.conv11(self.interpolate(out,1/2))
        #out = self.conv12(out)
        #'''
        return (out, saved)


class UpSampleModule_112(nn.Module):

    def __init__(self, nc_out, nf):
        super().__init__()
        # Upsample 2048-1024
        self.conv1 = NN3Dby2D(nf*48, nf*16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = NN3Dby2D(nf*16, nf*8, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Upsample 1024-512
        self.conv3 = NN3Dby2D(nf*24, nf*8, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = NN3Dby2D(nf*8, nf*4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Upsample 512-256
        self.conv5 = NN3Dby2D(nf*8, nf*4, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6  = NN3Dby2D(nf*4, nf*2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Upsample 256-128 
        self.conv7 = NN3Dby2D(nf*4, nf*2, kernel_size=(3, 3), stride=1, padding=1)
        self.conv8 = NN3Dby2D(nf*2, nf*1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Upsample 128-64
        self.conv9 = NN3Dby2D(nf*2, nf*1, kernel_size=(3, 3), stride=1, padding=1)
        self.conv10 = NN3Dby2D(nf*1, nf*1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Output
        self.conv11 = NN3Dby2D(nf*1, nf*1, kernel_size=(3, 3), stride=1, padding=1)
        self.conv12 = NN3Dby2D(nf*1, nc_out, kernel_size=(3, 3), stride=(1, 1), padding=1, bn=False, activation=nn.Tanh(), dropout=False)

    def interpolate(self, c, sf):
        return F.interpolate(c, scale_factor=(1, sf, sf))
    
    #def upsample()

    def concat(self, x_down, x_up):
      x_up_split = torch.unbind(x_up, dim=2) # [B, C, L, H, W]
      x_down_split = torch.unbind(x_down, dim=2) # [B, C, L, H, W]

      #print(x_up_split[0].shape, x_down_split[0].shape)
      xs = [torch.cat([x_up_split[i], x_down_split[i]], dim=1) for i in range(len(x_up_split))]
      xs = torch.stack([x for x in xs], dim=2)
      return xs

    def forward(self, inp):
        out = inp[0]
        saved = inp[1]
        #'''
        #out = self.conv1(self.concat(saved[-1] ,self.interpolate(out,2)))
        #out = self.conv2(out)
        out = self.conv3(self.concat(saved[-1],self.interpolate(out,2)))
        out = self.conv4(out)

        out = self.conv5(self.concat(saved[-2],self.interpolate(out,2)))
        out = self.conv6(out)

        out = self.conv7(self.concat(saved[-3],self.interpolate(out,2)))
        out = self.conv8(out)
        #'''
        out = self.conv9(self.concat(saved[-4],self.interpolate(out,2)))
        out = self.conv10(out)
        out = self.conv11(out)
        out = self.conv12(out)
        return out
