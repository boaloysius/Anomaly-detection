from model.blocks import NN3Dby2D, NN3Dby2DTSM
import torch.nn as nn
import torch.nn.functional as F

class DownSampleModule(nn.Module):
    def __init__(self, nc_in, nf, residual=True):
        super().__init__()

        self.residual = residual

        #Input 3-64
        self.conv1 = NN3Dby2D(nc_in, nf * 1, kernel_size=(3, 3), stride=1,padding=1)
        self.conv2 = NN3Dby2D(nf * 1, nf * 1, kernel_size=(3, 3), stride=1,padding=1)

        # Downsample 64-128
        self.conv3 = NN3Dby2D(nf * 1, nf * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv4 = NN3Dby2DTSM(nf * 2, nf * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        # Downsample 128-256
        self.conv5 = NN3Dby2D(nf * 2, nf * 4, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv6 = NN3Dby2DTSM(nf * 4, nf * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Downsample 256-512
        self.conv7 = NN3Dby2D(nf * 4, nf * 8, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv8 = NN3Dby2DTSM(nf * 8, nf * 8, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Downsample 512-1024
        self.conv9 = NN3Dby2D(nf * 8, nf * 16, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv10 = NN3Dby2DTSM(nf * 16, nf * 16, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Downsample 1024-2048
        self.conv11 = NN3Dby2D(nf * 16, nf * 32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv12 = NN3Dby2DTSM(nf * 32, nf * 32, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def interpolate(self, c, sf1=1, sf2=1, sf3=1):
        return F.interpolate(c, scale_factor=(sf1, sf2, sf3))

    def residual_TSM(self, layer, identity):
      x_conv = layer(identity)
      if(self.residual):
        return x_conv + identity
      else:
        return x_conv

    def forward(self, inp):
        out = inp
        out = self.conv1(out)
        out = self.conv2(out)
        
        #out = self.interpolate(out, sf2=1/2, sf3=1/2)
        out = self.conv3(out)
        out = self.residual_TSM(self.conv4, out)
        
        out = self.interpolate(out, sf1=6/8)
        out = self.conv5(out)
        out = self.residual_TSM(self.conv6, out)
        
        out = self.interpolate(out, sf1=4/6)
        out = self.conv7(out)
        out = self.residual_TSM(self.conv8, out)

        out = self.interpolate(out, sf1=3/4)
        out = self.conv9(out)
        out = self.residual_TSM(self.conv10, out)
        
        out = self.interpolate(out, sf1=2/3)
        out = self.conv11(out)
        out = self.residual_TSM(self.conv12, out)
        #'''
        return out


class UpSampleModule(nn.Module):
    def __init__(self, nc_out, nf, residual=True):
        super().__init__()

        self.residual = residual

        # Upsample 2048-1024
        self.upsample1 = NN3Dby2D(nf*32, nf*32, kernel_size=(2, 2), stride=2, padding=0, upsample=True)
        self.conv1 = NN3Dby2D(nf*32, nf*16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = NN3Dby2DTSM(nf*16, nf*16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        # Upsample 1024-512
        self.upsample2 = NN3Dby2D(nf*16, nf*16, kernel_size=(2, 2), stride=2, padding=0, upsample=True)
        self.conv3 = NN3Dby2D(nf*16, nf*8, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = NN3Dby2DTSM(nf*8, nf*8, kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        # Upsample 512-256
        self.upsample3 = NN3Dby2D(nf*8, nf*8, kernel_size=(2, 2), stride=2, padding=0, upsample=True)
        self.conv5 = NN3Dby2D(nf*8, nf*4, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6  = NN3Dby2DTSM(nf*4, nf*4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        # Upsample 256-128 
        self.upsample4 = NN3Dby2D(nf*4, nf*4, kernel_size=(2, 2), stride=2, padding=0, upsample=True)
        self.conv7 = NN3Dby2D(nf*4, nf*2, kernel_size=(3, 3), stride=1, padding=1)
        self.conv8 = NN3Dby2DTSM(nf*2, nf*2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        # Upsample 128-64
        self.upsample5 = NN3Dby2D(nf*2, nf*2, kernel_size=(2, 2), stride=2, padding=0, upsample=True)
        self.conv9 = NN3Dby2D(nf*2, nf*1, kernel_size=(3, 3), stride=1, padding=1)
        self.conv10 = NN3Dby2DTSM(nf*1, nf*1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Output
        self.conv11 = NN3Dby2D(nf*1, nf*1, kernel_size=(3, 3), stride=1, padding=1)
        self.conv12 = NN3Dby2D(nf*1, nc_out, kernel_size=(3, 3), stride=(1, 1), padding=1, bn=False, activation=nn.Tanh())

    def interpolate(self, c, sf1=1, sf2=1, sf3=1):
        return F.interpolate(c, scale_factor=(sf1, sf2, sf3))

    def residual_TSM(self, layer, identity):
      x_conv = layer(identity)
      if(self.residual):
        return x_conv + identity
      else:
        return x_conv

    def forward(self, inp):
        out = inp
        #'''
        #out = self.interpolate(out,sf2=2,sf3=2)
        out = self.upsample1(out)
        out = self.conv1(out)
        out = self.residual_TSM(self.conv2, out)

        out = self.upsample2(out)
        out = self.interpolate(out, sf1=3/2)
        out = self.conv3(out)
        out = self.residual_TSM(self.conv4, out)

        out = self.upsample3(out)
        out = self.interpolate(out, sf1=4/3)
        out = self.conv5(out)
        out = self.residual_TSM(self.conv6, out)

        out = self.upsample4(out)
        out = self.interpolate(out, sf1=6/4)
        out = self.conv7(out)
        out = self.residual_TSM(self.conv8, out)

        out = self.interpolate(out,sf1=8/6)
        out = self.upsample5(out)
        out = self.conv9(out)
        out = self.residual_TSM(self.conv10, out)
        
        out = self.conv11(out)
        out = self.conv12(out)
        return out