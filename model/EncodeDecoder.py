from model.blocks import NN3Dby2D, NN3Dby2DTSM, NN3Dby2DTSMDeconv
import torch.nn as nn

class DownSampleModule(nn.Module):
    def __init__(self, nc_in, nf):
        super().__init__()

        #3-64, 64-64
        self.conv1 = NN3Dby2D(nc_in, nf * 1, kernel_size=(3, 3), stride=1,padding=1)
        self.conv2 = NN3Dby2D(nf * 1, nf * 1, kernel_size=(3, 3), stride=1,padding=1)

        # Downsample 64-128, 128-128
        self.conv3 = NN3Dby2DTSM(nf * 1, nf * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = NN3Dby2DTSM(nf * 2, nf * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        # Downsample 128-256,256-256
        self.conv5 = NN3Dby2DTSM(nf * 2, nf * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv6 = NN3Dby2DTSM(nf * 4, nf * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Downsample 256-512
        self.conv7 = NN3Dby2DTSM(nf * 4, nf * 8, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv8 = NN3Dby2DTSM(nf * 8, nf * 8, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Downsample 512-1024
        self.conv9 = NN3Dby2DTSM(nf * 8, nf * 16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv10 = NN3Dby2DTSM(nf * 16, nf * 16, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Downsample 1024-2048
        self.conv11 = NN3Dby2DTSM(nf * 16, nf * 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv12 = NN3Dby2DTSM(nf * 32, nf * 32, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def interpolate(c, sf):
        return F.interpolate(c, scale_factor=(1, sf, sf))

    def forward(self, inp):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)
        c3 = self.conv3(self.interpolate(c2,1/2))
        c4 = self.conv4(c3)
        c5 = self.conv5(self.interpolate(c4,1/2))
        c6 = self.conv6(c5)
        c7 = self.conv7(self.interpolate(c6,1/2))
        c8 = self.conv8(c7)
        c9 = self.conv9(self.interpolate(c8,1/2))
        c10 = self.conv10(c9)
        c11 = self.conv11(self.interpolate(c10,1/2))
        c12 = self.conv11(c11)
        return c12


class UpSampleModule(nn.Module):
    def __init__(self, nc_in, nc_out, nf):
        super().__init__()
        # Upsample 1
        self.deconv1 = NN3Dby2DTSMDeconv(nc_in, nf * 2, kernel_size=(3, 3), stride=1, padding=1)
        self.conv9   = NN3Dby2DTSMDeconv(nf * 2, nf * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Upsample 2
        self.deconv2 = NN3Dby2DTSMDeconv(nf * 2, nf * 1, kernel_size=(3, 3), stride=1, padding=1)
        self.conv10 = NN3Dby2DTSMDeconv(nf * 1, nf // 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv11 = NN3Dby2DTSMDeconv(nf // 2, nc_out, kernel_size=(3, 3), stride=(1, 1), activation=None)


    def forward(self, inp):
        d1 = self.deconv1(inp)
        c9 = self.conv9(d1)
        d2 = self.deconv2(c9)
        c10 = self.conv10(d2)
        c11 = self.conv11(c10)
        return c11