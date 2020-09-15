import torch.nn as nn
import torch.nn.functional as F

class DownSampleModule(nn.Module):
    def __init__(self, nc_in, nf):
        super().__init__()

        #Input 3-64
        self.conv1 = nn.Conv3d(nc_in, nf * 1, kernel_size=(3, 3, 3), stride=1,padding=1)
        self.conv2 = nn.Conv3d(nf * 1, nf * 1, kernel_size=(3, 3, 3), stride=1,padding=1)

        # Downsample 64-128
        self.conv3 = nn.Conv3d(nf * 1, nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv3d(nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1)
        
        # Downsample 128-256
        self.conv5 = nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=1, padding=1)

        # Downsample 256-512
        self.conv7 = nn.Conv3d(nf * 4, nf * 8, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv8 = nn.Conv3d(nf * 8, nf * 8, kernel_size=(3, 3, 3), stride=1, padding=1)

        # Downsample 512-1024
        self.conv9 = nn.Conv3d(nf * 8, nf * 16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv10 = nn.Conv3d(nf * 16, nf * 16, kernel_size=(3, 3, 3), stride=1, padding=1)

        # Downsample 1024-2048
        self.conv11 = nn.Conv3d(nf * 16, nf * 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv12 = nn.Conv3d(nf * 32, nf * 32, kernel_size=(3, 3, 3), stride=1, padding=1)

    def interpolate(self, c, sf):
        return F.interpolate(c, scale_factor=(1, sf, sf))

    def forward(self, inp):
        out = inp
        out = nn.ReLU()(self.conv1(out))
        out = nn.ReLU()(self.conv2(out))
        out = nn.ReLU()(self.conv3(self.interpolate(out,1/2)))
        out = nn.ReLU()(self.conv4(out))
        #'''
        out = nn.ReLU()(self.conv5(self.interpolate(out,1/2)))
        out = nn.ReLU()(self.conv6(out))
        out = nn.ReLU()(self.conv7(self.interpolate(out,1/2)))
        out = nn.ReLU()(self.conv8(out))
        out = nn.ReLU()(self.conv9(self.interpolate(out,1/2)))
        out = nn.ReLU()(self.conv10(out))
        out = nn.ReLU()(self.conv11(self.interpolate(out,1/2)))
        out = nn.ReLU()(self.conv12(out))
        #'''
        return out


class UpSampleModule(nn.Module):
    def __init__(self, nc_out, nf):
        super().__init__()
        # Upsample 2048-1024
        self.conv1 = nn.Conv3d(nf*32, nf*16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(nf*16, nf*16, kernel_size=(3, 3, 3), stride=1, padding=1)
        # Upsample 1024-512
        self.conv3 = nn.Conv3d(nf*16, nf*8, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv3d(nf*8, nf*8, kernel_size=(3, 3, 3), stride=1 , padding=1)
        # Upsample 512-256
        self.conv5 = nn.Conv3d(nf*8, nf*4, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv6  = nn.Conv3d(nf*4, nf*4, kernel_size=(3, 3, 3), stride=1, padding=1)
        # Upsample 256-128 
        self.conv7 = nn.Conv3d(nf*4, nf*2, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv8 = nn.Conv3d(nf*2, nf*2, kernel_size=(3, 3, 3), stride=1 , padding=1)
        # Upsample 128-64
        self.conv9 = nn.Conv3d(nf*2, nf*1, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv10 = nn.Conv3d(nf*1, nf*1, kernel_size=(3, 3, 3), stride=1 , padding=1)
        # Output
        self.conv11 = nn.Conv3d(nf*1, nf*1, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv12 = nn.Conv3d(nf*1, nc_out, kernel_size=(3, 3, 3), stride=1, padding=1)

    def interpolate(self, c, sf):
        return F.interpolate(c, scale_factor=(1, sf, sf))

    def forward(self, inp):
        out = inp
        #'''
        out = nn.ReLU()(self.conv1(self.interpolate(out,2)))
        out = nn.ReLU()(self.conv2(out))
        out = nn.ReLU()(self.conv3(self.interpolate(out,2)))
        out = nn.ReLU()(self.conv4(out))
        out = nn.ReLU()(self.conv5(self.interpolate(out,2)))
        out = nn.ReLU()(self.conv6(out))
        out = nn.ReLU()(self.conv7(self.interpolate(out,2)))
        out = nn.ReLU()(self.conv8(out))
        #'''
        out = nn.ReLU()(self.conv9(self.interpolate(out,2)))
        out = nn.ReLU()(self.conv10(out))
        out = nn.ReLU()(self.conv11(out))
        out = nn.ReLU()(self.conv12(out))
        return out