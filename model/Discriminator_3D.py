from model.blocks import NN3Dby2D, NN3Dby2DTSM
import torch.nn as nn
import torch.nn.functional as F

class TemporalDiscriminator(nn.Module):
    def __init__(self, nc_in):
        super().__init__()

        # (224, 224) => (112, 112)
        self.conv1 = nn.Conv3d(nc_in, 64, kernel_size=(3, 3, 3), stride=2,padding=1)

        # (112, 112) => (56, 56)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(1, 1, 1))
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(1, 1, 1))
        self.conv4 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=2,padding=1)
        
        # (56, 56) => (26, 26)
        self.conv5 = nn.Conv3d(128, 128, kernel_size=(1, 1, 1))
        self.conv6 = nn.Conv3d(128, 128, kernel_size=(1, 1, 1))
        self.conv7 = nn.Conv3d(128, 256, kernel_size=(5, 5, 5), stride=2)

        # (26, 26) => (11, 11)
        self.conv8 = nn.Conv3d(256, 256, kernel_size=(1, 1, 1))
        self.conv9 = nn.Conv3d(256, 256, kernel_size=(1, 1, 1))
        self.conv10 = nn.Conv3d(256, 512, kernel_size=(5, 5, 5), stride=2)

        #　(11, 11) => (11, 11)
        self.conv11 = nn.Conv3d(512, 1024, kernel_size=(1, 1, 1))
        self.conv12 = nn.Conv3d(1024, 1024, kernel_size=(1, 1, 1))

        self.out_conv = nn.Conv3d(1024, 1, kernel_size=(1, 1, 1), activation=None)
        self.out = nn.Sigmoid()

    def forward(self, inp):
        out = inp
        out = nn.utils.spectral_norm(self.conv1(out))
        out = nn.utils.spectral_norm(self.conv2(out))
        out = nn.utils.spectral_norm(self.conv3(out))
        out = nn.utils.spectral_norm(self.conv4(out))
        out = nn.utils.spectral_norm(self.conv5(out))
        out = nn.utils.spectral_norm(self.conv6(out))
        out = nn.utils.spectral_norm(self.conv7(out))
        out = nn.utils.spectral_norm(self.conv8(out))
        out = nn.utils.spectral_norm(self.conv9(out))
        out = nn.utils.spectral_norm(self.conv10(out))
        out = nn.utils.spectral_norm(self.conv11(out))
        out = nn.utils.spectral_norm(self.conv12(out))
        out = nn.utils.spectral_norm(self.out_conv(out))
        out = nn.utils.spectral_norm(self.out(out))
        return out

class SpatialDiscriminator(nn.Module):
    def __init__(self, nc_in):
        super().__init__()

        # (224, 224) => (112, 112)
        self.conv1 = nn.Conv3d(nc_in, 64, kernel_size=(3, 3, 3), stride=2,padding=1)

        # (112, 112) => (56, 56)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(1, 1, 1))
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(1, 1, 1))
        self.conv4 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=2,padding=1)
        
        # (56, 56) => (26, 26)
        self.conv5 = nn.Conv3d(128, 128, kernel_size=(1, 1, 1))
        self.conv6 = nn.Conv3d(128, 128, kernel_size=(1, 1, 1))
        self.conv7 = nn.Conv3d(128, 256, kernel_size=(5, 5, 5), stride=2)

        # (26, 26) => (11, 11)
        self.conv8 = nn.Conv3d(256, 256, kernel_size=(1, 1, 1))
        self.conv9 = nn.Conv3d(256, 256, kernel_size=(1, 1, 1))
        self.conv10 = nn.Conv3d(256, 512, kernel_size=(5, 5, 5), stride=2)

        #　(11, 11) => (11, 11)
        self.conv11 = nn.Conv3d(512, 1024, kernel_size=(1, 1, 1))
        self.conv12 = nn.Conv3d(1024, 1024, kernel_size=(1, 1, 1))

        self.out_conv = nn.Conv3d(1024, 1, kernel_size=(1, 1, 1))
        self.out = nn.Sigmoid()

    def forward(self, inp):
        out = inp
        out = nn.utils.spectral_norm(self.conv1(out))
        out = nn.utils.spectral_norm(self.conv2(out))
        out = nn.utils.spectral_norm(self.conv3(out))
        out = nn.utils.spectral_norm(self.conv4(out))
        out = nn.utils.spectral_norm(self.conv5(out))
        out = nn.utils.spectral_norm(self.conv6(out))
        out = nn.utils.spectral_norm(self.conv7(out))
        out = nn.utils.spectral_norm(self.conv8(out))
        out = nn.utils.spectral_norm(self.conv9(out))
        out = nn.utils.spectral_norm(self.conv10(out))
        out = nn.utils.spectral_norm(self.conv11(out))
        out = nn.utils.spectral_norm(self.conv12(out))
        out = nn.utils.spectral_norm(self.out_conv(out))
        out = nn.utils.spectral_norm(self.out(out))
        return out
