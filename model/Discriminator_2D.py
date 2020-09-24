from model.blocks import NN3Dby2D, NN3Dby2DTSM
import torch.nn as nn
import torch.nn.functional as F

class TemporalDiscriminator(nn.Module):
    def __init__(self, nc_in):
        super().__init__()

        # (224, 224) => (112, 112)
        self.conv1 = NN3Dby2D(nc_in, 64, kernel_size=(3, 3), stride=2,padding=1)

        # (112, 112) => (56, 56)
        self.conv2 = NN3Dby2D(64, 64, kernel_size=(1, 1))
        self.conv3 = NN3Dby2D(64, 64, kernel_size=(1, 1))
        self.conv4 = NN3Dby2D(64, 128, kernel_size=(3, 3), stride=2,padding=1)
        
        # (56, 56) => (26, 26)
        self.conv5 = NN3Dby2D(128, 128, kernel_size=(1, 1))
        self.conv6 = NN3Dby2D(128, 128, kernel_size=(1, 1))
        self.conv7 = NN3Dby2D(128, 256, kernel_size=(5, 5), stride=2)

        # (26, 26) => (11, 11)
        self.conv8 = NN3Dby2D(256, 256, kernel_size=(1, 1))
        self.conv9 = NN3Dby2D(256, 256, kernel_size=(1, 1))
        self.conv10 = NN3Dby2D(256, 512, kernel_size=(5, 5), stride=2)

        #　(11, 11) => (11, 11)
        self.conv11 = NN3Dby2D(512, 1024, kernel_size=(1, 1))
        self.conv12 = NN3Dby2D(1024, 1024, kernel_size=(1, 1))

        self.out_conv = NN3Dby2D(1024, 1, kernel_size=(1, 1), activation=None, bn=False)
        self.out = nn.Sigmoid()

    def forward(self, inp):
        out = inp
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)
        out = self.conv12(out)
        out = self.out_conv(out)
        out = self.out(out)
        return out

class TemporalDiscriminator_mini(nn.Module):
    def __init__(self, nc_in):
        super().__init__()

        # (224, 224) => (112, 112)
        self.conv1 = NN3Dby2D(nc_in, 64, kernel_size=(3, 3), stride=2,padding=1)

        # (112, 112) => (56, 56)
        self.conv2 = NN3Dby2D(64, 64, kernel_size=(1, 1))
        self.conv3 = NN3Dby2D(64, 64, kernel_size=(1, 1))
        self.conv4 = NN3Dby2D(64, 128, kernel_size=(3, 3), stride=2,padding=1)
        
        # (56, 56) => (26, 26)
        self.conv5 = NN3Dby2D(128, 128, kernel_size=(1, 1))
        self.conv6 = NN3Dby2D(128, 128, kernel_size=(1, 1))
        self.conv7 = NN3Dby2D(128, 256, kernel_size=(5, 5), stride=2)

        # (26, 26) => (11, 11)
        self.conv8 = NN3Dby2D(256, 256, kernel_size=(1, 1))
        self.conv9 = NN3Dby2D(256, 256, kernel_size=(1, 1))
        self.conv10 = NN3Dby2D(256, 512, kernel_size=(5, 5), stride=2)

        self.out_conv = NN3Dby2D(512, 1, kernel_size=(1, 1), activation=None, bn=False)
        self.out = nn.Sigmoid()

    def forward(self, inp):
        out = inp
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.out_conv(out)
        out = self.out(out)
        return out
