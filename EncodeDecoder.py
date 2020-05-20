from blocks import VanillaConv, VanillaDeConv

class DownSampleModule(nn.Module):
    def __init__(self, nc_in, nf):
        super().__init__()

        self.conv1 = self.NN3Dby2D(nc_in, nf * 1, kernel_size=(5, 5), stride=1,padding=1)

        # Downsample 1
        self.conv2 = self.NN3Dby2DTSM(nf * 1, nf * 2, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
        self.conv3 = self.NN3Dby2DTSM(nf * 2, nf * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        # Downsample 2
        self.conv4 = self.NN3Dby2DTSM(nf * 2, nf * 4, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.conv5 = self.NN3Dby2DTSM(nf * 4, nf * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)


    def forward(self, inp):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        return c8


class UpSampleModule(nn.Module):
    def __init__(self, nc_in, nc_out, nf):
        super().__init__()
        # Upsample 1
        self.deconv1 = self.NN3Dby2DTSMDeconv(nc_in, nf * 2, kernel_size=(3, 3), stride=1, padding=1)
        self.conv9   = self.NN3Dby2DTSMDeconv(nf * 2, nf * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # Upsample 2
        self.deconv2 = self.NN3Dby2DTSMDeconv(nf * 2, nf * 1, kernel_size=(3, 3), stride=1, padding=1)
        self.conv10 = self.NN3Dby2DTSMDeconv(nf * 1, nf // 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv11 = self.NN3Dby2DTSMDeconv(nf // 2, nc_out, kernel_size=(3, 3), stride=(1, 1), activation=None)


    def forward(self, inp):
        d1 = self.deconv1(inp)
        c9 = self.conv9(d1)
        d2 = self.deconv2(c9)
        c10 = self.conv10(d2)
        c11 = self.conv11(c10)
        return c11