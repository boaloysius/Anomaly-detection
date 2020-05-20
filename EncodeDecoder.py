from blocks import VanillaConv, VanillaDeConv

class DownSampleModule(nn.Module):
    def __init__(self, nc_in, nf, use_bias, norm, conv_by, conv_type):
        super().__init__()

        self.conv1 = self.VanillaConv(nc_in, nf * 1, kernel_size=(3, 5, 5), stride=1,padding=1, conv_by="2d")

        # Downsample 1
        self.conv2 = self.VanillaConv(nf * 1, nf * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 2, 2), conv_by="2dtsm")
        self.conv3 = self.VanillaConv(nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, conv_by="2dtsm")
        
        # Downsample 2
        self.conv4 = self.VanillaConv(nf * 2, nf * 4, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, conv_by="2dtsm")
        self.conv5 = self.VanillaConv(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, conv_by="2dtsm")
        self.conv6 = self.VanillaConv(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, conv_by="2dtsm")

        # Dilated Convolutions
        self.dilated_conv1 = self.VanillaConv(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=-1, dilation=(1, 2, 2), conv_by="2dtsm")
        self.dilated_conv2 = self.VanillaConv(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=-1, dilation=(1, 4, 4), conv_by="2dtsm")
        self.conv7 = self.VanillaConv(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, conv_by="2dtsm")
        self.conv8 = self.VanillaConv(nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, conv_by="2dtsm")

    def forward(self, inp):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)

        a1 = self.dilated_conv1(c6)
        a2 = self.dilated_conv2(a1)

        c7 = self.conv7(a2)
        c8 = self.conv8(c7)
        return c8


class UpSampleModule(nn.Module):
    def __init__(self, nc_in, nc_out, nf):
        super().__init__()
        # Upsample 1
        self.deconv1 = self.VanillaDeConv(nc_in, nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1, conv_by="2dtsm")
        self.conv9   = self.VanillaConv(nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, conv_by="2dtsm")
        # Upsample 2
        self.deconv2 = self.VanillaDeConv(nf * 2, nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1, conv_by="2dtsm")
        self.conv10 = self.VanillaConv(nf * 1, nf // 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, conv_by="2dtsm")
        self.conv11 = self.VanillaConv(nf // 2, nc_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), conv_by="2dtsm", activation=None)


    def forward(self, inp):
        d1 = self.deconv1(inp)
        c9 = self.conv9(d1)
        d2 = self.deconv2(c9)
        c10 = self.conv10(d2)
        c11 = self.conv11(c10)
        return c11