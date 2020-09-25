from model.UNet_2D import DownSampleModule_112, UpSampleModule_112
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, nc_in, nc_out, nf):
        super().__init__()
        self.downsample_module = DownSampleModule_112(nc_in, nf)
        self.upsample_module = UpSampleModule_112(nc_out, nf)


    def forward(self, inp, guidances=None):
        encoded_features = self.downsample_module(inp)
        c11 = self.upsample_module(encoded_features)
        return c11