from model.EncodeDecoder_TSM import DownSampleModule, UpSampleModule
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, nc_in, nc_out, nf):
        super().__init__()
        self.downsample_module = DownSampleModule(nc_in, nf, residual = True)
        self.upsample_module = UpSampleModule(nc_out, nf, residual = True)


    def forward(self, inp, guidances=None):
        encoded_features = self.downsample_module(inp)
        c11 = self.upsample_module(encoded_features)
        return c11