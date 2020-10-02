import os
import sys
import torch
import torch.nn as nn

from utils import *

from libs.CannyEdgePytorch.net_canny import Net as CannyEdgeNet

device = device = 'cuda' if torch.cuda.is_available() else 'cpu'

canny_edge_net = CannyEdgeNet(threshold=2.0, use_cuda=True).to(device)
canny_edge_net.eval()

def get_edge(tensor):

    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = \
            canny_edge_net(tensor)
    return thresholded


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def edge_loss(self, output, target):
        output_edge = get_edge(output)
        gt_edge = get_edge(target)
        loss = self.l1_loss(output_edge, gt_edge)
        return loss

    def forward(self, data_input, model_output):

        #BXL,C,W,H
        data_input = data_input.permute(0, 2, 1, 3,4).contiguous()
        data_input = data_input.view(-1, 1, *(data_input.size()[2:]))
        data_input = tanh2sigmoid(data_input).unbind(0)

        model_output = model_output.permute(0, 2, 1, 3,4).contiguous()
        model_output = model_output.view(-1, 1, *(model_output.size()[2:]))
        model_output = tanh2sigmoid(model_output).unbind(0)

        mean_image_loss = [self.edge_loss(output, target) for output, target in zip(model_output, data_input)]

        return torch.stack(mean_image_loss, dim=0).mean(dim=0)


import libs.pytorch_ssim.pytorch_ssim as pytorch_ssim

def ssim_loss(x, target):
  unbind1 = torch.unbind(tanh2sigmoid(x), dim=2)
  unbind2 = torch.unbind(tanh2sigmoid(target), dim=2)
  ssim_loss = pytorch_ssim.SSIM(window_size = 11)
  mean_loss = [ssim_loss(img1, img2) for img1, img2 in zip(unbind1, unbind2)]
  mean_loss_tensor = torch.stack(mean_loss, dim=0).mean(dim=0)
  return(mean_loss_tensor)
