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
    with torch.no_grad():
        blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = \
            canny_edge_net(tensor)
    return thresholded


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def edge_loss(self, output, target):

        #print(output.shape, target.shape)
        #view_img([
        #          tanh2sigmoid(output.to("cpu"))[0],
        #          tanh2sigmoid(target.to("cpu"))[0]
        #          ])

        output_edge = get_edge(output)
        gt_edge = get_edge(target)
        loss = self.l1_loss(output_edge, gt_edge)
        return loss

    def forward(self, data_input, model_output):
        targets = data_input
        outputs = model_output

        mean_image_loss = []
        output_edges = []
        target_edges = []
        for batch_idx in range(len(targets)):
            edges_o = []
            edges_t = []
            for frame_idx in range(targets[0].size(0)):
                loss = self.edge_loss(
                    outputs[batch_idx][frame_idx:frame_idx + 1],
                    targets[batch_idx][frame_idx:frame_idx + 1]
                )
                mean_image_loss.append(loss)

        return torch.stack(mean_image_loss, dim=0).mean(dim=0)