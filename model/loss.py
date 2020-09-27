import os
import sys
import torch

from libs.CannyEdgePytorch.net_canny import Net as CannyEdgeNet

device = torch.device("cuda")
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
        from utils.edge import get_edge
        output_edge = get_edge(output)
        gt_edge = get_edge(target)
        loss = self.l1_loss(output_edge, gt_edge)
        return loss, output_edge, gt_edge

    def forward(self, data_input, model_output):
        targets = data_input['targets']
        outputs = model_output['outputs']

        mean_image_loss = []
        output_edges = []
        target_edges = []
        for batch_idx in range(targets.size(0)):
            edges_o = []
            edges_t = []
            for frame_idx in range(targets.size(1)):
                loss, output_edge, target_edge = self.edge_loss(
                    outputs[batch_idx, frame_idx:frame_idx + 1],
                    targets[batch_idx, frame_idx:frame_idx + 1]
                )
                mean_image_loss.append(loss)
                edges_o.append(output_edge)
                edges_t.append(target_edge)
            output_edges.append(torch.cat(edges_o, dim=0))
            target_edges.append(torch.cat(edges_t, dim=0))

        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        self.current_output_edges = output_edges
        self.current_target_edges = target_edges
        return mean_image_loss