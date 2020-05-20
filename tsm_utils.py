import torch
import torch.nn.functional as F
from torch import nn


class TemporalShiftConv(nn.Module):
    def __init__(self, weights: list):
        super().__init__()
        shift_width = len(weights)
        weights = torch.tensor(weights).view(1, 1, shift_width, 1, 1)
        conv = nn.Conv3d(1, 1, [shift_width, 1, 1], bias=False, padding=[shift_width // 2, 0, 0])
        conv.weight = nn.Parameter(weights)
        self.conv = conv

    def forward(self, tensor):
        B, C, T, H, W = tensor.shape
        shifted = self.conv(tensor.contiguous().view([B * C, 1, T, H, W])).view([B, C, T, H, W])

        return shifted


class LearnableTSM(nn.Module):
    def __init__(self, shift_ratio=0.5, shift_groups=2, shift_width=3, fixed=False):
        super().__init__()
        self.shift_ratio = shift_ratio
        self.shift_groups = shift_groups
        if shift_groups == 2:  # for backward compability
            self.conv_names = ['pre_shift_conv', 'post_shift_conv']
        else:
            self.conv_names = [f'shift_conv_{i}' for i in range(shift_groups)]

        # shift kernel weights are initialized to behave like normal TSM
        pos = shift_width // 2
        back_shift_w = [0.] * shift_width
        back_shift_w[-pos] = 1.
        forward_shift_w = [0.] * shift_width
        forward_shift_w[pos - 1] = 1.
        for i in range(shift_groups):
            ts_conv = TemporalShiftConv(back_shift_w) if i * 2 < shift_groups else TemporalShiftConv(forward_shift_w)
            setattr(self, self.conv_names[i], ts_conv)

        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, tensor):
        shape = B, C, T, H, W = tensor.shape
        split_size = int(C * self.shift_ratio) // self.shift_groups
        split_sizes = [split_size] * self.shift_groups + [C - split_size * self.shift_groups]
        tensors = tensor.split(split_sizes, dim=1)
        assert len(tensors) == self.shift_groups + 1

        tensors = [
            getattr(self, self.conv_names[i])(tensors[i])
            for i in range(self.shift_groups)
        ] + [tensors[-1]]
        return torch.cat(tensors, dim=1).view(shape)