import torch
import torch.nn as nn

import torch.nn.functional as F


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class DepthPredHead(nn.Module):

    def __init__(self, bins_channels, decoder_channels, scale=2):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels=decoder_channels, out_channels=bins_channels, kernel_size=3, padding=1)
        self.scale = scale

    def forward(self, x, bin_centers):
        x = self.conv3x3(x)
        x = x.softmax(dim=1)
        x = torch.sum(x * bin_centers, dim=1, keepdim=True)
        if self.scale > 1:
            x = upsample(x, scale_factor=self.scale)
        return x
