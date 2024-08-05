import torch
import torch.nn as nn

import torch.nn.functional as F


class UpSampleBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.LeakyReLU())

    def forward(self, x):
        up_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self._net(up_x)


class DecoderFVSkipModule(nn.Module):

    def __init__(self, in_channels=[96, 192, 384, 768], out_channels=[1024, 512, 256, 128]):
        super().__init__()
        self.in_channels = in_channels

        self.up1 = UpSampleBN(in_channels=in_channels[3] * 2, out_channels=out_channels[0])
        self.up2 = UpSampleBN(in_channels=out_channels[0] + in_channels[2], out_channels=out_channels[1])
        self.up3 = UpSampleBN(in_channels=out_channels[1] + in_channels[1], out_channels=out_channels[2])
        self.up4 = UpSampleBN(in_channels=out_channels[2] + in_channels[0], out_channels=out_channels[3])
        self.up5 = UpSampleBN(in_channels=out_channels[3], out_channels=out_channels[3])

    def forward(self, Fv, visual_feats):
        Fv = self.up1(torch.cat([Fv, visual_feats[3]], dim=1))
        Fv = self.up2(torch.cat([Fv, visual_feats[2]], dim=1))
        Fv = self.up3(torch.cat([Fv, visual_feats[1]], dim=1))
        Fv = self.up4(torch.cat([Fv, visual_feats[0]], dim=1))
        Fv = self.up5(Fv)
        return Fv
