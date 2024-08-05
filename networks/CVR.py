import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class CrossModalVolumeRefinement(nn.Module):

    def __init__(self, input_channels, out_channels):
        super().__init__()

        self.query_conv = nn.Sequential(nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(True),
                                        nn.BatchNorm2d(out_channels))

        self.key_conv = nn.Sequential(nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True))
        self.value_conv = nn.Sequential(nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True))
        self.fusion_conv = nn.Sequential(nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True))

    def forward(self, Fv, Fa):
        a_query = self.query_conv(Fa)

        v_key = self.key_conv(Fv)
        v_value = self.value_conv(Fv)

        qk = a_query * v_key
        qk = self.fusion_conv(qk)
        return v_value * qk


if __name__ == '__main__':
    input_v = torch.randn((16, 768, 4, 4))
    input_a = torch.randn((16, 768, 4, 4))
    model = CrossModalVolumeRefinement(768, 768)
    out = model(input_v, input_a)
    print(out.shape)
    torch.save(model.state_dict(),'CVR.pth')
# class Cross_Fusion(nn.Module):
#     def __init__(self, mode):
#         super(Cross_Fusion, self).__init__()
#
#         self.query_conv = nn.Sequential(nn.Conv2d(32, 256, kernel_size=3, stride=1, padding=1),
#                                         nn.ReLU(True),
#                                         nn.BatchNorm2d(256))
#
#         self.key_conv = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
#                                       nn.ReLU(inplace=True))
#
#         self.value_conv = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
#                                         nn.ReLU(inplace=True))
#
#         self.fusion_conv = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
#                                          nn.ReLU(inplace=True))
#
#     def forward(self, stereo_cost, audio_defea):
#         # import pdb; pdb.set_trace()
#         audio_query = self.query_conv(audio_defea)
#         audio_query = audio_query.reshape(stereo_cost.shape)
#
#         stereo_key = self.key_conv(stereo_cost)
#         stereo_value = self.value_conv(stereo_cost)
#
#         fusion_fea = audio_query * stereo_key
#         fusion_res = self.fusion_conv(fusion_fea)
#         stereo_cost = stereo_value * fusion_res
#
#         return stereo_cost
