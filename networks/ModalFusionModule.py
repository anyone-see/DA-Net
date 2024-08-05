import torch
import torch.nn as nn

from networks.CMA import CMA
from networks.DWAF import DWAF


class ModalFusionModule(nn.Module):

    def __init__(self, num_blocks=3, input_channels=96,
                 emb_size=96,
                 num_heads=4,
                 input_size=4,
                 patch_size=4,
                 mlp_channels=96,
                 dim_head=96,
                 att_drop=0.,
                 ffn_drop=0.,
                 fusion_type='DWAF'):
        super().__init__()

        self.num_blocks = num_blocks

        if fusion_type=='DWAF':
            self.audio_modal_fusion = nn.ModuleList([DWAF(input_channels=input_channels,
                                                           emb_size=emb_size,
                                                           num_heads=num_heads,
                                                           input_size=input_size,
                                                           patch_size=patch_size,
                                                           mlp_channels=mlp_channels,
                                                           dim_head=dim_head,
                                                           att_drop=att_drop,
                                                           ffn_drop=ffn_drop) for _ in range(num_blocks)])
            self.visual_modal_fusion = nn.ModuleList([DWAF(input_channels=input_channels,
                                                            emb_size=emb_size,
                                                            num_heads=num_heads,
                                                            input_size=input_size,
                                                            patch_size=patch_size,
                                                            mlp_channels=mlp_channels,
                                                            dim_head=dim_head,
                                                            att_drop=att_drop,
                                                            ffn_drop=ffn_drop) for _ in range(num_blocks)])
        elif fusion_type=='CMA':
            self.audio_modal_fusion = nn.ModuleList([CMA(input_channels=input_channels,
                                                           emb_size=emb_size,
                                                           num_heads=num_heads,
                                                           input_size=input_size,
                                                           patch_size=patch_size,
                                                           mlp_channels=mlp_channels,
                                                           dim_head=dim_head,
                                                           att_drop=att_drop,
                                                           ffn_drop=ffn_drop) for _ in range(num_blocks)])
            self.visual_modal_fusion = nn.ModuleList([CMA(input_channels=input_channels,
                                                            emb_size=emb_size,
                                                            num_heads=num_heads,
                                                            input_size=input_size,
                                                            patch_size=patch_size,
                                                            mlp_channels=mlp_channels,
                                                            dim_head=dim_head,
                                                            att_drop=att_drop,
                                                            ffn_drop=ffn_drop) for _ in range(num_blocks)])
    def forward(self, Fv, Fa):
        for i in range(self.num_blocks):
            temp_ = self.audio_modal_fusion[i](Fa, Fv)
            Fv = self.visual_modal_fusion[i](Fv, Fa)
            Fa = temp_
        return Fv, Fa
