import torch
import torch.nn as nn

from networks.BinsPredictModule import BinsPredictModule
from networks.CVR import CrossModalVolumeRefinement
from networks.DecoderFVSkipModule import DecoderFVSkipModule
from networks.DepthPredHead import DepthPredHead
from networks.EchoNet import EchoNet
from networks.OthersFusion import CatFusion, DotFusion, BillinearFusion
from networks.SwinTransformer import SwinTransformer
from networks.ModalFusionModule import ModalFusionModule


class ModelBuilder:
    def __init__(self):
        self._builders = {
            'EchoNet': self._build_echo_net,
            'SwinTransformer': self._build_swin_transformer,
            'ModalFusionModel': self._build_modal_fusion_module,
            'BinsPredictModule': self._build_bins_predict_module,
            'DepthPredHead': self._build_depth_pred_head,
            'DecoderFVSkipModule': self._build_decoder_fv_skip_module,

            'CatFusion': self._build_cat_fusion,
            'BilinearFusion': self._build_bilinear_fusion,
            'CrossModalVolumeRefinement': self._build_cross_modal_volume_refinement,
        }

    def _build_cat_fusion(self, weights='',in_channels=976*2, out_channels=976 ):
        net = CatFusion(in_channels,out_channels)
        if len(weights) > 0:
            print('Loading weights for CatFusion')
            net.load_state_dict(torch.load(weights))
        else:
            net.apply(init_weights)
        return net

    def _build_dot_fusion(self, weights=''):
        net = DotFusion()
        if len(weights) > 0:
            print('Loading weights for DotFusion')
            net.load_state_dict(torch.load(weights))
        else:
            net.apply(init_weights)
        return net

    def _build_cross_modal_volume_refinement(self, weights='', in_channels=768, out_channels=768):
        net = CrossModalVolumeRefinement(in_channels, out_channels)
        if len(weights) > 0:
            print('Loading weights for CVR ECCV2022')
            net.load_state_dict(torch.load(weights))
        else:
            net.apply(init_weights)
        return net

    def _build_bilinear_fusion(self, weights='', in_channels=768):
        net = BillinearFusion(in_channels)
        if len(weights) > 0:
            print('Loading weights for BilinearFusion')
            net.load_state_dict(torch.load(weights))
        else:
            net.apply(init_weights)
        return net

    def _build_decoder_fv_skip_module(self, weights='', in_channels=[96, 192, 384, 768],
                                      out_channels=[1024, 512, 256, 128]):
        net = DecoderFVSkipModule(in_channels=in_channels, out_channels=out_channels)
        if len(weights) > 0:
            print('Loading weights for DecoderModule')
            net.load_state_dict(torch.load(weights))
        else:
            net.apply(init_weights)
        return net

    def _build_depth_pred_head(self, weights='', bins_channels=768, decoder_channels=256, scale=1):
        net = DepthPredHead(bins_channels=bins_channels, decoder_channels=decoder_channels, scale=scale)
        if len(weights) > 0:
            print('Loading weights for DepthPredHead')
            net.load_state_dict(torch.load(weights))
        else:
            net.apply(init_weights)
        return net

    def _build_bins_predict_module(self, weights='', max_depth=10, min_depth=0,
                                   in_channel=768, out_channel=256, hidden_channel=768,
                                   act_layer=nn.GELU, drop=0.):
        net = BinsPredictModule(max_depth=max_depth, min_depth=min_depth, in_channel=in_channel,
                                out_channel=out_channel, hidden_channel=hidden_channel, act_layer=act_layer,
                                drop=drop)
        if len(weights) > 0:
            print('Loading weights for BinsPredictModule')
            net.load_state_dict(torch.load(weights))
        else:
            net.apply(init_weights)
        return net

    def _build_modal_fusion_module(self, weights='',
                                   num_blocks=3, input_channels=96,
                                   emb_size=96,
                                   num_heads=4,
                                   input_size=4,
                                   patch_size=1,
                                   mlp_channels=96,
                                   dim_head=96,
                                   att_drop=0.,
                                   ffn_drop=0.,
                                   fusion_type='CMAFM'):
        net = ModalFusionModule(num_blocks=num_blocks, input_channels=input_channels,
                                emb_size=emb_size, num_heads=num_heads, input_size=input_size, dim_head=dim_head,
                                patch_size=patch_size, mlp_channels=mlp_channels, att_drop=att_drop,
                                ffn_drop=ffn_drop, fusion_type=fusion_type)
        if len(weights) > 0:
            print('Loading weights for ModalFusionModule')
            net.load_state_dict(torch.load(weights))
        else:
            net.apply(init_weights)
        return net

    def _build_echo_net(self, weights='', conv1x1_dim=8, audio_shape=[2, 257, 166], audio_feature_length=512, repeat=4):
        net = EchoNet(conv1x1_dim, audio_shape, audio_feature_length, repeat=repeat)
        if len(weights) > 0:
            print('Loading weights for EchoNet')
            net.load_state_dict(torch.load(weights))
        else:
            net.apply(init_weights)
        return net

    def _build_swin_transformer(self, weights='', in_chans=3, type='tiny', pretrained='', device='cuda'):
        if type == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
        elif type == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
        elif type == 'small':
            embed_dim = 96
            depths = [2, 2, 18, 2]
            num_heads = [3, 6, 12, 24]
        elif type == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
        else:
            raise ValueError(f"Unknown type: {type}")
        net = SwinTransformer(in_chans=in_chans, embed_dim=embed_dim, depths=depths, num_heads=num_heads)

        if len(weights) > 0:
            print('Loading weights for SwinTransformer stream')
            net.load_state_dict(torch.load(weights, map_location=device))
        elif len(pretrained) > 0:
            net.init_weights(pretrained)
        else:
            net.apply(init_weights)
        return net

    def build(self, config):
        model_name = config.get('name')
        if model_name in self._builders:
            # Using dictionary unpacking to pass parameters to builders
            return self._builders[model_name](**{k: v for k, v in config.items() if k != 'name'})
        else:
            raise ValueError(f"Unknown model name: {model_name}")


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.normal_(1.0, 0.02)
