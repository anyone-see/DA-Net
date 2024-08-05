import torch.nn as nn

from networks.builder_networks import ModelBuilder as ModelBuilderNetworks
from networks.models.BaseLineAudio import BaseLineAudio
from networks.models.BaseLineAudioMyFusion import BaseLineAudioMyFusion

network_builder = ModelBuilderNetworks()


class ModelBuilder:
    def __init__(self):
        self._builders = {
            'BaseLineAudio': self._build_baseline_audio,
            'BaseLineAudioCatFusion': self._build_baseline_audio_cat_fusion,
            'BaseLineAudioCVRFusion': self._build_baseline_audio_cvr_fusion,
            'BaseLine_Audio_My_Fusion': self._build_baseline_audio_my_fusion,
        }

    def _build_baseline(self, weights='', visual_encoder_type='tiny', max_depth=0, pretrained='',
                        decoder_channels=[1024, 512, 256, 256],
                        bins_channels=256,
                        bins_drop=0., ):
        in_channels = self.get_in_channels(visual_encoder_type)
        if len(weights) > 0:
            weight_img = f'{weights}/visual_model.pth'
            weight_bins = f'{weights}/bins_pred_model.pth'
            weight_decoder = f'{weights}/decoder_model.pth'
            weight_head = f'{weights}/predict_head_model.pth'
        else:
            weight_img = weight_bins = weight_decoder = weight_head = ''
        img_config = {'name': 'SwinTransformer', 'weights': weight_img, 'in_chans': 3, 'type': visual_encoder_type,
                      'pretrained': pretrained}
        bins_pred_config = {'name': 'BinsPredictModule', 'weights': weight_bins, 'max_depth': max_depth, 'min_depth': 0,
                            'in_channel': in_channels[3], 'out_channel': bins_channels,
                            'hidden_channel': in_channels[3] * 2,
                            'act_layer': nn.GELU, 'drop': bins_drop}
        decoder_config = {'name': 'DecoderFVSkipModule', 'weights': weight_decoder, 'in_channels': in_channels,
                          'out_channels': decoder_channels}
        predict_head_config = {'name': 'DepthPredHead', 'weights': weight_head, 'bins_channels': bins_channels,
                               'decoder_channels': decoder_channels[-1]}
        img_encoder = network_builder.build(img_config)
        bins_pred = network_builder.build(bins_pred_config)
        decoder = network_builder.build(decoder_config)
        predict_head = network_builder.build(predict_head_config)

        return BaseLine({'rgb_encoder': img_encoder, 'bins_pred': bins_pred, 'decoder': decoder,
                         'predict_head': predict_head})

    def _build_baseline_audio(self, weights='', visual_encoder_type='tiny', max_depth=0, pretrained='',
                              audio_shape=[2, 257, 166],
                              decoder_channels=[1024, 512, 256, 256],
                              bins_channels=256,
                              bins_drop=0.,
                              input_size=128):
        in_channels = self.get_in_channels(visual_encoder_type)
        if len(weights) > 0:
            weight_audio = f'{weights}/audio_model.pth'
            weight_img = f'{weights}/visual_model.pth'
            weight_bins = f'{weights}/bins_pred_model.pth'
            weight_decoder = f'{weights}/decoder_model.pth'
            weight_head = f'{weights}/predict_head_model.pth'
        else:
            weight_audio = weight_img = weight_bins = weight_decoder = weight_head = ''
        repeat = input_size // 4 // 2 // 2 // 2
        audio_config = {'name': 'EchoNet', 'weights': weight_audio, 'audio_shape': audio_shape,
                        'audio_feature_length': in_channels[3], 'repeat': repeat}
        img_config = {'name': 'SwinTransformer', 'weights': weight_img, 'in_chans': 3, 'type': visual_encoder_type,
                      'pretrained': pretrained}
        bins_pred_config = {'name': 'BinsPredictModule', 'weights': weight_bins, 'max_depth': max_depth, 'min_depth': 0,
                            'in_channel': in_channels[3], 'out_channel': bins_channels,
                            'hidden_channel': in_channels[3] * 2,
                            'act_layer': nn.GELU, 'drop': bins_drop}
        decoder_config = {'name': 'DecoderFVSkipModule', 'weights': weight_decoder, 'in_channels': in_channels,
                          'out_channels': decoder_channels}
        predict_head_config = {'name': 'DepthPredHead', 'weights': weight_head, 'bins_channels': bins_channels,
                               'decoder_channels': decoder_channels[-1]}
        audio_encoder = network_builder.build(audio_config)
        img_encoder = network_builder.build(img_config)
        bins_pred = network_builder.build(bins_pred_config)
        decoder = network_builder.build(decoder_config)
        predict_head = network_builder.build(predict_head_config)

        return BaseLineAudio({'rgb_encoder': img_encoder, 'audio_encoder': audio_encoder,
                              'bins_pred': bins_pred, 'decoder': decoder,
                              'predict_head': predict_head})

    def _build_baseline_audio_cat_fusion(self, weights='', visual_encoder_type='tiny', max_depth=0, pretrained='',
                                         audio_shape=[2, 257, 166],
                                         decoder_channels=[1024, 512, 256, 256],
                                         bins_channels=256,
                                         bins_drop=0.,
                                         input_size=128):
        in_channels = self.get_in_channels(visual_encoder_type)
        if len(weights) > 0:
            weight_audio = f'{weights}/audio_model.pth'
            weight_img = f'{weights}/visual_model.pth'
            weight_bins = f'{weights}/bins_pred_model.pth'
            weight_fusion = f'{weights}/fusion_model.pth'
            weight_decoder = f'{weights}/decoder_model.pth'
            weight_head = f'{weights}/predict_head_model.pth'
        else:
            weight_audio = weight_img = weight_bins = weight_fusion = weight_decoder = weight_head = ''
        repeat = input_size // 4 // 2 // 2 // 2

        audio_config = {'name': 'EchoNet', 'weights': weight_audio, 'audio_shape': audio_shape,
                        'audio_feature_length': in_channels[3], 'repeat': repeat}
        img_config = {'name': 'SwinTransformer', 'weights': weight_img, 'in_chans': 3, 'type': visual_encoder_type,
                      'pretrained': pretrained}
        bins_pred_config = {'name': 'BinsPredictModule', 'weights': weight_bins, 'max_depth': max_depth, 'min_depth': 0,
                            'in_channel': in_channels[3], 'out_channel': bins_channels,
                            'hidden_channel': in_channels[3] * 2,
                            'act_layer': nn.GELU, 'drop': bins_drop}
        modal_fusion_config = {'name': 'CatFusion', 'weights': weight_fusion, 'in_channels': in_channels[3] * 2,
                               'out_channels':in_channels[3]}
        decoder_config = {'name': 'DecoderFVSkipModule', 'weights': weight_decoder, 'in_channels': in_channels,
                          'out_channels': decoder_channels}
        predict_head_config = {'name': 'DepthPredHead', 'weights': weight_head, 'bins_channels': bins_channels,
                               'decoder_channels': decoder_channels[-1]}
        audio_encoder = network_builder.build(audio_config)
        img_encoder = network_builder.build(img_config)
        bins_pred = network_builder.build(bins_pred_config)
        modal_fusion = network_builder.build(modal_fusion_config)
        decoder = network_builder.build(decoder_config)
        predict_head = network_builder.build(predict_head_config)
        return BaseLineAudioFusion({'rgb_encoder': img_encoder, 'audio_encoder': audio_encoder,
                                    'bins_pred': bins_pred, 'decoder': decoder, 'modal_fusion': modal_fusion,
                                    'predict_head': predict_head})

    def _build_baseline_audio_cvr_fusion(self, weights='', visual_encoder_type='tiny', max_depth=0, pretrained='',
                                         audio_shape=[2, 257, 166],
                                         decoder_channels=[1024, 512, 256, 256],
                                         bins_channels=256,
                                         bins_drop=0.,
                                         input_size=128):
        in_channels = self.get_in_channels(visual_encoder_type)
        if len(weights) > 0:
            weight_audio = f'{weights}/audio_model.pth'
            weight_img = f'{weights}/visual_model.pth'
            weight_bins = f'{weights}/bins_pred_model.pth'
            weight_fusion = f'{weights}/fusion_model.pth'
            weight_decoder = f'{weights}/decoder_model.pth'
            weight_head = f'{weights}/predict_head_model.pth'
        else:
            weight_audio = weight_img = weight_bins = weight_fusion = weight_decoder = weight_head = ''
        repeat = input_size // 4 // 2 // 2 // 2

        audio_config = {'name': 'EchoNet', 'weights': weight_audio, 'audio_shape': audio_shape,
                        'audio_feature_length': in_channels[3], 'repeat': repeat}
        img_config = {'name': 'SwinTransformer', 'weights': weight_img, 'in_chans': 3, 'type': visual_encoder_type,
                      'pretrained': pretrained}
        bins_pred_config = {'name': 'BinsPredictModule', 'weights': weight_bins, 'max_depth': max_depth, 'min_depth': 0,
                            'in_channel': in_channels[3], 'out_channel': bins_channels,
                            'hidden_channel': in_channels[3] * 2,
                            'act_layer': nn.GELU, 'drop': bins_drop}
        modal_fusion_config = {'name': 'CrossModalVolumeRefinement', 'weights': weight_fusion,
                               'in_channels': in_channels[3], 'out_channels': in_channels[3]}
        decoder_config = {'name': 'DecoderFVSkipModule', 'weights': weight_decoder, 'in_channels': in_channels,
                          'out_channels': decoder_channels}
        predict_head_config = {'name': 'DepthPredHead', 'weights': weight_head, 'bins_channels': bins_channels,
                               'decoder_channels': decoder_channels[-1]}
        audio_encoder = network_builder.build(audio_config)
        img_encoder = network_builder.build(img_config)
        bins_pred = network_builder.build(bins_pred_config)
        modal_fusion = network_builder.build(modal_fusion_config)
        decoder = network_builder.build(decoder_config)
        predict_head = network_builder.build(predict_head_config)
        return BaseLineAudioFusion({'rgb_encoder': img_encoder, 'audio_encoder': audio_encoder,
                                    'bins_pred': bins_pred, 'decoder': decoder, 'modal_fusion': modal_fusion,
                                    'predict_head': predict_head})

    def _build_baseline_audio_billinear_fusion(self, weights='', visual_encoder_type='tiny', max_depth=0, pretrained='',
                                               audio_shape=[2, 257, 166],
                                               decoder_channels=[1024, 512, 256, 256],
                                               bins_channels=256,
                                               bins_drop=0.,
                                               input_size=128):
        in_channels = self.get_in_channels(visual_encoder_type)
        if len(weights) > 0:
            weight_audio = f'{weights}/audio_model.pth'
            weight_img = f'{weights}/visual_model.pth'
            weight_bins = f'{weights}/bins_pred_model.pth'
            weight_fusion = f'{weights}/fusion_model.pth'
            weight_decoder = f'{weights}/decoder_model.pth'
            weight_head = f'{weights}/predict_head_model.pth'
        else:
            weight_audio = weight_img = weight_bins = weight_fusion = weight_decoder = weight_head = ''
        repeat = input_size // 4 // 2 // 2 // 2
        audio_config = {'name': 'EchoNet', 'weights': weight_audio, 'audio_shape': audio_shape,
                        'audio_feature_length': in_channels[3], 'repeat': repeat}
        img_config = {'name': 'SwinTransformer', 'weights': weight_img, 'in_chans': 3, 'type': visual_encoder_type,
                      'pretrained': pretrained}
        bins_pred_config = {'name': 'BinsPredictModule', 'weights': weight_bins, 'max_depth': max_depth, 'min_depth': 0,
                            'in_channel': in_channels[3], 'out_channel': bins_channels,
                            'hidden_channel': in_channels[3] * 2,
                            'act_layer': nn.GELU, 'drop': bins_drop}
        modal_fusion_config = {'name': 'BilinearFusion', 'weights': weight_fusion, 'in_channels': in_channels[3]}
        decoder_config = {'name': 'DecoderFVSkipModule', 'weights': weight_decoder, 'in_channels': in_channels,
                          'out_channels': decoder_channels}
        predict_head_config = {'name': 'DepthPredHead', 'weights': weight_head, 'bins_channels': bins_channels,
                               'decoder_channels': decoder_channels[-1]}
        audio_encoder = network_builder.build(audio_config)
        img_encoder = network_builder.build(img_config)
        bins_pred = network_builder.build(bins_pred_config)
        modal_fusion = network_builder.build(modal_fusion_config)
        decoder = network_builder.build(decoder_config)
        predict_head = network_builder.build(predict_head_config)
        return BaseLineAudioFusion({'rgb_encoder': img_encoder, 'audio_encoder': audio_encoder,
                                    'bins_pred': bins_pred, 'decoder': decoder, 'modal_fusion': modal_fusion,
                                    'predict_head': predict_head})

    def _build_baseline_audio_my_fusion(self, weights='', visual_encoder_type='tiny', max_depth=0, pretrained='',
                                        fusion_num_blocks=3, fusion_num_heads=32,
                                        decoder_channels=[1024, 512, 256, 256],
                                        bins_channels=256,
                                        audio_shape=[2, 257, 166],
                                        bins_drop=0.,
                                        modal_fusion_ffn_drop=0.,
                                        modal_fusion_att_drop=0.,
                                        fusion_type='CMAFM',
                                        input_size=128,
                                        ):
        in_channels = self.get_in_channels(visual_encoder_type)
        if len(weights) > 0:
            weight_audio = f'{weights}/audio_model.pth'
            weight_img = f'{weights}/visual_model.pth'
            weight_fusion = f'{weights}/fusion_model.pth'
            weight_bins = f'{weights}/bins_pred_model.pth'
            weight_decoder = f'{weights}/decoder_model.pth'
            weight_head = f'{weights}/predict_head_model.pth'
        else:
            weight_audio = weight_img = weight_fusion = weight_bins = weight_decoder = weight_head = ''
        repeat = input_size // 4 // 2 // 2 // 2
        audio_config = {'name': 'EchoNet', 'weights': weight_audio, 'audio_shape': audio_shape,
                        'audio_feature_length': in_channels[3], 'repeat': repeat}
        img_config = {'name': 'SwinTransformer', 'weights': weight_img, 'in_chans': 3, 'type': visual_encoder_type,
                      'pretrained': pretrained}
        modal_fusion_config = {'name': 'ModalFusionModel', 'weights': weight_fusion, 'num_blocks': fusion_num_blocks,
                               'input_channels': in_channels[3], 'emb_size': in_channels[3],
                               'num_heads': fusion_num_heads, 'input_size': repeat, 'patch_size': 1,
                               'mlp_channels': in_channels[3] * 2, 'dim_head': in_channels[3] // fusion_num_heads,
                               'att_drop': modal_fusion_att_drop, 'ffn_drop': modal_fusion_ffn_drop,
                               'fusion_type': fusion_type}
        bins_pred_config = {'name': 'BinsPredictModule', 'weights': weight_bins, 'max_depth': max_depth, 'min_depth': 0,
                            'in_channel': in_channels[3], 'out_channel': bins_channels,
                            'hidden_channel': in_channels[3] * 2,
                            'act_layer': nn.GELU, 'drop': bins_drop}
        decoder_config = {'name': 'DecoderFVSkipModule', 'weights': weight_decoder, 'in_channels': in_channels,
                          'out_channels': decoder_channels}
        predict_head_config = {'name': 'DepthPredHead', 'weights': weight_head, 'bins_channels': bins_channels,
                               'decoder_channels': decoder_channels[-1]}

        audio_encoder = network_builder.build(audio_config)
        img_encoder = network_builder.build(img_config)
        modal_fusion = network_builder.build(modal_fusion_config)
        bins_pred = network_builder.build(bins_pred_config)
        decoder = network_builder.build(decoder_config)
        predict_head = network_builder.build(predict_head_config)

        return BaseLineAudioMyFusion({'rgb_encoder': img_encoder, 'audio_encoder': audio_encoder,
                                      'modal_fusion': modal_fusion, 'bins_pred': bins_pred, 'decoder': decoder,
                                      'predict_head': predict_head})

    def get_in_channels(self, visual_encoder_type):
        if visual_encoder_type == 'base':
            in_channels = [128, 256, 512, 1024]
        elif visual_encoder_type == 'large':
            in_channels = [192, 384, 768, 1536]
        elif visual_encoder_type == 'small':
            in_channels = [96, 192, 384, 768]
        elif visual_encoder_type == 'tiny':
            in_channels = [96, 192, 384, 768]
        else:
            raise ValueError(f"Unknown type: {visual_encoder_type}")
        return in_channels

    def build(self, config):
        model_name = config.get('name')
        if model_name in self._builders:
            # Using dictionary unpacking to pass parameters to builders
            return self._builders[model_name](**{k: v for k, v in config.items() if k != 'name'})
        else:
            raise ValueError(f"Unknown model name: {model_name}")
