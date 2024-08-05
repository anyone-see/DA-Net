import torch
import torch.nn as nn


class BaseLineAudioMyFusion(nn.Module):

    def __init__(self, nets):
        super().__init__()
        self.rgb_encoder = nets['rgb_encoder']
        self.audio_encoder = nets['audio_encoder']
        self.modal_fusion = nets['modal_fusion']
        self.bins_pred = nets['bins_pred']
        self.decoder = nets['decoder']
        self.predict_head = nets['predict_head']

    def forward(self, input):
        rgb_input = input['img']
        audio_input = input['audio']
        depth_gt = input['depth']

        audio_feat = self.audio_encoder(audio_input)
        img_feats = self.rgb_encoder(rgb_input)

        Fv, Fa = self.modal_fusion(img_feats[-1], audio_feat[-1])

        bins_centers = self.bins_pred(Fa)
        decoder_out = self.decoder(Fv, img_feats)

        depth_prediction = self.predict_head(decoder_out, bins_centers)

        output = {'depth_predicted': depth_prediction,
                  'img': rgb_input,
                  'audio': audio_input,
                  'depth_gt': depth_gt}
        return output

    def save(self, path):
        torch.save(self.rgb_encoder.state_dict(), path + '/visual_model.pth')
        torch.save(self.audio_encoder.state_dict(), path + '/audio_model.pth')
        torch.save(self.modal_fusion.state_dict(), path + '/fusion_model.pth')
        torch.save(self.bins_pred.state_dict(), path + '/bins_pred_model.pth')
        torch.save(self.decoder.state_dict(), path + '/decoder_model.pth')
        torch.save(self.predict_head.state_dict(), path + '/predict_head_model.pth')

    def get_all_parameters(self, lr):
        return [
            {'params': self.rgb_encoder.parameters(), 'lr': lr},
            {'params': self.audio_encoder.parameters(), 'lr': lr},
            {'params': self.modal_fusion.parameters(), 'lr': lr},
            {'params': self.bins_pred.parameters(), 'lr': lr},
            {'params': self.decoder.parameters(), 'lr': lr},
            {'params': self.predict_head.parameters(), 'lr': lr}
        ]
