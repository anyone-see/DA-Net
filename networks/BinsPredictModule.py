import torch
import torch.nn as nn

import torch.nn.functional as F


class BinsPredictModule(nn.Module):

    def __init__(self, max_depth, min_depth, in_channel=768, out_channel=256, hidden_channel=768,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()

        hidden_channel = hidden_channel or in_channel * 2

        self.max_depth = max_depth
        self.min_depth = min_depth
        self.fc1 = nn.Linear(in_features=in_channel * 2, out_features=hidden_channel)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_features=hidden_channel, out_features=out_channel)
        self.drop = nn.Dropout(drop)

    def forward(self, Fa):
        max_pool_out, _ = torch.max(Fa.flatten(start_dim=2), dim=2)
        avg_pool_out = torch.mean(Fa.flatten(start_dim=2), dim=2)
        pool_out = torch.cat([max_pool_out, avg_pool_out], dim=1)
        x = self.fc1(pool_out)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        bins = torch.softmax(x, dim=1)
        bins = bins / bins.sum(dim=1, keepdim=True)
        bin_widths = (self.max_depth - self.min_depth) * bins
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1).contiguous()

        return centers


if __name__ == '__main__':
    data = torch.randn([2, 768, 4, 4])


    d = BinsPredictModule(max_depth=10, min_depth=0)
    print(d(data).shape)

