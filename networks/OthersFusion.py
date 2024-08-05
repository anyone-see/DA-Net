import torch
import torch.nn as nn


class CatFusion(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        )

    def forward(self, x1, x2):
        x1 = torch.cat([x1, x2], dim=1)

        x1 = self.block(x1)
        return x1


class DotFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x1 = x1 * x2
        return x1


class BillinearFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.billinear = nn.Bilinear(in_channels, in_channels, in_channels)

    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x2 = x2.permute(0, 2, 3, 1).contiguous()
        x1 = self.billinear(x1, x2)
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        return x1


if __name__ == '__main__':
    x1 = torch.rand(32, 768, 4, 4)
    x2 = torch.rand(32, 768, 4, 4)
    fusion = CatFusion()
    x = fusion(x1, x2)
    print(x.shape)
    torch.save(fusion.state_dict(), 'CatFusion.pth')
    fusion = DotFusion()
    x = fusion(x1, x2)
    print(x.shape)
    torch.save(fusion.state_dict(), 'DotFusion.pth')
    fusion = BillinearFusion(768)
    x = fusion(x1, x2)
    print(x.shape)
    torch.save(fusion.state_dict(), 'BillinearFusion.pth')
