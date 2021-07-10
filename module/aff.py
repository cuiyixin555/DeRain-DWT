import torch
import torch.nn as nn
import torchvision

# TODO: ResBlock
class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()

        self.channel = channel

        self.convLayer = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel, self.channel, 3, 1, 1),
        )

        self.active = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, fea):
        residual = fea
        output = self.convLayer(fea)
        output = torch.add(output, residual)
        final = self.active(output)
        return final

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.resLayer = ResBlock(32)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = self.resLayer(x)
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

if __name__ == '__main__':
    net = AFF(32, 4)
    ceshi = torch.Tensor(12, 32, 100, 100)
    output = net(ceshi)

    print('output: ', output.shape)