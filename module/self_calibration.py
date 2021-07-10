import torch.nn as nn
import torch
import torch.nn.functional as F

class SCConv(nn.Module):
    def __init__(self, planes, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(planes, planes, 3, 1, 1),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(planes, planes, 3, 1, 1),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(planes, planes, 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

class SCBottleneck(nn.Module):
    # expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, in_planes, planes):
        super(SCBottleneck, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = nn.Conv2d(in_planes, planes, 1, 1)

        self.k1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.conv1_b = nn.Conv2d(in_planes, planes, 1, 1)

        self.scconv = SCConv(planes, self.pooling_r)

        self.conv3 = nn.Conv2d(planes * 2, planes * 2, 1, 1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.relu(out_a)
        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.relu(out_b)
        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))

        return out

if __name__ == '__main__':
    net = SCBottleneck(32, 32)
    ceshi = torch.Tensor(12, 32, 100, 100)
    output = net(ceshi)
    print('output: ', output.shape)
