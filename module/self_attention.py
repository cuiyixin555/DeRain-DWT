import torch
import torch.nn as nn
from function.modules import Subtraction, Subtraction2, Aggregation

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

class SAM(nn.Module):
    def __init__(self, in_planes, rel_planes, out_planes, share_planes, sa_type=0, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        # in_planes = 32  rel_planes = 2, out_planes = 8, share_planes = 32
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride # 0, 3, 1

        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1) # nn.Conv2d(32, 2, 1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1) # nn.Conv2d(32, 2, 1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1) # nn.Conv2d(32, 8, 1)

        # if sa_type == 0:
        self.conv_w = nn.Sequential(nn.LeakyReLU(0.2),
                                    nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                    nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
        self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
        self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
        self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
        self.softmax = nn.Softmax(dim=-2)

        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
        self.convLayer = nn.Conv2d(share_planes, in_planes, 1, 1, 0)

    def forward(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        if self.sa_type == 0:  # pairwise
            p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
            w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])
            x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1, x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])

        x = self.aggregation(x3, w)
        x = self.convLayer(x)
        return x

if __name__ == '__main__':
    channel = 32
    net = SAM(channel, channel // 16, channel // 4, 8).cuda()
    ceshi = torch.Tensor(12, 32, 256, 256).cuda()
    output = net(ceshi)
