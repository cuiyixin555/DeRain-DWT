import torch
import torch.nn as nn
import module.common as common
import torch.nn.functional as F

# in-scale non-local attention
class NonLocalAttention(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True, conv=common.default_conv):
        super(NonLocalAttention, self).__init__()
        self.conv_match1 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match2 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())

    def forward(self, input):
        x_embed_1 = self.conv_match1(input) # [12, 16, 256, 256]
        x_embed_2 = self.conv_match2(input) # [12, 16, 256, 256]
        x_assembly = self.conv_assembly(input) # [12, 32, 256, 256]

        N, C, H, W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C))
        print('x_embed_1: ', x_embed_1.shape) # [12, 65536, 16]
        x_embed_2 = x_embed_2.view(N, C, H * W)
        print('x_embed_2: ', x_embed_2.shape) # [12, 16, 65536]
        score = torch.matmul(x_embed_1, x_embed_2)
        print('score: ', score.shape)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(N, -1, H * W).permute(0, 2, 1)
        print('x_assembly: ', x_assembly.shape)
        x_final = torch.matmul(score, x_assembly)

        return x_final.permute(0, 2, 1).view(N, -1, H, W)

if __name__ == '__main__':
    net = NonLocalAttention(32, 2)
    ceshi = torch.Tensor(12, 32, 100, 100)
    output = net(ceshi)
    print('output: ', output.shape)
