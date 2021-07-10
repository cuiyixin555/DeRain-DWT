import torch.nn.functional as F
import torch.nn as nn
import torch
import module.common as common
from module.tools import *

# cross-scale non-local attention
class CrossScaleAttention(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True,
                 conv=common.default_conv):
        super(CrossScaleAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.softmax_scale = softmax_scale

        self.scale = scale
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_1 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match_2 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())

    def forward(self, input):
        # get embedding
        embed_w = self.conv_assembly(input)
        match_input = self.conv_match_1(input)

        # b*c*h*w
        shape_input = list(embed_w.size())  # b*c*h*w
        input_groups = torch.split(match_input, 1, dim=0)

        # kernel size on input for matching
        kernel = self.scale * self.ksize

        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(embed_w, ksizes=[kernel, kernel],
                                      strides=[self.stride * self.scale, self.stride * self.scale],
                                      rates=[1, 1],
                                      padding='same')  # [N, C*k*k, L]

        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(shape_input[0], shape_input[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling X to form Y for cross-scale matching
        ref = F.interpolate(input, scale_factor=1. / self.scale, mode='bilinear')
        ref = self.conv_match_2(ref)
        w = extract_image_patches(ref, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')

        shape_ref = ref.shape
        # w shape: [N, C, k, k, L]
        w = w.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        y = []
        scale = self.softmax_scale
        # 1*1*k*k
        # fuse_weight = self.fuse_weight

        for xi, wi, raw_wi in zip(input_groups, w_groups, raw_w_groups):
            # normalize
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi / max_wi

            # Compute correlation map
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W] L = shape_ref[2]*shape_ref[3]

            yi = yi.view(1, shape_ref[2] * shape_ref[3], shape_input[2], shape_input[3])  # (B=1, C=32*32, H=32, W=32)
            # rescale matching score
            yi = F.softmax(yi * scale, dim=1)
            if self.average == False:
                yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()

            # deconv for reconsturction
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.stride * self.scale, padding=self.scale)

            yi = yi / 6.
            y.append(yi)

        y = torch.cat(y, dim=0)
        return y

if __name__ == '__main__':
    net = CrossScaleAttention(32, 2)
    ceshi = torch.Tensor(12, 32, 100, 100)
    output = net(ceshi)
    print('output: ', output.shape)

