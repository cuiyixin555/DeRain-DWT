import torch
from torch import nn
import torch.nn.functional as F
import networks.settings_dcsfn as settings
from itertools import combinations,product
import math

class Inner_scale_connection_block(nn.Module):
    def __init__(self):
        super(Inner_scale_connection_block, self).__init__()
        self.channel = 20

        self.scale1 = nn.ModuleList()
        self.scale2 = nn.ModuleList()
        self.scale4 = nn.ModuleList()
        self.scale8 = nn.ModuleList()

        for i in range(4):
            self.scale1.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.scale2.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.scale4.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.scale8.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
        self.fusion84 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        self.fusion42 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        self.fusion21 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        self.pooling8 = nn.MaxPool2d(8, 8)
        self.pooling4 = nn.MaxPool2d(4, 4)
        self.pooling2 = nn.MaxPool2d(2, 2)
        self.fusion_all = nn.Sequential(nn.Conv2d(4 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))

    def forward(self, x):
        feature8 = self.pooling8(x)
        b8, c8, h8, w8 = feature8.size()
        feature4 = self.pooling4(x)
        b4, c4, h4, w4 = feature4.size()
        feature2 = self.pooling2(x)
        b2, c2, h2, w2 = feature2.size()
        feature1 = x
        b1, c1, h1, w1 = feature1.size()
        for i in range(4):
            feature8 = self.scale8[i](feature8)
        scale8 = feature8
        feature4 = self.fusion84(torch.cat([feature4, F.upsample(scale8, [h4, w4])], dim=1))
        for i in range(4):
            feature4 = self.scale4[i](feature4)
        scale4 = feature4
        feature2 = self.fusion42(torch.cat([feature2, F.upsample(scale4, [h2, w2])], dim=1))
        for i in range(4):
            feature2 = self.scale2[i](feature2)
        scale2 = feature2
        feature1 = self.fusion21(torch.cat([feature1, F.upsample(scale2, [h1, w1])], dim=1))
        for i in range(4):
            feature1 = self.scale1[i](feature1)
        scale1 = feature1
        fusion_all = self.fusion_all(torch.cat([scale1, F.upsample(scale2, [h1, w1]), F.upsample(scale4, [h1, w1]), F.upsample(scale8, [h1, w1])], dim=1))
        return fusion_all + x

class Encoder_decoder_block(nn.Module):
    def __init__(self):
        super(Encoder_decoder_block, self).__init__()
        self.channel_num = settings.channel
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_1 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_2 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fusion2_3 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.pooling2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        input = x
        encoder1 = self.encoder_conv1(x)
        b1, c1, h1, w1 = encoder1.size()
        pooling1 = self.pooling2(encoder1)
        encoder2 = self.encoder_conv2(pooling1)
        b2, c2, h2, w2 = encoder2.size()
        pooling2 = self.pooling2(encoder2)
        encoder3 = self.encoder_conv3(pooling2)

        decoder_conv1 = self.decoder_conv1(encoder3)
        decoder_conv2 = self.decoder_conv2(self.fusion2_2(torch.cat([F.upsample(encoder2, [h2, w2]), F.upsample(decoder_conv1, [h2, w2])], dim=1)))
        decoder_conv3 = self.decoder_conv3(self.fusion2_3(torch.cat([F.upsample(encoder1, [h1, w1]), F.upsample(decoder_conv2, [h1, w1])], dim=1)))
        return decoder_conv3 + input

Scale_block = Inner_scale_connection_block()

class ConvGRU(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_xz = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xr = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xn = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_hz = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hr = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hn = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        if h is None:
            z = F.sigmoid(self.conv_xz(x))
            f = F.tanh(self.conv_xn(x))
            h = z * f
        else:
            z = F.sigmoid(self.conv_xz(x) + self.conv_hz(h))
            r = F.sigmoid(self.conv_xr(x) + self.conv_hr(h))
            n = F.tanh(self.conv_xn(x) + self.conv_hn(r * h))
            h = (1 - z) * h + z * n

        h = self.relu(h)

        return h, h

RecUnit = {
    'GRU': ConvGRU,
}['GRU']

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.unit_num = 16
        self.units = nn.ModuleList()
        self.channel_num = 20
        self.conv1x1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.units.append(Scale_block)
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i + 2) * self.channel_num, self.channel_num, 1, 1), nn.LeakyReLU(0.2)))

    def forward(self, x):
        catcompact = []
        catcompact.append(x)
        feature = []
        out = x
        for i in range(self.unit_num):
            tmp = self.units[i](out)
            feature.append(tmp)
            catcompact.append(tmp)
            out = self.conv1x1[i](torch.cat(catcompact, dim=1))
        return out, feature


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.unit_num = 16
        self.units = nn.ModuleList()
        self.channel_num = 20
        self.conv1x1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.units.append(Scale_block)
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i + 2) * self.channel_num, self.channel_num, 1, 1), nn.LeakyReLU(0.2)))

    def forward(self, x, feature):
        catcompact=[]
        catcompact.append(x)
        out = x
        for i in range(self.unit_num):
            tmp = self.units[i](out + feature[i])
            catcompact.append(tmp)
            out = self.conv1x1[i](torch.cat(catcompact, dim=1))
        return out

class Multi_model_fusion_learning(nn.Module):
    def __init__(self):
        super(Multi_model_fusion_learning, self).__init__()
        self.channel_num = 20
        self.convert = nn.Sequential(
            nn.Conv2d(3, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )

        self.encoder_scale1 = Encoder()
        self.encoder_scale2 = Encoder()
        self.encoder_scale4 = Encoder()

        self.fusion_1_2_1 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2))
        self.fusion_1_2_2 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2))
        self.fusion_2_2_1 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2))
        self.fusion_2_2_2 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1),
            nn.LeakyReLU(0.2))

        self.decoder_scale1 = Decoder()
        self.decoder_scale2 = Decoder()
        self.decoder_scale4 = Decoder()

        self.rec1_1 = RecUnit(self.channel_num, self.channel_num, 3, 1)
        self.rec1_2 = RecUnit(self.channel_num, self.channel_num, 3, 1)
        self.rec1_4 = RecUnit(self.channel_num, self.channel_num, 3, 1)

        self.rec2_1 = RecUnit(self.channel_num, self.channel_num, 3, 1)
        self.rec2_2 = RecUnit(self.channel_num, self.channel_num, 3, 1)
        self.rec2_4 = RecUnit(self.channel_num, self.channel_num, 3, 1)

        self.merge = nn.Sequential(
            nn.Conv2d(3 * self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(3, 3, 3, 1, 1)
        )

        self.pooling2 = nn.MaxPool2d(2, 2)
        self.pooling4 = nn.MaxPool2d(4, 4)

    def forward(self, x):
        convert = self.convert(x)
        feature1 = convert
        feature2 = self.pooling2(convert)
        feature4 = self.pooling4(convert)

        b1, c1, h1, w1 = feature1.size()
        b2, c2, h2, w2 = feature2.size()
        b4, c4, h4, w4 = feature4.size()

        scale1_encoder, scale1_feature = self.encoder_scale1(feature1)
        scale2_encoder, scale2_feature = self.encoder_scale2(feature2)
        scale4_encoder, scale4_feature = self.encoder_scale4(feature4)

        current1_4, rec1_4 = self.rec1_4(scale4_encoder)
        rec1_4_ori = rec1_4

        rec1_4 = F.upsample(rec1_4_ori, [h2, w2])
        current1_2, rec1_2 = self.rec1_2(scale2_encoder, rec1_4)
        rec1_2_ori = rec1_2
        rec1_2 = F.upsample(rec1_2_ori, [h1, w1])
        rec1_4 = F.upsample(rec1_4_ori, [h1, w1])
        current1_1, rec1_1 = self.rec1_1(scale1_encoder, self.fusion_1_2_1(torch.cat([rec1_2, rec1_4], dim=1)))

        current2_1, rec2_1 = self.rec2_1(current1_1)
        rec2_1_ori = rec2_1

        rec2_1 = F.upsample(rec2_1_ori, [h2, w2])
        current2_2, rec2_2 = self.rec2_2(current1_2, rec2_1)
        rec2_2_ori = rec2_2
        rec2_2 = F.upsample(rec2_2_ori, [h4, w4])
        rec2_1 = F.upsample(rec2_1_ori, [h4, w4])
        current2_4, rec2_4 = self.rec2_4(current1_4, self.fusion_2_2_1(torch.cat([rec2_1, rec2_2],dim=1)))

        scale1_decoder = self.decoder_scale1(current2_1, scale1_feature)
        scale2_decoder = self.decoder_scale2(current2_2, scale2_feature)
        scale4_decoder = self.decoder_scale4(current2_4, scale4_feature)
        merge = self.merge(torch.cat([scale1_decoder, F.upsample(scale2_decoder,[h1,w1]), F.upsample(scale4_decoder,[h1,w1])],dim=1))

        return x-merge

Net = Multi_model_fusion_learning
