# PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
# Tools lib
import numpy as np
import random
import time
import os
from common import *
from module.self_attention import *

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class BRN(nn.Module):
    def __init__(self, recurrent_iter=4, use_GPU=True):
        super(BRN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

        self.conv0_r = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_r = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

        # TODO: SAM
        self.aid_module = SAM(32, 2, 8, 8)

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        r = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))
        h_r = Variable(torch.zeros(batch_size, 32, row, col))
        c_r = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()
            h_r = h_r.cuda()
            c_r = c_r.cuda()

        x_list = []
        r_list = []

        for i in range(self.iteration):
            r = torch.cat((input, r), 1)
            r = self.conv0_r(r)
            r = torch.cat((r, h_r, h), 1)
            i_r = self.conv_i_r(r)
            f_r = self.conv_f_r(r)
            g_r = self.conv_g_r(r)
            o_r = self.conv_o_r(r)
            c_r = f_r * c_r + i_r * g_r
            h_r = o_r * F.tanh(c_r)

            resr = h_r
            r = F.relu(self.res_conv1_r(h_r) + resr)  # [6, 32, 100, 100]
            # TODO 1: san_r1
            aid_r1 = self.aid_module(r)

            resr = r
            r = F.relu(self.res_conv2_r(r) + resr)  # [6, 32, 100, 100]
            # TODO 2: san_r2
            aid_r2 = self.aid_module(r)

            resr = r
            r = F.relu(self.res_conv3_r(r) + resr)  # [6, 32, 100, 100]
            # TODO 3: san_r3
            aid_r3 = self.aid_module(r)

            r = self.conv_r(r)
            r_list.append(r)

            x = torch.cat((input, x, r), 1)
            x = self.conv0(x)
            x = torch.cat((x, h, h_r), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * F.tanh(c)

            resx = h
            x = F.relu(self.res_conv1(h) + resx)
            # TODO: san_x1
            aid_x1 = aid_r1 * x + x
            resx = aid_x1
            x = F.relu(self.res_conv2(aid_x1) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            # TODO: san_x2
            aid_x2 = aid_r2 * x + x
            resx = aid_x2
            x = F.relu(self.res_conv4(aid_x2) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            # TODO: san_x3
            aid_x3 = aid_r3 * x + x

            x = self.conv(aid_x3)
            x_list.append(x)

        return x, x_list, r, r_list

if __name__ == '__main__':
    net = BRN(1).cuda()
    ceshi = torch.Tensor(12, 3, 256, 256).cuda()
    # print('ceshi: ', ceshi.shape)
    output, *_ = net(ceshi)
    # print('output: ', output.shape)





