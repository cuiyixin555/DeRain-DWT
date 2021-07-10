import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *
from module.generator_cbam135 import BRN, print_network
import time

parser = argparse.ArgumentParser(description="AID_DWT_Test")
parser.add_argument("--logdir", type=str, default='/media/ubuntu/Seagate/PRICAI/DWT-Net/weight/CBAM/Rain200H/')
parser.add_argument("--data_path", type=str, default='/media/ubuntu/Seagate/RainData/Rain200H/test/small/rain/')
parser.add_argument("--save_path", type=str, default='/media/ubuntu/Seagate/PRICAI/experiments/Rain200H_CBAM135_100/')
parser.add_argument('--save_path_r', type=str, default='/media/ubuntu/Seagate/PRICAI/experiments/Rain200H_CBAM135_100/streak/')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default='0,1', help='GPU id')
parser.add_argument("--inter_iter", type=int, default=8, help='number of inter_iteration')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def main():
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.isdir(opt.save_path_r):
        os.makedirs(opt.save_path_r)

    # Build model
    print('Loading model ...\n')
    model = BRN(opt.inter_iter, opt.use_GPU)
    model = nn.DataParallel(model)
    # print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    state_dict = torch.load(os.path.join(opt.logdir, 'net_latest.pth'))
    model.load_state_dict(state_dict)

    model.eval()
    # load data info
    print('Loading data info ...\n')

    # process data
    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)
            fname = os.path.basename(img_name)
            fname = os.path.splitext(fname)[0]

            # image
            Img = cv2.imread(img_path)
            h, w, c = Img.shape

            b, g, r = cv2.split(Img)
            Img = cv2.merge([r, g, b])

            Img = normalize(np.float32(Img))
            Img = np.expand_dims(Img.transpose(2, 0, 1), 0)

            ISource = torch.Tensor(Img)

            INoisy = ISource

            if opt.use_GPU:
                ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
            else:
                ISource, INoisy = Variable(ISource), Variable(INoisy)

            with torch.no_grad():  # this can save much memory
                torch.cuda.synchronize()
                start_time = time.time()
                out, out_list, rain, rain_list, = model(INoisy)

                out = torch.clamp(out, 0., 1.)
                rain = torch.clamp(rain, 0., 1.)

                torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                print(img_name)
                print(dur_time)
                time_test += dur_time

            if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())
                save_rain = np.uint8(255 * rain.data.cpu().numpy().squeeze())
            else:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())
                save_rain = np.uint8(255 * rain.data.cpu().numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            save_rain = save_rain.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])
            b, g, r = cv2.split(save_rain)
            save_rain = cv2.merge([r, g, b])

            save_path = opt.save_path
            save_path_r = opt.save_path_r
            cv2.imwrite(os.path.join(save_path, img_name), save_out)
            cv2.imwrite(os.path.join(save_path_r, img_name), save_rain)

            count = count + 1

    print('Avg. time:', time_test / count)

if __name__ == "__main__":
    main()
