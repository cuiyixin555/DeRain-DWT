import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from DerainDataset import *
from utils import *
import cv2
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms.functional as F
import pytorch_ssim
from networks.generator_aid135 import BRN
from DWT import *
from patchGan import *
from ganLoss import *

parser = argparse.ArgumentParser(description="AID-DWT")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=22, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")  # 1e-3L   5e-4H   5e-5R1400
parser.add_argument("--save_path", type=str, default='./logs/Ablation/r9/')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
parser.add_argument("--data_path", type=str, default='/media/ubuntu/Seagate/RainData/Rain200H/train/small/')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default='0,1', help='GPU id')
parser.add_argument("--inter_iter", type=int, default=9, help='number of inter_iteration')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def main():
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)

    # Load dataset
    print('Loading dataset ...\n')
    if (opt.data_path.find('Light') != -1 or opt.data_path.find('Heavy') != -1):
        dataset_train = newDataset(data_path=opt.data_path)
    else:
        dataset_train = MyDataset(data_path=opt.data_path)

    # dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=8, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    net = BRN(recurrent_iter=opt.inter_iter, use_GPU=opt.use_GPU)
    net = nn.DataParallel(net)

    # Build discriminator
    net_D = NLayerDiscriminator(3)
    net_D = nn.DataParallel(net_D)

    criterion = pytorch_ssim.SSIM()

    # Move to GPU
    model = net.cuda()
    model_D = net_D.cuda()
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    optimizer_D = optim.Adam(net_D.parameters(), lr=opt.lr)

    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 80], gamma=0.2)  # learning rates
    scheduler_D = MultiStepLR(optimizer, milestones=[30, 50, 80], gamma=0.2)

    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)  # load the last model in matconvnet style

    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        scheduler_D.step(epoch)

        # set learning rate
        for param_group in optimizer.param_groups:
            # param_group["lr"] = current_lr
            print('learning rate %f' % param_group["lr"])

        # train
        for i, (input, target, clear) in enumerate(loader_train, 0):

            # training step
            model.train()
            model.zero_grad()

            model_D.train()
            model_D.zero_grad()

            optimizer.zero_grad()
            optimizer_D.zero_grad()

            # read original data
            input_train = Variable(input.cuda())
            target_train = Variable(target.cuda())
            clear_train = Variable(clear.cuda())

            out_train, _, _, _ = model(input_train)

            # dwt convert
            _, out_hl, out_lh, out_hh, _ = dwt_init(out_train)
            _, clear_hl, clear_lh, clear_hh, _ = dwt_init(clear_train)

            # TODO:
            output_clear_lh = model_D(clear_lh)
            errD_clear_lh = -output_clear_lh.mean()

            output_clear_hl = model_D(clear_hl)
            errD_clear_hl = -output_clear_hl.mean()

            output_clear_hh = model_D(clear_hh)
            errD_clear_hh = -output_clear_hh.mean()

            fake_lh = out_lh
            output_fake_lh = model_D(fake_lh.detach())
            errD_fake_lh = output_fake_lh.mean()

            fake_hl = out_hl
            output_fake_hl = model_D(fake_hl.detach())
            errD_fake_hl = output_fake_hl.mean()

            fake_hh = out_hh
            output_fake_hh = model_D(fake_hh.detach())
            errD_fake_hh = output_fake_hh.mean()

            gradient_penalty_lh = calc_gradient_penalty(model_D, clear_lh, out_lh, 0.1)
            errD_lh = errD_clear_lh + errD_fake_lh + gradient_penalty_lh
            errD_lh.backward()

            gradient_penalty_hl = calc_gradient_penalty(model_D, clear_hl, out_hl, 0.1)
            errD_hl = errD_clear_hl + errD_fake_hl + gradient_penalty_hl
            errD_hl.backward()

            gradient_penalty_hh = calc_gradient_penalty(model_D, clear_hh, out_hh, 0.1)
            errD_hh = errD_clear_hh + errD_fake_hh + gradient_penalty_hh
            errD_hh.backward()

            optimizer_D.step()

            # pixel Loss
            pixel_loss = criterion(target_train, out_train)

            loss = (-pixel_loss)  # + mse
            loss.backward()

            optimizer.step()

            # results
            model.eval()

            with torch.no_grad():
                out_train, _, out_r_train, _ = model(input_train)
                out_train = torch.clamp(out_train, 0., 1.)
                out_r_train = torch.clamp(out_r_train, 0., 1.)
                psnr_train = batch_PSNR(out_train, target_train, 1.)

            print("[epoch %d][%d/%d] loss: %.4f, PSNR_train: %.4f" % (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))

        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch + 1)))


if __name__ == "__main__":
    if opt.preprocess:
        prepare_data_Rain200H(data_path=opt.data_path, patch_size=100, stride=100)
    main()
