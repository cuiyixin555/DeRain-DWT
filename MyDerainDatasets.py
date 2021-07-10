import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
import glob
import random

class TrainValDataset(Dataset):
    def __init__(self, data_path, patch_size):
        super().__init__()
        self.rand_state = RandomState(66)

        self.root_rain = os.path.join(data_path, 'rain')
        self.root_norain = os.path.join(data_path, 'norain')
        self.root_clean = os.path.join(data_path, 'clean')

        self.rain_path = glob.glob(os.path.join(self.root_rain, '*.png'))
        self.rain_path.sort()

        self.norain_path = glob.glob(os.path.join(self.root_norain, '*.png'))
        self.norain_path.sort()

        self.clean_path = glob.glob(os.path.join(self.root_clean, '*.png'))
        self.clean_path.sort()

        self.file_num = len(self.rain_path) # 12000
        # print('file_num: ', self.file_num)

        self.patch_size = patch_size

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        imageId = (idx) % self.file_num
        cleanId = self.rand_state.randint(1, 100)

        # imageName = 'norain-%d.png' % (imageId)
        cleanName = 'clear_%03d.png' % (cleanId )

        # rain_name = os.path.join(self.root_rain, imageName)
        rain_name = self.rain_path[imageId]
        # print('rain_name: ', rain_name)

        # norain_name = os.path.join(self.root_norain, imageName)
        norain_name = self.norain_path[imageId]
        # print('norain_name: ', norain_name)

        clean_name = os.path.join(self.root_clean, cleanName)
        # print('clean_name: ', clean_name)

        rainImage = cv2.imread(rain_name).astype(np.float32) / 255
        # print('rainImage: ', rainImage.shape)

        norainImage = cv2.imread(norain_name).astype(np.float32) / 255
        # print('norainImage: ', norainImage.shape)

        cleanImage = cv2.imread(clean_name).astype(np.float32) / 255
        # print('cleanImgae: ', cleanImage.shape)

        O, B, C = self.crop(rainImage, norainImage, cleanImage)

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        C = np.transpose(C, (2, 0, 1))

        sample = {'O': O, 'B': B, 'C': C}

        return sample

    def crop(self, rainImage, norainImage, cleanImage):
        patch_size = self.patch_size
        rh, rw, rc = rainImage.shape
        nh, nw, nc = norainImage.shape
        ph, pw, pc = cleanImage.shape

        if (rh == nh) and (rw == nw) and (rc == nc):
            p_h = self.patch_size
            p_w = self.patch_size

        r = self.rand_state.randint(0, rh - p_h)
        c = self.rand_state.randint(0, rw - p_w)

        pr = self.rand_state.randint(0, ph - p_h)
        pc = self.rand_state.randint(0, pw - p_w)

        O = rainImage[r: r+p_h, c: c+p_w]
        B = norainImage[r: r+p_h, c: c+p_w]
        C = cleanImage[pr: pr+p_h, pc: pc+p_w]

        O = cv2.resize(O, (patch_size, patch_size))
        B = cv2.resize(B, (patch_size, patch_size))
        C = cv2.resize(C, (patch_size, patch_size))

        return O, B, C


# class TestDataset(Dataset):
#     def __init__(self, name):
#         super().__init__()
#         self.rand_state = RandomState(66)
#         self.root_dir = os.path.join(settings.data_dir, name)
#         self.mat_files = os.listdir(self.root_dir)
#         self.patch_size = settings.patch_size
#         self.file_num = len(self.mat_files)
#
#     def __len__(self):
#         return self.file_num
#
#     def __getitem__(self, idx):
#         file_name = self.mat_files[idx % self.file_num]
#         img_file = os.path.join(self.root_dir, file_name)
#         img_pair = cv2.imread(img_file).astype(np.float32) / 255
#         h, ww, c = img_pair.shape
#         w = ww // 2
#         O = np.transpose(img_pair[:, w:], (2, 0, 1))
#         B = np.transpose(img_pair[:, :w], (2, 0, 1))
#         sample = {'O': O, 'B': B}
#         return sample
#
#
# class ShowDataset(Dataset):
#     def __init__(self, name):
#         super().__init__()
#         self.rand_state = RandomState(66)
#         self.root_dir = os.path.join(settings.data_dir, name)
#         self.img_files = sorted(os.listdir(self.root_dir))
#         self.file_num = len(self.img_files)
#
#     def __len__(self):
#         return self.file_num
#
#     def __getitem__(self, idx):
#         file_name = self.img_files[idx % self.file_num]
#         img_file = os.path.join(self.root_dir, file_name)
#         print(img_file)
#         img_pair = cv2.imread(img_file).astype(np.float32) / 255
#
#         h, ww, c = img_pair.shape
#         w = int(ww / 2)
#
#         #h_8 = h % 8
#         #w_8 = w % 8
#
#         if settings.pic_is_pair:
#             O = np.transpose(img_pair[:, w:], (2, 0, 1))
#             B = np.transpose(img_pair[:, :w], (2, 0, 1))
#         else:
#             O = np.transpose(img_pair[:, :], (2, 0, 1))
#             B = np.transpose(img_pair[:, :], (2, 0, 1))
#
#         sample = {'O': O, 'B': B,'file_name':file_name[:-4]}
#
#         return sample


if __name__ == '__main__':
    dt = TrainValDataset('val')
    print('TrainValDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    dt = TestDataset('test')
    print('TestDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    print('ShowDataset')
    dt = ShowDataset('test')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())
