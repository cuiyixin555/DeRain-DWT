import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import *
import random


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            # print('k: \n', k)
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1

    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data_Rain200H(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rain')
    target_path = os.path.join(data_path, 'norain')
    clean_path = os.path.join(data_path, 'clean')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')
    save_clean_path = os.path.join(data_path, 'train_clean.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')
    clean_h5f = h5py.File(save_clean_path, 'w')

    train_num = 0
    for i in range(1800):
        # clear image name
        j = random.randint(1, 100)
        # print('j: ', j)
        clean_file = "clear_%03d.png" % (j)

        # target image name
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%d.png" % (i + 1)

        if os.path.exists(os.path.join(target_path, target_file)) and os.path.exists(os.path.join(input_path, input_file)) and os.path.exists(os.path.join(clean_path, clean_file)):
        # if os.path.exists(os.path.join(clear_path, clear_file)):
            target_img = cv2.imread(os.path.join(target_path, target_file))
            b, g, r = cv2.split(target_img)
            target_img = cv2.merge([r, g, b])
            target_img = np.float32(normalize(target_img))  # cv normalize
            target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            input_img = cv2.imread(os.path.join(input_path, input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])
            input_img = np.float32(normalize(input_img))  # cv normalize
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            clean_img = cv2.imread(os.path.join(clean_path, clean_file))
            b, g, r = cv2.split(clean_img)
            clean_img = cv2.merge([r, g, b])
            clean_img = np.float32(normalize(clean_img))  # cv normalize
            clean_patches = Im2Patch(clean_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            # print("target file: %s # samples: %d" % (clear_file, clear_patches.shape[3]))

            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                clean_data = clean_patches[:, :, :, n].copy()
                clean_h5f.create_dataset(str(train_num), data=clean_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()
    clean_h5f.close()

    print('training set, # samples %d\n' % train_num)

def prepare_data_Rain200L(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rain')
    target_path = os.path.join(data_path, 'norain')
    clean_path = os.path.join(data_path, 'clean')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')
    save_clean_path = os.path.join(data_path, 'train_clean.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')
    clean_h5f = h5py.File(save_clean_path, 'w')

    train_num = 0
    for i in range(1800):
        # clear image name
        j = random.randint(1, 100)
        # print('j: ', j)
        clean_file = "clear_%03d.png" % (j)

        # target image name
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%d.png" % (i + 1)

        if os.path.exists(os.path.join(target_path, target_file)) and os.path.exists(os.path.join(input_path, input_file)) and os.path.exists(os.path.join(clean_path, clean_file)):
        # if os.path.exists(os.path.join(clear_path, clear_file)):
            target_img = cv2.imread(os.path.join(target_path, target_file))
            b, g, r = cv2.split(target_img)
            target_img = cv2.merge([r, g, b])
            target_img = np.float32(normalize(target_img))  # cv normalize
            target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            input_img = cv2.imread(os.path.join(input_path, input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])
            input_img = np.float32(normalize(input_img))  # cv normalize
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            clean_img = cv2.imread(os.path.join(clean_path, clean_file))
            b, g, r = cv2.split(clean_img)
            clean_img = cv2.merge([r, g, b])
            clean_img = np.float32(normalize(clean_img))  # cv normalize
            clean_patches = Im2Patch(clean_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            # print("target file: %s # samples: %d" % (clear_file, clear_patches.shape[3]))

            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                clean_data = clean_patches[:, :, :, n].copy()
                clean_h5f.create_dataset(str(train_num), data=clean_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()
    clean_h5f.close()

    print('training set, # samples %d\n' % train_num)

def prepare_data_Rain1400(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rain')
    target_path = os.path.join(data_path, 'norain')
    clean_path = os.path.join(data_path, 'clean')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')
    save_clean_path = os.path.join(data_path, 'train_clean.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')
    clean_h5f = h5py.File(save_clean_path, 'w')

    train_num = 0
    for i in range(12600):
        # clear image name
        j = random.randint(1, 100)
        # print('j: ', j)
        clean_file = "clear_%03d.png" % (j)

        # target image name
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%d.png" % (i + 1)

        if os.path.exists(os.path.join(target_path, target_file)) and os.path.exists(os.path.join(input_path, input_file)) and os.path.exists(os.path.join(clean_path, clean_file)):
        # if os.path.exists(os.path.join(clear_path, clear_file)):
            target_img = cv2.imread(os.path.join(target_path, target_file))
            b, g, r = cv2.split(target_img)
            target_img = cv2.merge([r, g, b])
            target_img = np.float32(normalize(target_img))  # cv normalize
            target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            input_img = cv2.imread(os.path.join(input_path, input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])
            input_img = np.float32(normalize(input_img))  # cv normalize
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            clean_img = cv2.imread(os.path.join(clean_path, clean_file))
            b, g, r = cv2.split(clean_img)
            clean_img = cv2.merge([r, g, b])
            clean_img = np.float32(normalize(clean_img))  # cv normalize
            clean_patches = Im2Patch(clean_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            # print("target file: %s # samples: %d" % (clear_file, clear_patches.shape[3]))

            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                clean_data = clean_patches[:, :, :, n].copy()
                clean_h5f.create_dataset(str(train_num), data=clean_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()
    clean_h5f.close()

    print('training set, # samples %d\n' % train_num)

def prepare_data_Rain12600(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rainy_image')
    target_path = os.path.join(data_path, 'ground_truth')

    save_target_path = os.path.join(target_path, 'train_target.h5')
    save_input_path = os.path.join(input_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(900):
        target_file = "%d.jpg" % (i + 1)
        target = cv2.imread(os.path.join(target_path, target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(14):
            input_file = "%d_%d.jpg" % (i+1, j+1)
            input_img = cv2.imread(os.path.join(input_path, input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target
            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)
            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)
                train_num += 1

    target_h5f.close()
    input_h5f.close()
    print('training set, # samples %d\n' % train_num)

def prepare_data_RainTrainH(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        if os.path.exists(os.path.join(target_path, target_file)):
            target = cv2.imread(os.path.join(target_path,target_file))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])

            input_file = "rain-%d.png" % (i + 1)

            if os.path.exists(os.path.join(input_path, input_file)): # we delete 546 samples

                input_img = cv2.imread(os.path.join(input_path, input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                target_img = target
                target_img = np.float32(normalize(target_img)) # cv normalize
                target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

                for n in range(target_patches.shape[3]):
                    target_data = target_patches[:, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data)

                    input_data = input_patches[:, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data)

                    train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)

def prepare_data_RainTrainL(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path, target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2):
            input_file = "rain-%d.png" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path, input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])
            target_img = target

            if j == 1:
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()
    print('training set, # samples %d\n' % train_num)

def prepare_data_real(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path, target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2):
            input_file = "rain-%d.png" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path, input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])
            target_img = target

            if j == 1:
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()
    print('training set, # samples %d\n' % train_num)

def prepare_data_newrain(data_path, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    scales = [1]
    input_path = os.path.join(data_path, 'rain')
    target_path = os.path.join(data_path, 'norain')


    save_target_path = os.path.join(data_path, 'norain', 'train_target.h5')
    save_input_path = os.path.join(data_path, 'rain', 'train_input.h5')


    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')


    train_num = 0
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])
        #print(target.shape)
        h, w, c = target.shape

        #for j in range(2):
        input_file = "norain-%dx2.png" % (i + 1)
        input_img = cv2.imread(os.path.join(input_path,input_file))
        b, g, r = cv2.split(input_img)
        input_img = cv2.merge([r, g, b])


    #for k in range(len(scales)):
        target_img = target

        # if j == 1:
        #     target_img = cv2.flip(target_img, 1)
        #     input_img = cv2.flip(input_img, 1)
        #     rain_img = cv2.flip(rain_img, 1)

        #target_img = np.expand_dims(target_img.copy(), 0)
        target_img = np.float32(normalize(target_img))
        target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

        #input_img = np.expand_dims(input_img.copy(), 0)
        input_img = np.float32(normalize(input_img))
        input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)


        print("target file: %s # samples: %d" % (
        input_file, target_patches.shape[3] * aug_times))
        for n in range(target_patches.shape[3]):
            target_data = target_patches[:, :, :, n].copy()
            target_h5f.create_dataset(str(train_num), data=target_data)

            input_data = input_patches[:, :, :, n].copy()
            input_h5f.create_dataset(str(train_num), data=input_data)

            train_num += 1
            # for m in range(aug_times-1):
            #    target_data_aug = data_augmentation(target_data, np.random.randint(1,8))
            #    target_h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=target_data_aug)
            #    train_num += 1
    target_h5f.close()
    input_h5f.close()
    print('training set, # samples %d\n' % train_num)


class newDataset(udata.Dataset):
    def __init__(self, train=True, data_path='.'):
        super(newDataset, self).__init__()
        self.train = train
        self.data_path = data_path
        if self.train:
            target_path = os.path.join(self.data_path, 'norain', 'train_target.h5')
            input_path = os.path.join(self.data_path, 'rain', 'train_input.h5')

            target_h5f = h5py.File(target_path, 'r')
            input_h5f = h5py.File(input_path, 'r')

        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            target_path = os.path.join(self.data_path, 'norain', 'train_target.h5')
            input_path = os.path.join(self.data_path, 'rain', 'train_input.h5')

            target_h5f = h5py.File(target_path, 'r')
            input_h5f = h5py.File(input_path, 'r')

        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(target)


class Dataset(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset, self).__init__()
        self.data_path = data_path

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(target)

class MyDataset(udata.Dataset):
    def __init__(self, data_path='.'):
        super(MyDataset, self).__init__()

        self.data_path = data_path
        self.data_path = os.path.join(self.data_path)

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')
        clean_path = os.path.join(self.data_path, 'train_clean.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')
        clean_h5f = h5py.File(clean_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()
        clean_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')
        clean_path = os.path.join(self.data_path, 'train_clean.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')
        clean_h5f = h5py.File(clean_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])
        clean = np.array(clean_h5f[key])

        target_h5f.close()
        input_h5f.close()
        clean_h5f.close()

        return torch.Tensor(input), torch.Tensor(target), torch.Tensor(clean)
