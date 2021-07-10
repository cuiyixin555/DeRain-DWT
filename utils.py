import math
import torch
import re
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
import  os
import glob 

import torch
import math
import cv2

irange = range

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid

def make_image_grid(x, ngrid):
    x = x.clone().cpu()
    if pow(ngrid,2) < x.size(0):
        grid = make_grid(x[:ngrid*ngrid], nrow=ngrid, padding=0, normalize=True, scale_each=False)
    else:
        grid = torch.FloatTensor(ngrid*ngrid, x.size(1), x.size(2), x.size(3)).fill_(1)
        grid[:x.size(0)].copy_(x)
        grid = make_grid(grid, nrow=ngrid, padding=0, normalize=True, scale_each=False)
    return grid

def save_image_grid(x, path, imsize=512, ngrid=4):
    from PIL import Image
    grid = make_image_grid(x, ngrid)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize,imsize), Image.NEAREST)
    im.save(path)

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def normalize(data):
    return data / 255.

def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return  False

# TODO: two pixel shuffle functions to process the images
def pixelshuffle(image, scale):
    '''
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    '''
    if scale == 1:
        return image

    w, h, c = image.shape
    mosaic = np.array([])
    temp_list = []
    temp_cv_list = []
    for ws in range(scale):
        band = np.array([])
        # print('band: ', band)
        for hs in range(scale):
            temp = image[ws::scale, hs::scale, :]  # get the sub-sampled image
            temp_list.append(temp)
            # print('temp: ', temp.shape) # [161, 241, 3] -> [161, 241, 3] -> [160, 241, 3] -> [160, 240, 3]
            band = np.concatenate((band, temp), axis=1) if band.size else temp
            # print('band: ', band.shape) # [161, 241, 3] -> [161, 481, 3] -> [160, 241, 3] -> [160, 481, 3]
        mosaic = np.concatenate((mosaic, band), axis=0) if mosaic.size else band
    # # mosaic_copy = np.clip(mosaic, 0.0, 1.0)
    # b, g, r = cv2.split(mosaic)
    # mosaic_copy = cv2.merge([r, g, b])
    # mosaic_copy = np.uint8(mosaic_copy)

    # # add small downsampler image
    # for i in range(len(temp_list)):
    #     temp_copy = temp_list[i]
    #     b, g, r = cv2.split(temp_copy)
    #     temp_copy = cv2.merge([r, g, b])
    #     temp_copy = np.uint8(temp_copy)
    #     temp_cv_list.append(temp_copy)

    # cv2.imwrite('./ceshi/pixelshuffle.png', mosaic_copy)
    # return mosaic, mosaic_copy, temp_cv_list
    return mosaic

def reverse_pixelshuffle(image, scale, fill=0, fill_image=0, ind=[0, 0]):
    '''
    Discription: Given a mosaic image of subsampling, recombine it to a full image
    [Input]: Image
    [Return]: Recombine it using different portions of pixels
    '''
    w, h, c = image.shape
    # print('w: ', w) # 512
    # print('h: ', h) # 512
    # print('c: ', c) # 3
    real = np.zeros((w, h, c))  # real image
    wf = 0
    hf = 0
    for ws in range(scale):
        hf = 0
        for hs in range(scale):
            temp = real[ws::scale, hs::scale, :]
            wc, hc, cc = temp.shape  # get the shape of the current images
            # print('wc: ', wc) # 256
            # print('hc: ', hc) # 256
            # print('cc: ', cc) # 3
            if fill == 1 and ws == ind[0] and hs == ind[1]:
                real[ws::scale, hs::scale, :] = fill_image[wf:wf + wc, hf:hf + hc, :]
            else:
                real[ws::scale, hs::scale, :] = image[wf:wf + wc, hf:hf + hc, :]
            # print('hf: ', hf)
            hf = hf + hc
        wf = wf + wc
        # print('wf: ', wf)
    return real

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


