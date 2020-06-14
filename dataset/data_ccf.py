import os
import os.path
import random

import cv2
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from .data_transforms import RandomRotate
from PIL import Image


# ###### Data loading #######
def make_dataset(root, lst):
    # append all index
    fid = open(lst, 'r')
    imgs, segs = [], []
    for line in fid.readlines():
        idx = line.strip().split(' ')[0]
        image_path = os.path.join(root, 'JPEGImages/' + str(idx) + '.jpg')
        seg_path = os.path.join(root, 'Segmentations/' + str(idx) + '.png')
        imgs.append(image_path)
        segs.append(seg_path)
    return imgs, segs


# ###### val resize & crop ######
def scale_crop(img, seg, crop_size):
    oh, ow = seg.shape
    pad_h = max(crop_size - oh, 0)
    pad_ht, pad_hb = pad_h // 2, pad_h - pad_h // 2
    pad_w = max(crop_size - ow, 0)
    pad_wl, pad_wr = pad_w // 2, pad_w - pad_w // 2
    if pad_h > 0 or pad_w > 0:
        img_pad = cv2.copyMakeBorder(img, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                     value=(0.0, 0.0, 0.0))
        seg_pad = cv2.copyMakeBorder(seg, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                     value=(255,))
    else:
        img_pad, seg_pad = img, seg

    return img_pad, seg_pad


class DatasetGenerator(data.Dataset):
    def __init__(self, root, list_path, crop_size, training=True):

        imgs, segs = make_dataset(root, list_path)

        self.root = root
        self.imgs = imgs
        self.segs = segs
        self.crop_size = crop_size
        self.training = training
        self.colorjitter = transforms.ColorJitter(brightness=0.1, contrast=0.5, saturation=0.5, hue=0.1)
        self.random_rotate=RandomRotate(20)

    def __getitem__(self, index):
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        # load data
        name = self.imgs[index].split('/')[-1][:-4]
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        seg = cv2.imread(self.segs[index], cv2.IMREAD_GRAYSCALE)

        if self.training:
            #colorjitter and rotate
            if random.random() < 0.5:
                img = Image.fromarray(img)
                seg = Image.fromarray(seg)
                img = self.colorjitter(img)
                img, seg = self.random_rotate(img, seg)
                img = np.array(img).astype(np.uint8)
                seg = np.array(seg).astype(np.uint8)
            # random scale
            ratio = random.uniform(0.5, 2.0)
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean

            # pad & crop
            img_h, img_w = seg.shape[:2]
            pad_h = max(self.crop_size - img_h, 0)
            pad_ht, pad_hb = pad_h // 2, pad_h - pad_h // 2
            pad_w = max(self.crop_size - img_w, 0)
            pad_wl, pad_wr = pad_w // 2, pad_w - pad_w // 2
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(img, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                seg_pad = cv2.copyMakeBorder(seg, pad_ht, pad_hb, pad_wl, pad_wr, cv2.BORDER_CONSTANT,
                                             value=(255,))
            else:
                img_pad, seg_pad = img, seg

            seg_pad_h, seg_pad_w = seg_pad.shape
            h_off = random.randint(0, seg_pad_h - self.crop_size)
            w_off = random.randint(0, seg_pad_w - self.crop_size)
            img = np.asarray(img_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size], np.float32)
            seg = np.asarray(seg_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size], np.uint8)
            # random mirror
            flip = np.random.choice(2) * 2 - 1
            img = img[:, ::flip, :]
            seg = seg[:, ::flip]
            # Generate target maps
            img = img.transpose((2, 0, 1))
            seg_half = seg.copy()
            seg_half[(seg_half > 0) & (seg_half <= 4)] = 1
            seg_half[seg_half == 5] = 2
            seg_half[(seg_half > 5) & (seg_half <= 7)] = 1
            seg_half[(seg_half > 7) & (seg_half <= 9)] = 2
            seg_half[(seg_half > 9) & (seg_half <= 11)] = 1
            seg_half[seg_half == 12] = 2
            seg_half[(seg_half > 12) & (seg_half <= 15)] = 1
            seg_half[seg_half == 16] = 2
            seg_half[(seg_half > 16) & (seg_half < 255)] = 1
            seg_full = seg.copy()
            seg_full[(seg_full > 0) & (seg_full < 255)] = 1

        else:
            h, w = seg.shape
            max_size = max(w, h)
            ratio = self.crop_size / max_size
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean
            img, seg = scale_crop(img, seg, crop_size=self.crop_size)
            img = img.transpose((2, 0, 1))
            seg_half = seg.copy()
            seg_half[(seg_half > 0) & (seg_half <= 4)] = 1
            seg_half[seg_half == 5] = 2
            seg_half[(seg_half > 5) & (seg_half <= 7)] = 1
            seg_half[(seg_half > 7) & (seg_half <= 9)] = 2
            seg_half[(seg_half > 9) & (seg_half <= 11)] = 1
            seg_half[seg_half == 12] = 2
            seg_half[(seg_half > 12) & (seg_half <= 15)] = 1
            seg_half[seg_half == 16] = 2
            seg_half[(seg_half > 16) & (seg_half < 255)] = 1
            seg_full = seg.copy()
            seg_full[(seg_full > 0) & (seg_full < 255)] = 1

        images = img.copy()
        segmentations = seg.copy()
        segmentations_half = seg_half.copy()
        segmentations_full = seg_full.copy()

        return images, segmentations, segmentations_half, segmentations_full, name

    def __len__(self):
        return len(self.imgs)


class TestGenerator(data.Dataset):

    def __init__(self, root, list_path, crop_size):

        imgs, segs = make_dataset(root, list_path)
        self.root = root
        self.imgs = imgs
        self.segs = segs
        self.crop_size = crop_size

    def __getitem__(self, index):
        # load data
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        name = self.imgs[index].split('/')[-1][:-4]
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        seg = cv2.imread(self.segs[index], cv2.IMREAD_GRAYSCALE)
        ori_size = img.shape

        h, w = seg.shape
        length = max(w, h)
        ratio = self.crop_size / length
        img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        img = np.array(img).astype(np.float32) - mean
        img = img.transpose((2, 0, 1))

        images = img.copy()
        segmentations = seg.copy()

        return images, segmentations, np.array(ori_size), name

    def __len__(self):
        return len(self.imgs)
