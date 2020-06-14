import os
import os.path
import random

import cv2
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from .data_transforms import RandomRotate
from PIL import Image
map_idx = [0, 9, 19, 29, 50, 39, 60, 62]
# 0background, 1hair, 2face, 3upper clothes, 4arms, 5lower clothes, 6legs, 7shoes


# ###### Data loading #######
def make_dataset(root, lst):
    # append all index
    fid = open(lst, 'r')
    imgs, segs = [], []
    for line in fid.readlines():
        idx = line.strip()
        image_path = os.path.join(root, str(idx) + '.jpg')
        seg_path = os.path.join(root, str(idx) + '_m.png')
        imgs.append(image_path)
        segs.append(seg_path)
    return imgs, segs


# ###### val resize & crop ######
def scale_crop(img, seg, crop_size):
    oh, ow = seg.shape
    pad_h = max(0, crop_size[0] - oh)
    pad_w = max(0, crop_size[1] - ow)
    if pad_h > 0 or pad_w > 0:
        img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                     value=(0.0, 0.0, 0.0))
        seg_pad = cv2.copyMakeBorder(seg, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                     value=255)
    else:
        img_pad, seg_pad = img, seg

    img = np.asarray(img_pad[0: crop_size[0], 0: crop_size[1]], np.float32)
    seg = np.asarray(seg_pad[0: crop_size[0], 0: crop_size[1]], np.float32)

    return img, seg


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
        # load data
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        name = self.imgs[index].split('/')[-1][:-4]
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        seg = np.array(Image.open(self.segs[index]))
        # seg_h, seg_w = seg.shape
        seg_h, seg_w, _ = img.shape
        # img = cv2.resize(img, (seg_w, seg_h), interpolation=cv2.INTER_LINEAR)
        seg = cv2.resize(seg, (seg_w, seg_h), interpolation=cv2.INTER_NEAREST)
        new_seg = (np.ones_like(seg)*255).astype(np.uint8)
        for i in range(len(map_idx)):
            new_seg[seg == map_idx[i]] = i
        seg = new_seg
        if self.training:
            #colorjitter and rotate
            if random.random() < 0.5:
                img = Image.fromarray(img)
                seg = Image.fromarray(seg)
                img = self.colorjitter(img)
                img, seg = self.random_rotate(img, seg)
                img = np.array(img).astype(np.uint8)
                seg = np.array(seg).astype(np.uint8)
            # random mirror
            flip = np.random.choice(2) * 2 - 1
            img = img[:, ::flip, :]
            seg = seg[:, ::flip]
            # random scale
            ratio = random.uniform(0.75, 2.5)
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean

            # pad & crop
            img_h, img_w = seg.shape
            assert img_w < img_h
            pad_h = max(self.crop_size[0] - img_h, 0)
            pad_w = max(self.crop_size[1] - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                seg_pad = cv2.copyMakeBorder(seg, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                             value=(255,))
            else:
                img_pad, seg_pad = img, seg

            img_h, img_w = seg_pad.shape
            h_off = random.randint(0, img_h - self.crop_size[0])
            w_off = random.randint(0, img_w - self.crop_size[1])
            img = np.asarray(img_pad[h_off: h_off + self.crop_size[0], w_off: w_off + self.crop_size[1]], np.float32)
            seg = np.asarray(seg_pad[h_off: h_off + self.crop_size[0], w_off: w_off + self.crop_size[1]], np.uint8)
            img = img.transpose((2, 0, 1))
            # generate body masks
            seg_half = seg.copy()
            seg_half[(seg_half > 0) & (seg_half <= 4)] = 1
            seg_half[(seg_half > 4) & (seg_half < 255)] = 2
            seg_full = seg.copy()
            seg_full[(seg_full > 0) & (seg_full < 255)] = 1

        else:
            h, w = seg.shape
            max_size = max(w, h)
            ratio = self.crop_size[0] / max_size
            img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            img = np.array(img).astype(np.float32) - mean
            img, seg = scale_crop(img, seg, crop_size=self.crop_size)
            img = img.transpose((2, 0, 1))
            # generate body masks
            seg_half = seg.copy()
            seg_half[(seg_half > 0) & (seg_half <= 4)] = 1
            seg_half[(seg_half > 4) & (seg_half < 255)] = 2
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
        seg = np.array(Image.open(self.segs[index]))
        seg_h, seg_w = seg.shape
        img = cv2.resize(img, (seg_w, seg_h), interpolation=cv2.INTER_LINEAR)
        for i in range(len(map_idx)):
            seg[seg == map_idx[i]] = i
        ori_size = img.shape

        h, w = seg.shape
        length = max(w, h)
        ratio = self.crop_size[0] / length
        img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        img = np.array(img).astype(np.float32) - mean
        img = img.transpose((2, 0, 1))

        images = img.copy()
        segmentations = seg.copy()

        return images, segmentations, np.array(ori_size), name

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    dl = DatasetGenerator('/media/jzzz/Data/Dataset/PPSS/TrainData/', './PPSS/train_id.txt',
                          crop_size=(321, 161), training=False)

    item = iter(dl)
    for i in range(len(dl)):
        imgs, segs, segs_half, segs_full, idx = next(item)
        pass
