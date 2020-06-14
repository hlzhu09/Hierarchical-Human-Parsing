import os
import shutil

import cv2
import numpy as np
from PIL import Image


def flip_labels():
    """
   1.Hat 2.Hair 3.Glove 4.Sunglasses 5.UpperClothes 6.Dress 7.Coat 8.Socks 9.Pants 10.Jumpsuits
   11.Scarf 12.Skirt 13.Face 14.Left-arm 15.Right-arm 16.Left-leg 17.Right-leg 18.Left-shoe 19.Right-shoe

    left-right (14, 15) (16, 17) (18, 19)
    """
    target_root = '/home/ubuntu/Data/CIHP/Segmentations/'
    save_root = '/home/ubuntu/Data/CIHP/Segmentations_rev/'
    fid_lst = open('../dataset/CIHP/all_id.txt', 'r')

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for count, idx in enumerate(fid_lst.readlines()):
        name = idx.strip()
        label_path = os.path.join(target_root, name + '.png')
        label = np.array(Image.open(label_path).convert('L'))
        label_rev = label[:, ::-1].copy()
        label_rev[label_rev == 14] = 255
        label_rev[label_rev == 15] = 14
        label_rev[label_rev == 255] = 15

        label_rev[label_rev == 16] = 255
        label_rev[label_rev == 17] = 16
        label_rev[label_rev == 255] = 17

        label_rev[label_rev == 18] = 255
        label_rev[label_rev == 19] = 18
        label_rev[label_rev == 255] = 19
        rev_img = Image.fromarray(label_rev)
        rev_img.save(os.path.join(save_root, name + '.png'))
        print('{} imgs have been processed'. format(count + 1))
    print('Done!')


def cap_lst():
    name = []
    lst = open('/home/jzzz/Proj/ParsingPose/ParseNet/dataset/LIP/val_id.txt', 'r')
    for line in lst.readlines():
        idx = line.strip()
        img_path = 'val_set/images/' + idx + '.jpg'
        label_path = 'val_set/segmentations/' + idx + '.png'
        rev_path = 'val_set/segmentations_rev/' + idx + '.png'
        name.append([img_path, label_path, rev_path])

    target = open('./val.txt', 'w')
    for item in name:
        print(' '.join(item), file=target)


def pro_img():
    ref = np.array(Image.open('/home/jzzz/Proj/PytorchProj/MagNet/results/x.png').convert('RGB'))
    target = np.array(Image.open('/home/jzzz/Proj/PytorchProj/MagNet/results/y.png').convert('RGB'))
    target[200:300, :, :] = ref[200:300, :, :]
    new_img = Image.fromarray(target)
    new_img.save('/home/jzzz/Proj/PytorchProj/MagNet/results/base.png')


def ensemble_prediction():
    # data path
    root_part1 = '../results/trainval_results/'
    root_part2 = '../results/val_fine_results/'
    root_part3 = '../results/513_results/'
    # lst path
    lst = open('../dataset/LIP/test_id.txt', 'r')
    # save root
    save_path = '../results/ensemble_results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # iterations
    for count, line in enumerate(lst.readlines()):
        name = line.strip()
        pred_part1 = np.array(Image.open(os.path.join(root_part1, name + '.png')))
        pred_part2 = np.array(Image.open(os.path.join(root_part2, name + '.png')))
        pred_part3 = np.array(Image.open(os.path.join(root_part3, name + '.png')))
        pred_ensemble = (pred_part1 + pred_part2 + pred_part3) // 3
        ensemble_results = Image.fromarray(pred_ensemble.astype(np.uint8))
        ensemble_results.save(os.path.join(save_path, name + '.png'))
        print('{} images have been processed'.format(count))
    print('Done!')


def select_imgs():
    source_root = '/home/jzzz/Documents/JzzZ/ICCV/TODO_PPSS/Labels/'
    target_root = '/home/jzzz/Documents/JzzZ/ICCV/TODO_PPSS/Labelsss/'
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    fid_lst = open('../dataset/PPSS/select_id.txt')
    for count, idx in enumerate(fid_lst.readlines()):
        img_id = idx.strip()
        source_path = os.path.join(source_root, img_id + '.png')
        target_path = os.path.join(target_root, img_id + '.png')
        shutil.copy(source_path, target_path)
        print('--> {} targets'.format(count))
    print('Done!')


def weighted_vis():
    fid = open('../dataset/PPSS/select_id.txt', 'r')
    image_root = '/home/jzzz/Documents/JzzZ/ICCV/TODO_PPSS/Images/'
    mask_root = '/home/jzzz/Documents/JzzZ/ICCV/TODO_PPSS/Labels/'
    save_roota = '/home/jzzz/Documents/JzzZ/ICCV/TODO_PPSS/Labels/E-Masks/'
    save_rootb = '/home/jzzz/Documents/JzzZ/ICCV/TODO_PPSS/Labels/W-Masks/'
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    if not os.path.exists(save_roota):
        os.makedirs(save_roota)

    if not os.path.exists(save_rootb):
        os.makedirs(save_rootb)

    for count, line in enumerate(fid.readlines()):
        idx = line.strip()
        image_data = np.array(Image.open(os.path.join(image_root, idx + '.jpg')).convert('RGB'))
        mask_data = np.array(Image.open(os.path.join(mask_root, idx + '.png')).convert('RGB'))
        edge_data = np.array(Image.open(os.path.join(mask_root, idx + '.png')).convert('L'))
        img_h, img_w = image_data.shape[:2]
        detec_edge = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        for i in range(1, img_h - 1):
            for j in range(1, img_w - 1):
                if edge_data[i, j] != edge_data[i - 1, j] or edge_data[i, j] != edge_data[i + 1, j] \
                        or edge_data[i, j] != edge_data[i, j - 1] or edge_data[i, j] != edge_data[i, j + 1]:
                    detec_edge[i, j, 0] = mask_data[i, j, 0]
                    detec_edge[i, j, 1] = mask_data[i, j, 1]
                    detec_edge[i, j, 2] = mask_data[i, j, 2]
        dilated_edge = detec_edge.copy()
        dilated_edge[:, :, 0] = cv2.dilate(dilated_edge[:, :, 0], kernel=kernel)
        dilated_edge[:, :, 1] = cv2.dilate(dilated_edge[:, :, 1], kernel=kernel)
        dilated_edge[:, :, 2] = cv2.dilate(dilated_edge[:, :, 2], kernel=kernel)
        # ENHANCE THE EDGE MAP
        enhanced_mask = cv2.addWeighted(mask_data, 0.7, dilated_edge, 0.3, 0.0)
        processed_mask = Image.fromarray(enhanced_mask)
        processed_mask.save(os.path.join(save_roota, idx + '.png'))
        weighted_mask = cv2.addWeighted(image_data, 0.3, enhanced_mask, 0.7, 0.0)
        processed_img = Image.fromarray(weighted_mask)
        processed_img.save(os.path.join(save_rootb, idx + '.png'))
        print('{} items have been processed'.format(count + 1))
    print('Done!')


if __name__ == '__main__':
    flip_labels()
