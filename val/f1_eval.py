import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils import data

from dataset.data_ccf import TestGenerator
from network.baseline import get_model


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Pytorch Segmentation")
    parser.add_argument('--root', default='./data/CCF', type=str)
    parser.add_argument("--data-list", type=str, default='./dataset/CCF/test_id.txt')
    parser.add_argument("--crop-size", type=int, default=513)
    parser.add_argument("--num-classes", type=int, default=18)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument('--restore-from', default='./checkpoints/exp/model_best.pth', type=str)
    parser.add_argument("--is-mirror", action="store_true")
    parser.add_argument('--eval-scale', nargs='+', type=float, default=[1.0])
    # parser.add_argument('--eval-scale', nargs='+', type=float, default=[0.50, 0.75, 1.0, 1.25, 1.50, 1.75])
    parser.add_argument("--save-dir", type=str, default='./runs')
    parser.add_argument("--gpu", type=str, default='0')
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    # initialization
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)

    # obtain the color map
    palette = get_atr_palette()

    # conduct model & load pre-trained weights
    model = get_model(num_classes=args.num_classes)
    restore_from = args.restore_from
    saved_state_dict = torch.load(restore_from)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda()
    # data loader
    testloader = data.DataLoader(TestGenerator(args.root, args.data_list, crop_size=args.crop_size),
                                 batch_size=1, shuffle=False, pin_memory=True)

    confusion_matrix = np.zeros((args.num_classes, args.num_classes))

    for index, batch in enumerate(testloader):
        if index % 10 == 0:
            print('%d images have been proceeded' % index)
        image, label, ori_size, name = batch

        ori_size = ori_size[0].numpy()
        output = predict(model, image.numpy(), (np.asscalar(ori_size[0]), np.asscalar(ori_size[1])),
                         is_mirror=args.is_mirror, scales=args.eval_scale)
        seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        # output_im = PILImage.fromarray(seg_pred)
        # output_im.putpalette(palette)
        # output_im.save(args.save_dir + name[0] + '.png')

        # seg_gt = np.asarray(label[0].numpy(), dtype=np.int)
        # ignore_index = seg_gt != 255
        # seg_gt = seg_gt[ignore_index]
        # seg_pred = seg_pred[ignore_index]
        #
        # confusion_matrix = get_confusion_matrix(seg_gt, seg_pred, args.num_classes)
        #
        # pos = confusion_matrix.sum(1)
        # res = confusion_matrix.sum(0)
        # tp = np.diag(confusion_matrix)
        # non_zero_idx = res != 0
        # pos_idx = pos[non_zero_idx]
        # res_idx = res[non_zero_idx]
        # tp_idx = tp[non_zero_idx]
        #
        # iou_array = tp_idx / np.maximum(1.0, pos_idx + res_idx - tp_idx)
        # mean_iou = np.nanmean(iou_array)
        #
        # output_im = Image.fromarray(seg_pred)
        # output_im.putpalette(palette)
        # output_im.save(os.path.join(args.save_dir, str(mean_iou)[2:5] + '_' + name[0] + '.png'))

        seg_gt = np.asarray(label[0].numpy(), dtype=np.int)
        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]

        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)

    pos = confusion_matrix.sum(1)  # TP + FP
    res = confusion_matrix.sum(0)  # p
    tp = np.diag(confusion_matrix)

    pixel_accuracy = tp.sum() / pos.sum()  # mean Acc
    fg_accuracy = tp[1:].sum() / pos[1:].sum()  # foreground Acc
    mean_accuracy = (tp / np.maximum(1.0, pos)).mean()  # mean Precision
    # cal_list = 0
    # for i in range(len(res)):
    #     if res[i] != 0:
    #         cal_list += (tp[i] / res[i])
    # mean_recall = cal_list / len(res)  # mean Recall
    mean_recall = (tp / res).mean()  # mean Recall
    f1_score = 2 * mean_accuracy * mean_recall / (mean_accuracy + mean_recall)
    accuracy = (tp / np.maximum(1.0, pos))
    recall = (tp / res)
    cls_f1_score = 2 * accuracy * recall / (accuracy + recall)

    # mean_recall = (tp / (res+1e-10)).mean()  # mean Recall
    # f1_score = 2 * mean_accuracy * mean_recall / (mean_accuracy + mean_recall + 1e-10)
    # accuracy = (tp / (np.maximum(1.0, pos)+1e-10))
    # recall = (tp / (res+1e-10))
    # cls_f1_score = 2 * accuracy * recall / (accuracy + recall + 1e-10)

    print('pixelAcc. --> {}'.format(pixel_accuracy))
    print('foregroundAcc. --> {}'.format(fg_accuracy))
    print('meanPrecision --> {}'.format(mean_accuracy))
    print('meanRecall --> {}'.format(mean_recall))
    print('meanF1-score --> {}'.format(f1_score))
    for i in range(args.num_classes):
        print('cls {}, F1-score --> {}'.format(i, cls_f1_score[i]))


def scale_image(image, scale):
    image = image[0, :, :, :]
    image = image.transpose((1, 2, 0))
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    image = image.transpose((2, 0, 1))
    return image


def predict(net, image, output_size, is_mirror=True, scales=[1]):
    if is_mirror:
        image_rev = image[:, :, :, ::-1]

    interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)

    outputs = []
    if is_mirror:
        for scale in scales:
            if scale != 1:
                image_scale = scale_image(image=image, scale=scale)
                image_rev_scale = scale_image(image=image_rev, scale=scale)
            else:
                image_scale = image[0, :, :, :]
                image_rev_scale = image_rev[0, :, :, :]

            image_scale = np.stack((image_scale, image_rev_scale))

            with torch.no_grad():
                prediction = net(Variable(torch.from_numpy(image_scale)).cuda())
                alpha_pred = interp(prediction[0]).cpu().data.numpy()
                prediction = alpha_pred

            prediction_rev = prediction[1, :, :, :].copy()
            # left right
            # prediction_rev[9, :, :] = prediction[1, 10, :, :]
            # prediction_rev[10, :, :] = prediction[1, 9, :, :]
            # prediction_rev[12, :, :] = prediction[1, 13, :, :]
            # prediction_rev[13, :, :] = prediction[1, 12, :, :]
            # prediction_rev[14, :, :] = prediction[1, 15, :, :]
            # prediction_rev[15, :, :] = prediction[1, 14, :, :]
            prediction_rev = prediction_rev[:, :, ::-1]
            prediction = prediction[0, :, :, :]
            prediction = np.mean([prediction, prediction_rev], axis=0)
            outputs.append(prediction)

        outputs = np.mean(outputs, axis=0)
        outputs = outputs.transpose(1, 2, 0)

    else:
        for scale in scales:
            if scale != 1:
                image_scale = scale_image(image=image, scale=scale)
            else:
                image_scale = image[0, :, :, :]

            with torch.no_grad():
                prediction = net(Variable(torch.from_numpy(image_scale).unsqueeze(0)).cuda())
                alpha_pred = interp(prediction[0]).cpu().data.numpy()
                prediction = alpha_pred

            outputs.append(prediction[0, :, :, :])

        outputs = np.mean(outputs, axis=0)
        outputs = outputs.transpose(1, 2, 0)

    return outputs


def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calculate the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def get_atr_palette():
    palette = [0, 0, 0,
               128, 0, 0,
               0, 128, 0,
               128, 128, 0,
               0, 0, 128,
               128, 0, 128,
               0, 128, 128,
               170, 0, 51,
               255, 85, 0,
               0, 119, 221,
               85, 51, 0,
               52, 86, 128,
               51, 170, 221,
               0, 255, 255,
               85, 255, 170,
               170, 255, 85,
               255, 255, 0,
               255, 170, 0]
    return palette


if __name__ == '__main__':
    main()
