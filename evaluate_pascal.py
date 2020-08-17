import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data

from dataset.data_pascal import TestGenerator
# from network.fullfusenet import get_model
from network.abrnet2 import get_model


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Pytorch Segmentation")
    parser.add_argument('--root', default='./data/Person', type=str)
    parser.add_argument("--data-list", type=str, default='./dataset/Pascal/val_id.txt')
    parser.add_argument("--crop-size", type=int, default=473)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument('--restore-from', default='./checkpoints/exp/baseline_pascal.pth', type=str)

    parser.add_argument("--is-mirror", action="store_true")
    parser.add_argument('--eval-scale', nargs='+', type=float, default=[1.0])
    # parser.add_argument('--eval-scale', nargs='+', type=float, default=[0.50, 0.75, 1.0, 1.25, 1.50])
    # parser.add_argument('--eval-scale', nargs='+', type=float, default=[0.50, 0.75, 1.0, 1.25, 1.50, 1.75])

    parser.add_argument("--save-dir", type=str)
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

    model = get_model(num_classes=args.num_classes)

    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)

    palette = get_lip_palette()
    restore_from = args.restore_from
    saved_state_dict = torch.load(restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    testloader = data.DataLoader(TestGenerator(args.root, args.data_list, crop_size=args.crop_size),
                                 batch_size=1, shuffle=False, pin_memory=True)

    confusion_matrix = np.zeros((args.num_classes, args.num_classes))

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d images have been proceeded' % index)
        image, label, ori_size, name = batch

        ori_size = ori_size[0].numpy()

        output = predict(model, image.numpy(), (np.asscalar(ori_size[0]), np.asscalar(ori_size[1])),
                         is_mirror=args.is_mirror, scales=args.eval_scale)
        seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        # output_im = PILImage.fromarray(seg_pred)
        # output_im.putpalette(palette)
        # output_im.save(args.save_dir + name[0] + '.png')

        seg_gt = np.asarray(label[0].numpy(), dtype=np.int)
        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]

        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = tp.sum() / pos.sum()
    mean_accuracy = (tp / np.maximum(1.0, pos)).mean()
    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()

    # get_confusion_matrix_plot()

    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IU)
    for index, IU in enumerate(IU_array):
        print('%f ', IU)


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
                prediction = interp(prediction[0]).cpu().data.numpy()

            prediction_rev = prediction[1, :, :, :].copy()
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
                prediction = interp(prediction[0]).cpu().data.numpy()
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


def get_confusion_matrix_plot(conf_arr):
    norm_conf = []
    for i in conf_arr:
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / max(1.0, float(a)))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest')

    width, height = conf_arr.shape

    cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('confusion_matrix.png', format='png')


def get_lip_palette():
    palette = [0, 0, 0,
               128, 0, 0,
               255, 0, 0,
               0, 85, 0,
               170, 0, 51,
               255, 85, 0,
               0, 0, 85,
               0, 119, 221,
               85, 85, 0,
               0, 85, 85,
               85, 51, 0,
               52, 86, 128,
               0, 128, 0,
               0, 0, 255,
               51, 170, 221,
               0, 255, 255,
               85, 255, 170,
               170, 255, 85,
               255, 255, 0,
               255, 170, 0]
    return palette


if __name__ == '__main__':
    main()
