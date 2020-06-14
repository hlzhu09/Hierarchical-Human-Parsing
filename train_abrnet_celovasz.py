import argparse
import os
import random
import sys
import time

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.nn.parallel.scatter_gather import gather
from torch.utils import data
from dataset.data_pascal import DatasetGenerator
from network.baseline import get_model
from utils.lovasz_loss import ABRLovaszCELoss as ABRLovaszLoss
from utils.metric import *
from utils.parallel import DataParallelModel, DataParallelCriterion
from utils.visualize import inv_preprocess, decode_predictions


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation')
    parser.add_argument('--method', type=str, default='abr')
    # Datasets
    parser.add_argument('--root', default='./data/Person', type=str)
    parser.add_argument('--val-root', default='./data/Person', type=str)
    parser.add_argument('--lst', default='./dataset/Pascal/train_id.txt', type=str)
    parser.add_argument('--val-lst', default='./dataset/Pascal/val_id.txt', type=str)
    parser.add_argument('--crop-size', type=int, default=473)
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--hbody-cls', type=int, default=3)
    parser.add_argument('--fbody-cls', type=int, default=2)
    # Optimization options
    parser.add_argument('--epochs', default=151, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--learning-rate', default=7e-3, type=float)
    parser.add_argument('--lr-mode', type=str, default='poly')
    parser.add_argument('--ignore-label', type=int, default=255)
    # Checkpoints
    parser.add_argument('--restore-from', default='./checkpoints/init/resnet101_stem.pth', type=str)
    parser.add_argument('--snapshot_dir', type=str, default='./checkpoints/exp/')
    parser.add_argument('--log-dir', type=str, default='./runs/')
    parser.add_argument('--init', action="store_true")
    parser.add_argument('--save-num', type=int, default=4)
    # Misc
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, i_iter, iters_per_epoch, method='poly'):
    if method == 'poly':
        current_step = epoch * iters_per_epoch + i_iter
        max_step = args.epochs * iters_per_epoch
        lr = args.learning_rate * ((1 - current_step / max_step) ** 0.9)
    else:
        lr = args.learning_rate
    optimizer.param_groups[0]['lr'] = lr
    return lr


def main(args):
    # initialization
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.method))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    # conduct seg network
    seg_model = get_model(num_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)
    new_params = seg_model.state_dict().copy()

    if args.init:
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                new_params['encoder.' + '.'.join(i_parts[:])] = saved_state_dict[i]
        seg_model.load_state_dict(new_params)
        print('loading params w/o fc')
    else:
        seg_model.load_state_dict(saved_state_dict)
        print('loading params all')

    model = DataParallelModel(seg_model)
    model.float()
    model.cuda()

    # define dataloader
    train_loader = data.DataLoader(DatasetGenerator(root=args.root, list_path=args.lst,
                                                    crop_size=args.crop_size, training=True),
                                   batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = data.DataLoader(DatasetGenerator(root=args.val_root, list_path=args.val_lst,
                                                  crop_size=args.crop_size, training=False),
                                 batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # define criterion & optimizer
    criterion = ABRLovaszLoss(ignore_index=args.ignore_label, only_present=True)
    criterion = DataParallelCriterion(criterion).cuda()

    optimizer = optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, seg_model.parameters()), 'lr': args.learning_rate}],
        lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    # key points
    best_val_mIoU = 0
    best_val_pixAcc = 0
    start = time.time()

    for epoch in range(0, args.epochs):
        print('\n{} | {}'.format(epoch, args.epochs - 1))
        # training
        _ = train(model, train_loader, epoch, criterion, optimizer, writer)

        # validation
        if epoch %10 ==0 or epoch > args.epochs-5:
            val_pixacc, val_miou = validation(model, val_loader, epoch, writer)
            # save model
            if val_pixacc > best_val_pixAcc:
                best_val_pixAcc = val_pixacc
            if val_miou > best_val_mIoU:
                best_val_mIoU = val_miou
                model_dir = os.path.join(args.snapshot_dir, args.method + '_miou.pth')
                torch.save(seg_model.state_dict(), model_dir)
                print('Model saved to %s' % model_dir)

    os.rename(model_dir, os.path.join(args.snapshot_dir, args.method + '_miou'+str(best_val_mIoU)+'.pth'))
    print('Complete using', time.time() - start, 'seconds')
    print('Best pixAcc: {} | Best mIoU: {}'.format(best_val_pixAcc, best_val_mIoU))


def train(model, train_loader, epoch, criterion, optimizer, writer):
    # set training mode
    model.train()
    train_loss = 0.0
    iter_num = 0

    # Iterate over data.
    from tqdm import tqdm
    tbar = tqdm(train_loader)
    for i_iter, batch in enumerate(tbar):
        sys.stdout.flush()
        start_time = time.time()
        iter_num += 1
        # adjust learning rate
        iters_per_epoch = len(train_loader)
        lr = adjust_learning_rate(optimizer, epoch, i_iter, iters_per_epoch, method=args.lr_mode)
        image, label, hlabel, flabel, _ = batch
        images, labels, hlabel, flabel = image.cuda(), label.long().cuda(), hlabel.cuda(), flabel.cuda()
        torch.set_grad_enabled(True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # compute output loss
        preds = model(images)
        loss = criterion(preds, [labels, hlabel, flabel])  # batch mean
        train_loss += loss.item()

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        if i_iter % 10 == 0:
            writer.add_scalar('learning_rate', lr, iter_num + epoch * len(train_loader))
            writer.add_scalar('train_loss', train_loss / iter_num, iter_num + epoch * len(train_loader))

        batch_time = time.time() - start_time
        # plot progress
        tbar.set_description('{} / {} | Time: {batch_time:.4f} | Loss: {loss:.4f}'.format(iter_num, len(train_loader),
                                                                                  batch_time=batch_time,
                                                                                  loss=train_loss / iter_num))

    epoch_loss = train_loss / iter_num
    writer.add_scalar('train_epoch_loss', epoch_loss, epoch)
    tbar.close()

    return epoch_loss


def validation(model, val_loader, epoch, writer):
    # set evaluate mode
    model.eval()

    total_correct, total_label = 0, 0
    total_correct_hb, total_label_hb = 0, 0
    total_correct_fb, total_label_fb = 0, 0
    hist = np.zeros((args.num_classes, args.num_classes))
    hist_hb = np.zeros((args.hbody_cls, args.hbody_cls))
    hist_fb = np.zeros((args.fbody_cls, args.fbody_cls))

    # Iterate over data.
    from tqdm import tqdm
    tbar = tqdm(val_loader)
    for idx, batch in enumerate(tbar):
        image, target, hlabel, flabel, _ = batch
        image, target, hlabel, flabel = image.cuda(), target.cuda(), hlabel.cuda(), flabel.cuda()
        with torch.no_grad():
            h, w = target.size(1), target.size(2)
            outputs = model(image)
            outputs = gather(outputs, 0, dim=0)
            preds = F.interpolate(input=outputs[0], size=(h, w), mode='bilinear', align_corners=True)
            preds_hb = F.interpolate(input=outputs[1], size=(h, w), mode='bilinear', align_corners=True)
            preds_fb = F.interpolate(input=outputs[2], size=(h, w), mode='bilinear', align_corners=True)
            if idx % 50 == 0:
                img_vis = inv_preprocess(image, num_images=args.save_num)
                label_vis = decode_predictions(target.int(), num_images=args.save_num, num_classes=args.num_classes)
                pred_vis = decode_predictions(torch.argmax(preds, dim=1), num_images=args.save_num,
                                              num_classes=args.num_classes)

                # visual grids
                img_grid = torchvision.utils.make_grid(torch.from_numpy(img_vis.transpose(0, 3, 1, 2)))
                label_grid = torchvision.utils.make_grid(torch.from_numpy(label_vis.transpose(0, 3, 1, 2)))
                pred_grid = torchvision.utils.make_grid(torch.from_numpy(pred_vis.transpose(0, 3, 1, 2)))
                writer.add_image('val_images', img_grid, epoch * len(val_loader) + idx + 1)
                writer.add_image('val_labels', label_grid, epoch * len(val_loader) + idx + 1)
                writer.add_image('val_preds', pred_grid, epoch * len(val_loader) + idx + 1)

            # pixelAcc
            correct, labeled = batch_pix_accuracy(preds.data, target)
            correct_hb, labeled_hb = batch_pix_accuracy(preds_hb.data, hlabel)
            correct_fb, labeled_fb = batch_pix_accuracy(preds_fb.data, flabel)
            # mIoU
            hist += fast_hist(preds, target, args.num_classes)
            hist_hb += fast_hist(preds_hb, hlabel, args.hbody_cls)
            hist_fb += fast_hist(preds_fb, flabel, args.fbody_cls)

            total_correct += correct
            total_correct_hb += correct_hb
            total_correct_fb += correct_fb
            total_label += labeled
            total_label_hb += labeled_hb
            total_label_fb += labeled_fb
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = round(np.nanmean(per_class_iu(hist)) * 100, 2)
            pixAcc_hb = 1.0 * total_correct_hb / (np.spacing(1) + total_label_hb)
            IoU_hb = round(np.nanmean(per_class_iu(hist_hb)) * 100, 2)
            pixAcc_fb = 1.0 * total_correct_fb / (np.spacing(1) + total_label_fb)
            IoU_fb = round(np.nanmean(per_class_iu(hist_fb)) * 100, 2)
            # plot progress
            tbar.set_description('{} / {} | {pixAcc:.4f}, {IoU:.4f} |' \
                         '{pixAcc_hb:.4f}, {IoU_hb:.4f} |' \
                         '{pixAcc_fb:.4f}, {IoU_fb:.4f}'.format(idx + 1, len(val_loader), pixAcc=pixAcc, IoU=IoU,pixAcc_hb=pixAcc_hb, IoU_hb=IoU_hb,pixAcc_fb=pixAcc_fb, IoU_fb=IoU_fb))


    print('\n per class iou part: {}'.format(per_class_iu(hist)*100))
    print('per class iou hb: {}'.format(per_class_iu(hist_hb)*100))
    print('per class iou fb: {}'.format(per_class_iu(hist_fb)*100))

    mIoU = round(np.nanmean(per_class_iu(hist)) * 100, 2)
    mIoU_hb = round(np.nanmean(per_class_iu(hist_hb)) * 100, 2)
    mIoU_fb = round(np.nanmean(per_class_iu(hist_fb)) * 100, 2)

    writer.add_scalar('val_pixAcc', pixAcc, epoch)
    writer.add_scalar('val_mIoU', mIoU, epoch)
    writer.add_scalar('val_pixAcc_hb', pixAcc_hb, epoch)
    writer.add_scalar('val_mIoU_hb', mIoU_hb, epoch)
    writer.add_scalar('val_pixAcc_fb', pixAcc_fb, epoch)
    writer.add_scalar('val_mIoU_fb', mIoU_fb, epoch)
    tbar.close()
    return pixAcc, mIoU


if __name__ == '__main__':
    args = parse_args()
    main(args)
