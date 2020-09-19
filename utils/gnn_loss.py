from itertools import filterfalse as ifilterfalse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn import BCELoss
import utils.aaf.losses as lossx
class gnn_loss_att(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, adj_matrix, ignore_index=None, only_present=True, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], cls_p=7, cls_h=3, cls_f=2):
        super(gnn_loss_dp, self).__init__()
        self.edge_index = torch.nonzero(adj_matrix)
        self.edge_index_num = self.edge_index.shape[0]
        self.part_list_list = [[] for i in range(cls_p - 1)]
        for i in range(self.edge_index_num):
            self.part_list_list[self.edge_index[i, 1]].append(self.edge_index[i, 0])

        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687,  1.01665831, 1.05470914])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.num_classes = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f
        self.bceloss = torch.nn.BCELoss(reduction='mean')

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        loss=[]
        for i in range(len(preds[0])):
            pred = F.interpolate(input=preds[0][i], size=(h, w), mode='bilinear', align_corners=True)
            loss_ce = self.criterion(pred, targets[0])
            pred = F.softmax(input=pred, dim=1)
            loss.append(loss_ce + lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present))

        # loss = sum(loss)
        # half body
        loss_hb = []
        for i in range(len(preds[1])):
            pred_hb = F.interpolate(input=preds[1][i], size=(h, w), mode='bilinear', align_corners=True)
            pred_hb = F.softmax(input=pred_hb, dim=1)
            loss_hb.append(lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present))
        # loss_hb = sum(loss_hb)


        # full body
        loss_fb = []
        for i in range(len(preds[2])):
            pred_fb = F.interpolate(input=preds[2][i], size=(h, w), mode='bilinear', align_corners=True)
            pred_fb = F.softmax(input=pred_fb, dim=1)
            loss_fb.append(lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present))
        # loss_fb = sum(loss_fb)

        #one hot part
        labels_p = targets[0]
        one_label_p = labels_p.clone().long()
        one_label_p[one_label_p == self.ignore_index] = 0
        one_hot_lab_p = F.one_hot(one_label_p, num_classes=self.num_classes)
        one_hot_pb_list = list(torch.split(one_hot_lab_p, 1, dim=-1))
        for i in range(0, self.num_classes):
            one_hot_pb_list[i] = one_hot_pb_list[i].squeeze(-1)
            # one_hot_pb_list[i][targets[0]==255]=255

        #one hot half
        labels_h = targets[1]
        one_label_h = labels_h.clone().long()
        one_label_h[one_label_h == self.ignore_index] = 0
        one_hot_lab_h = F.one_hot(one_label_h, num_classes=self.cls_h)
        one_hot_hb_list = list(torch.split(one_hot_lab_h, 1, dim=-1))
        for i in range(0, self.cls_h):
            one_hot_hb_list[i] = one_hot_hb_list[i].squeeze(-1)
            # one_hot_hb_list[i][targets[1]==255]=255

        #one hot full
        labels_f = targets[2]
        one_label_f = labels_f.clone().long()
        one_label_f[one_label_f == self.ignore_index] = 0
        one_hot_lab_f = F.one_hot(one_label_f, num_classes=self.cls_f)
        one_hot_fb_list = list(torch.split(one_hot_lab_f, 1, dim=-1))
        for i in range(0, self.cls_f):
            one_hot_fb_list[i] = one_hot_fb_list[i].squeeze(-1)
            # one_hot_fb_list[i][targets[2]==255]=255


        # #
        valid = (targets[0] != self.ignore_index).unsqueeze(1)

        #decomp fh
        target_fh = one_hot_hb_list[2]+(1-one_hot_fb_list[1])*self.ignore_index

        loss_fh_att = []
        for i in range(len(preds[3])):
            pred_fh = F.interpolate(input=preds[3][i], size=(h, w), mode='bilinear', align_corners=True)
            loss_fh_att.append(self.criterion2(pred_fh, target_fh.long()))
        loss_fh_att = sum(loss_fh_att)

        #decomp up
        upper_bg_node = 1-one_hot_hb_list[1]
        upper_parts=[]
        for i in self.upper_part_list:
            upper_parts.append(one_hot_pb_list[i])
        targets_up = torch.stack(upper_parts, dim=1)
        targets_up = targets_up.argmax(dim=1, keepdim=False)
        targets_up[upper_bg_node == 1] = self.ignore_index
        loss_up_att = []
        for i in range(len(preds[4])):
            pred_up = F.interpolate(input=preds[4][i], size=(h, w), mode='bilinear', align_corners=True)
            loss_up_att.append(self.criterion2(pred_up, targets_up))
        loss_up_att = sum(loss_up_att)

        #decomp lp
        lower_bg_node = 1-one_hot_hb_list[2]
        lower_parts = []
        for i in self.lower_part_list:
            lower_parts.append(one_hot_pb_list[i])
        targets_lp = torch.stack(lower_parts, dim=1)
        targets_lp = targets_lp.argmax(dim=1,keepdim=False)
        targets_lp[lower_bg_node==1]=self.ignore_index
        loss_lp_att = []
        for i in range(len(preds[5])):
            pred_lp = F.interpolate(input=preds[5][i], size=(h, w), mode='bilinear', align_corners=True)
            loss_lp_att.append(self.criterion2(pred_lp, targets_lp))
        loss_lp_att = sum(loss_lp_att)


        # comp_f, comp_u, comp_l, bce loss
        com_full_onehot = one_hot_fb_list[1].float().unsqueeze(1)
        com_u_onehot = one_hot_hb_list[1].float().unsqueeze(1)
        com_l_onehot = one_hot_hb_list[2].float().unsqueeze(1)
        loss_com_f_att = []
        loss_com_u_att = []
        loss_com_l_att = []
        for i in range(len(preds[6])):
            pred_com_full = F.interpolate(input=preds[6][i], size=(h, w), mode='bilinear', align_corners=True)
            pred_com_u = F.interpolate(input=preds[7][i], size=(h, w), mode='bilinear', align_corners=True)
            pred_com_l = F.interpolate(input=preds[8][i], size=(h, w), mode='bilinear', align_corners=True)
            loss_com_f_att.append(self.bceloss(pred_com_full[valid], com_full_onehot[valid]))
            loss_com_u_att.append(self.bceloss(pred_com_u[valid], com_u_onehot[valid]))
            loss_com_l_att.append(self.bceloss(pred_com_l[valid], com_l_onehot[valid]))
            
        loss_com_att = sum(loss_com_f_att)+sum(loss_com_u_att)+sum(loss_com_l_att)
        
        # dependency decomposition
        # loss_context_att =[]
        loss_dp_att = []
        for i in range(len(preds[-2])):
            # loss_context = []
            loss_dp = []
            for j in range(self.num_classes-1):
                part_list = self.part_list_list[j]
                parts_onehot = []
                for k in part_list:
                    parts_onehot.append(one_hot_pb_list[k+1])
                parts_bg_node = 1-sum(parts_onehot)
                targets_dp_onehot = torch.stack(parts_onehot, dim=1)
                targets_dp = targets_dp_onehot.argmax(dim=1, keepdim=False)
                targets_dp[parts_bg_node == 1] = self.ignore_index
                pred_dp = F.interpolate(input=preds[-2][i][j], size=(h, w), mode='bilinear', align_corners=True)
                loss_dp.append(self.criterion2(pred_dp, targets_dp))
                
            loss_dp = sum(loss_dp)
            loss_dp_att.append(loss_dp)
        loss_dp_att = sum(loss_dp_att)

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return (sum(loss) + 0.4*sum(loss_hb) + 0.4*sum(loss_fb))/(len(preds[0]))+ 0.4 * loss_dsn + 0.1*(loss_fh_att + loss_up_att + loss_lp_att + loss_dp_att + loss_com_att)/(len(preds[3]))

class gnn_loss_noatt(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, adj_matrix, ignore_index=None, only_present=True, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], cls_p=7, cls_h=3, cls_f=2):
        super(gnn_loss_noatt, self).__init__()
        self.edge_index = torch.nonzero(adj_matrix)
        self.edge_index_num = self.edge_index.shape[0]
        self.part_list_list = [[] for i in range(cls_p - 1)]
        for i in range(self.edge_index_num):
            self.part_list_list[self.edge_index[i, 1]].append(self.edge_index[i, 0])

        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687,  1.01665831, 1.05470914])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.weight)

        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.num_classes = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        loss=[]
        for i in range(len(preds[0])-1):
            pred = F.interpolate(input=preds[0][i], size=(h, w), mode='bilinear', align_corners=True)
            pred = F.softmax(input=pred, dim=1)
            loss.append(lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present))

        #part seg loss final
        pred0 = F.interpolate(input=preds[0][-1], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred0, dim=1)
        #lovasz loss
        lovasz_loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        loss.append(lovasz_loss)
        #ce loss
        loss_ce = self.criterion(pred0, targets[0])
        loss.append(loss_ce)
        loss = sum(loss)


        # half body
        loss_hb = []
        for i in range(len(preds[1])):
            pred_hb = F.interpolate(input=preds[1][i], size=(h, w), mode='bilinear', align_corners=True)
            pred_hb = F.softmax(input=pred_hb, dim=1)
            loss_hb.append(lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present))
        loss_hb = sum(loss_hb)


        # full body
        loss_fb = []
        for i in range(len(preds[2])):
            pred_fb = F.interpolate(input=preds[2][i], size=(h, w), mode='bilinear', align_corners=True)
            pred_fb = F.softmax(input=pred_fb, dim=1)
            loss_fb.append(lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present))
        loss_fb = sum(loss_fb)

        
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])

        return (loss + 0.4 * loss_hb + 0.4 * loss_fb)/len(preds[1]) + 0.4 * loss_dsn

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat_ori(*flatten_probas_ori(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat_ori(*flatten_probas_ori(probas, labels, ignore), classes=classes)
    return loss

def lovasz_softmax_flat_ori(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas_ori(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def lovasz_softmax_flat(preds, targets, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      :param preds: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      :param targets: [P] Tensor, ground truth labels (between 0 and C - 1)
      :param only_present: average only on classes present in ground truth
    """
    if preds.numel() == 0:
        # only void pixels, the gradients should be 0
        return preds * 0.

    C = preds.size(1)
    losses = []
    for c in range(C):
        fg = (targets == c).float()  # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - preds[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_probas(preds, targets, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = preds.size()
    preds = preds.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    targets = targets.view(-1)
    if ignore is None:
        return preds, targets
    valid = (targets != ignore)
    vprobas = preds[valid.nonzero().squeeze()]
    vlabels = targets[valid]
    return vprobas, vlabels

# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def mean(l, ignore_nan=True, empty=0):
    """
    nan mean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def isnan(x):
    return x != x
