import functools

import torch
from torch import nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import SEModule, ContextContrastedModule

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

class ASPPModule(nn.Module):
    """ASPP"""

    def __init__(self, in_dim, out_dim, scale=1):
        super(ASPPModule, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_dim, out_dim, 1, bias=False), InPlaceABNSync(out_dim))

        self.dilation_0 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim), SEModule(out_dim, reduction=16))

        self.dilation_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=6, dilation=6, bias=False),
                                        InPlaceABNSync(out_dim),SEModule(out_dim, reduction=16))

        self.dilation_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=12, dilation=12, bias=False),
                                        InPlaceABNSync(out_dim), SEModule(out_dim, reduction=16))

        self.dilation_3 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=18, dilation=18, bias=False),
                                        InPlaceABNSync(out_dim), SEModule(out_dim, reduction=16))

        self.psaa_conv = nn.Sequential(nn.Conv2d(in_dim + 5 * out_dim, out_dim, 1, padding=0, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, 5, 1, bias=True),
                                        nn.Sigmoid())

        self.project = nn.Sequential(nn.Conv2d(out_dim * 5, out_dim, kernel_size=1, padding=0, bias=False),
                                       InPlaceABNSync(out_dim))

    def forward(self, x):
        # parallel branch
        feat0 = self.dilation_0(x)
        feat1 = self.dilation_1(x)
        feat2 = self.dilation_2(x)
        feat3 = self.dilation_3(x)
        n, c, h, w = feat0.size()
        gp = self.gap(x)

        feat4 = gp.expand(n, c, h, w)
        # psaa
        y1 = torch.cat((x, feat0, feat1, feat2, feat3, feat4), 1)

        psaa_att = self.psaa_conv(y1)

        psaa_att_list = torch.split(psaa_att, 1, dim=1)

        y2 = torch.cat((psaa_att_list[0] * feat0, psaa_att_list[1] * feat1, psaa_att_list[2] * feat2, psaa_att_list[3] * feat3, psaa_att_list[4]*feat4), 1)
        out = self.project(y2)
        return out
