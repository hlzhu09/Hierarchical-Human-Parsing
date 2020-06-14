import functools

import torch
from torch import nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import SEModule, ContextContrastedModule

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, key_dim, value_dim, out_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # xp = self.pool(x)
        xp = x
        # xp = x
        m_batchsize, C, height, width = x.size()
        m_batchsize, C, hp, wp = xp.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(xp).view(m_batchsize, -1, wp*hp)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = xp.view(m_batchsize, -1, wp*hp)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = x+self.gamma*out
        return out

class ASPPModule(nn.Module):
    """ASPP with OC module: aspp + oc context"""

    def __init__(self, in_dim, out_dim):
        super(ASPPModule, self).__init__()

        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_dim, out_dim, 1, bias=False), InPlaceABNSync(out_dim))

        self.dilation_0 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        SEModule(out_dim, reduction=16))

        self.dilation_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=6, dilation=6, bias=False),
                                        InPlaceABNSync(out_dim),
                                        SEModule(out_dim, reduction=16))

        self.dilation_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=12, dilation=12, bias=False),
                                        InPlaceABNSync(out_dim),
                                        SEModule(out_dim, reduction=16))

        self.dilation_3 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=18, dilation=18, bias=False),
                                        InPlaceABNSync(out_dim),
                                        SEModule(out_dim, reduction=16))

        self.psaa_conv = nn.Sequential(nn.Conv2d(in_dim+5 * out_dim, out_dim, 1, padding=0, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, 5, 1, bias=True),
                                        nn.Sigmoid())

        self.project = nn.Sequential(nn.Conv2d(out_dim * 5, out_dim, kernel_size=1, padding=0, bias=False),
                                       InPlaceABNSync(out_dim))
        self.pam0 = PAM_Module(in_dim=out_dim, key_dim=out_dim//8,value_dim=out_dim,out_dim=out_dim)
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

class ASPPModule2(nn.Module):
    """ASPP with OC module: aspp + oc context"""

    def __init__(self, in_dim, out_dim):
        super(ASPPModule2, self).__init__()

        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_dim, out_dim, 1, bias=False), InPlaceABNSync(out_dim))

        self.dilation_0 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        SEModule(out_dim, reduction=16))

        self.dilation_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=6, dilation=6, bias=False),
                                        InPlaceABNSync(out_dim),
                                        SEModule(out_dim, reduction=16))

        self.dilation_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=12, dilation=12, bias=False),
                                        InPlaceABNSync(out_dim),
                                        SEModule(out_dim, reduction=16))

        self.dilation_3 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim),
                                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=18, dilation=18, bias=False),
                                        InPlaceABNSync(out_dim),
                                        SEModule(out_dim, reduction=16))

        self.project = nn.Sequential(nn.Conv2d(out_dim * 5, out_dim, kernel_size=1, padding=0, bias=False),
                                       InPlaceABNSync(out_dim))
        self.pam0 = PAM_Module(in_dim=out_dim, key_dim=out_dim//8,value_dim=out_dim,out_dim=out_dim)
    def forward(self, x):
        # parallel branch
        feat0 = self.dilation_0(x)
        feat1 = self.dilation_1(x)
        feat2 = self.dilation_2(x)
        feat3 = self.dilation_3(x)
        n, c, h, w = feat0.size()
        gp = self.gap(x)
        feat4 = gp.expand(n, c, h, w)
        y2 = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        out = self.project(y2)
        out = self.pam0(out)
        return out

class MagicModule(nn.Module):
    """ASPP based on SE and OC and Context Contrasted """

    def __init__(self, in_dim, out_dim, scale):
        super(MagicModule, self).__init__()
        self.atte_branch = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, dilation=1, bias=False),
                                         InPlaceABNSync(out_dim),
                                         SelfAttentionModule(in_dim=out_dim, out_dim=out_dim, key_dim=out_dim // 2,
                                                             value_dim=out_dim, scale=scale))
        # TODO: change SE Module to Channel Attention Module
        self.dilation_x = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                        InPlaceABNSync(out_dim), SEModule(out_dim, reduction=16))

        # self.dilation_x = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
        #                                 InPlaceABNSync(out_dim), ChannelAttentionModule(out_dim))

        self.dilation_0 = nn.Sequential(ContextContrastedModule(in_dim, out_dim, rate=6),
                                        SEModule(out_dim, reduction=16))

        self.dilation_1 = nn.Sequential(ContextContrastedModule(in_dim, out_dim, rate=12),
                                        SEModule(out_dim, reduction=16))

        self.dilation_2 = nn.Sequential(ContextContrastedModule(in_dim, out_dim, rate=18),
                                        SEModule(out_dim, reduction=16))

        self.dilation_3 = nn.Sequential(ContextContrastedModule(in_dim, out_dim, rate=24),
                                        SEModule(out_dim, reduction=16))

        self.head_conv = nn.Sequential(nn.Conv2d(out_dim * 6, out_dim, kernel_size=1, padding=0, bias=False),
                                       InPlaceABNSync(out_dim),
                                       nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                       InPlaceABNSync(out_dim))

    def forward(self, x):
        # parallel branch
        feat0 = self.atte_branch(x)
        feat1 = self.dilation_0(x)
        feat2 = self.dilation_1(x)
        feat3 = self.dilation_2(x)
        feat4 = self.dilation_3(x)
        featx = self.dilation_x(x)
        # fusion branch
        concat = torch.cat([feat0, feat1, feat2, feat3, feat4, featx], 1)
        output = self.head_conv(concat)
        return output

class SelfAttentionModule(nn.Module):
    """The basic implementation for self-attention block/non-local block
    Parameters:
        in_dim       : the dimension of the input feature map
        key_dim      : the dimension after the key/query transform
        value_dim    : the dimension after the value transform
        scale        : choose the scale to downsample the input feature maps (save memory cost)
    """

    def __init__(self, in_dim, out_dim, key_dim, value_dim, scale=2):
        super(SelfAttentionModule, self).__init__()
        self.scale = scale
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.func_key = nn.Sequential(nn.Conv2d(in_channels=self.in_dim, out_channels=self.key_dim,
                                                kernel_size=1, stride=1, padding=0, bias=False),
                                      InPlaceABNSync(self.key_dim))
        self.func_query = self.func_key
        self.func_value = nn.Conv2d(in_channels=self.in_dim, out_channels=self.value_dim,
                                    kernel_size=1, stride=1, padding=0)
        self.weights = nn.Conv2d(in_channels=self.value_dim, out_channels=self.out_dim,
                                 kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.weights.weight, 0)
        nn.init.constant_(self.weights.bias, 0)

        self.refine = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, bias=False),
                                    InPlaceABNSync(out_dim))

    def forward(self, x):
        batch, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.func_value(x).view(batch, self.value_dim, -1)  # bottom
        value = value.permute(0, 2, 1)
        query = self.func_query(x).view(batch, self.key_dim, -1)  # top
        query = query.permute(0, 2, 1)
        key = self.func_key(x).view(batch, self.key_dim, -1)  # mid

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_dim ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch, self.value_dim, *x.size()[2:])
        context = self.weights(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        output = self.refine(context)
        return output

