import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import ASPPModule

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvGRU, self).__init__()
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv_can = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, padding=self.padding, stride=1, bias=False)
        )

        nn.init.orthogonal_(self.conv_gates.weight)
        nn.init.constant_(self.conv_gates.bias, 0.)

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, 1, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next
    
class DecoderModule(nn.Module):
    
    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(256, 256, 1, bias=False),
                            nn.ReLU(True),
                            nn.Conv2d(256, 256, 1, bias=True),
                            nn.Sigmoid())
    def forward(self, x):
        out=self.conv0(x)
        out = out + self.se(out)*out
        return out

class Composition(nn.Module):
    def __init__(self, hidden_dim, parts_num):
        super(Composition, self).__init__()
        self.relation = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 2*hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2*hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False),
        )
        self.comp_att = nn.Sequential(
            nn.Conv2d(parts_num * hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, parent, child_list):
        comp_att = self.comp_att(torch.cat(child_list, dim=1))
        comp_message = self.relation(torch.cat([parent, sum(child_list) * comp_att], dim=1))
        return comp_message, comp_att
    
class Decomposition(nn.Module):
    def __init__(self, hidden_dim=10, child_num=2):
        super(Decomposition, self).__init__()
        self.decomp_att = nn.Sequential(
            nn.Conv2d((child_num+1)*hidden_dim, child_num, kernel_size=1, padding=0, stride=1, bias=True)
        )

        self.relation = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 2*hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2*hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
                   )

    def forward(self, parent, child_list, parent_att):
        decomp_map = self.decomp_att(torch.cat([parent]+child_list, dim=1))
        decomp_att = torch.softmax(decomp_map, dim=1)
        decomp_att_list = torch.split(decomp_att, 1, dim=1)
        decomp_list = [self.relation(torch.cat([child_list[i], parent * decomp_att_list[i]*parent_att], dim=1)) for i in
                          range(len(child_list))]
        return decomp_list, decomp_map

def generate_spatial_batch(featmap_H, featmap_W):
    import numpy as np
    spatial_batch_val = np.zeros((1, featmap_H, featmap_W, 8), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w + 1) / featmap_W * 2 - 1
            xctr = (xmin + xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h + 1) / featmap_H * 2 - 1
            yctr = (ymin + ymax) / 2
            spatial_batch_val[:, h, w, :] = \
                [xmin, ymin, xmax, ymax, xctr, yctr, 1 / featmap_W, 1 / featmap_H]
    return spatial_batch_val



class Dep_Context(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, parts_num = 6):
        super(Dep_Context, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.query_conv = nn.Sequential(nn.Conv2d(in_dim+8, 64, 1, bias=True))
        self.key_conv = nn.Sequential(nn.Conv2d(in_dim+8, 64, 1, bias=True))
        
        self.project = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        ) for i in range(parts_num)])
        self.pool = nn.MaxPool2d(2)
    def forward(self, p_fea, hu_att_list):
        n, c, h, w = p_fea.size()
        p_fea0 = p_fea
        p_fea = self.pool(p_fea)
        n, c, hp, wp = p_fea.size()
        
        coord_fea = torch.from_numpy(generate_spatial_batch(hp,wp))
        coord_fea = coord_fea.to(p_fea.device).repeat((n, 1, 1, 1)).permute(0,3,1,2)
        p_fea_coord = torch.cat([p_fea, coord_fea], dim=1)
        key = self.key_conv(p_fea_coord).view(n, -1, hp*wp).permute(0,2,1) # n, hw, c
        dep_cont = []
        dep_cont_att = []
        for i in range(len(hu_att_list)):
            query = self.query_conv(p_fea_coord*self.pool(hu_att_list[i])).view(n, -1, hp*wp) # n, c, hw 
            energy = torch.bmm(key, query)  # n,hw,hw
            attention = torch.sum(torch.softmax(energy, dim=-1), dim=-1) #n,hw
            
            attention = F.interpolate(attention.view(n,1,hp,wp), (h,w), mode = 'bilinear', align_corners=True)*hu_att_list[i]

            co_context = attention*p_fea0*(1-hu_att_list[i])
            co_context = self.project[i](co_context)
            dep_cont.append(co_context)
            dep_cont_att.append(attention.view(n,1,h,w)*(1-hu_att_list[i]))
        return dep_cont, dep_cont_att


class Contexture(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, part_list_list=None):
        super(Contexture, self).__init__()
        self.hidden_dim =hidden_dim
        self.F_cont = Dep_Context(in_dim, hidden_dim, len(part_list_list))

        self.att_list = nn.ModuleList([nn.Sequential(
            nn.Conv2d(hidden_dim, len(part_list_list[i]), kernel_size=1, padding=0, stride=1, bias=True)
        ) for i in range(len(part_list_list))])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, p_att_list, p_fea):
        F_dep_list, F_dep_att_list = self.F_cont(p_fea, p_att_list)

        att_list = [self.att_list[i](F_dep_list[i]) for i in range(len(p_att_list))]

        att_list_list = [list(torch.split(self.softmax(att_list[i])*F_dep_att_list[i], 1, dim=1)) for i in range(len(p_att_list))]
        return F_dep_list, att_list_list, att_list


class Dependency(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10):
        super(Dependency, self).__init__()
        self.relation = nn.Sequential(
            nn.Conv2d(2*hidden_dim, 2*hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2*hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )
    def forward(self, hv, hu_context, dep_att_huv):
        dep_message = self.relation(torch.cat([hu_context*dep_att_huv, hv], dim=1))
        return dep_message
    
class Full_Graph(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Full_Graph, self).__init__()
        self.cls_f = cls_f
        self.comp_full = Composition(hidden_dim, cls_h-1)
        
        self.update = nn.ModuleList([ConvGRU(hidden_dim,hidden_dim,(1,1)) for i in range(cls_f)])

    def forward(self, f_node_list, h_node_list, p_node_list, xf, h_node_att_list):
        f_node_list_new = []
        for i in range(self.cls_f):
            if i==0:
                node =self.update[i](h_node_list[0], f_node_list[0])
            elif i==1:
                comp_full, comp_f_att = self.comp_full(f_node_list[1], h_node_list[1:])
                node = self.update[i](comp_full, f_node_list[1])
            f_node_list_new.append(node)

        return f_node_list_new, comp_f_att

class Half_Graph(nn.Module):
    def __init__(self, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], in_dim=256, hidden_dim=10, cls_p=7,
                 cls_h=3, cls_f=2):
        super(Half_Graph, self).__init__()
        self.cls_h = cls_h
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.upper_parts_len = len(upper_part_list)
        self.lower_parts_len = len(lower_part_list)
        self.decomp_f = Decomposition(hidden_dim, 2)
        
        self.comp_u = Composition(hidden_dim, self.upper_parts_len)
        self.comp_l = Composition(hidden_dim, self.lower_parts_len)

        self.update = nn.ModuleList([ConvGRU(hidden_dim,hidden_dim,(1,1)) for i in range(cls_h)])

    def forward(self, f_node_list, h_node_list, p_node_list, xh, f_node_att_list, p_node_att_list):
        upper_parts = []
        upper_parts_att = []
        for part in self.upper_part_list:
            upper_parts.append(p_node_list[part])
            upper_parts_att.append(p_node_att_list[part])

        lower_parts = []
        lower_parts_att = []
        for part in self.lower_part_list:
            lower_parts.append(p_node_list[part])
            lower_parts_att.append(p_node_att_list[part])

        h_node_list_new = []
        decomp_f_list, decomp_f_att = self.decomp_f(f_node_list[1], h_node_list[1:], f_node_att_list[1])
        
    
        for i in range(self.cls_h):
            if i==0:
                node = self.update[i](f_node_list[0]+p_node_list[0], h_node_list[0])
            elif i==1:
                comp, comp_u_att = self.comp_u(h_node_list[1], upper_parts) 
                decomp = decomp_f_list[i-1]
                node = self.update[i](comp + decomp, h_node_list[i])
            elif i==2:
                comp, comp_l_att = self.comp_l(h_node_list[2], lower_parts)
                decomp = decomp_f_list[i-1]
                node = self.update[i](comp + decomp, h_node_list[i])
            h_node_list_new.append(node)

        return h_node_list_new, decomp_f_att, comp_u_att, comp_l_att


class Part_Graph(nn.Module):
    def __init__(self, adj_matrix, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], in_dim=256, hidden_dim=10,
                 cls_p=7, cls_h=3, cls_f=2):
        super(Part_Graph, self).__init__()
        self.cls_p = cls_p
        self.hidden = hidden_dim
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.edge_index = torch.nonzero(adj_matrix)
        self.edge_index_num = self.edge_index.shape[0]
        self.part_list_list = [[] for i in range(self.cls_p - 1)]
        for i in range(self.edge_index_num):
            self.part_list_list[self.edge_index[i, 1]].append(self.edge_index[i, 0])

        self.decomp_u = Decomposition(hidden_dim, len(upper_part_list))
        self.decomp_l = Decomposition(hidden_dim, len(lower_part_list))
        self.update = nn.ModuleList([ConvGRU(hidden_dim,hidden_dim,(1,1)) for i in range(cls_p)])
        
        self.F_dep_list = Contexture(in_dim=in_dim, hidden_dim=hidden_dim, part_list_list=self.part_list_list)
        self.part_dp = Dependency(in_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.ones(1))


    def forward(self, f_node_list, h_node_list, p_node_list, xp, h_node_att_list, p_node_att_list):
        # upper half
        upper_parts = []
        for part in self.upper_part_list:
            upper_parts.append(p_node_list[part])
        # lower half
        lower_parts = []
        for part in self.lower_part_list:
            lower_parts.append(p_node_list[part])

        # decomposition
        p_node_list_new = []
        decomp_u_list, decomp_u_att = self.decomp_u(h_node_list[1], upper_parts, h_node_att_list[1])
        decomp_l_list, decomp_l_att = self.decomp_l(h_node_list[2], lower_parts, h_node_att_list[2])
        
        # dependency
        F_dep_list, att_list_list, Fdep_att_list = self.F_dep_list(p_node_att_list[1:], xp)
        xpp_list_list = [[] for i in range(self.cls_p - 1)]
        for i in range(self.edge_index_num):
            xpp_list_list[self.edge_index[i, 1]].append(
                self.part_dp(p_node_list[self.edge_index[i, 1]], 
                F_dep_list[self.edge_index[i, 0]], 
                att_list_list[self.edge_index[i, 0]][self.part_list_list[self.edge_index[i, 0]].index(self.edge_index[i, 1])]))
            
        
        for i in range(self.cls_p):
            if i==0:
                node = self.update[i](h_node_list[0], p_node_list[0])
            elif i in self.upper_part_list:
                decomp = decomp_u_list[self.upper_part_list.index(i)]
                part_dp = sum(xpp_list_list[i-1])
                node = self.update[i](decomp+part_dp*self.alpha, p_node_list[i])
            elif i  in self.lower_part_list:
                decomp = decomp_l_list[self.lower_part_list.index(i)]
                part_dp = sum(xpp_list_list[i-1])
                node = self.update[i](decomp+part_dp*self.alpha, p_node_list[i])

            p_node_list_new.append(node)
        return p_node_list_new, decomp_u_att, decomp_l_att, Fdep_att_list


class GNN(nn.Module):
    def __init__(self, adj_matrix, upper_half_node=[1, 2, 3, 4], lower_half_node=[5, 6], in_dim=256, hidden_dim=10,
                 cls_p=7, cls_h=3, cls_f=2):
        super(GNN, self).__init__()
        self.cp = cls_p
        self.ch = cls_h
        self.cf = cls_f
        self.ch_in = in_dim
        self.hidden = hidden_dim
        self.upper_half_node = upper_half_node
        self.upper_node_len = len(self.upper_half_node)
        self.lower_half_node = lower_half_node
        self.lower_node_len = len(self.lower_half_node)

        self.full_infer = Full_Graph(in_dim, hidden_dim, cls_p, cls_h, cls_f)
        self.half_infer = Half_Graph(self.upper_half_node, self.lower_half_node, in_dim, hidden_dim, cls_p, cls_h,
                                     cls_f)
        self.part_infer = Part_Graph(adj_matrix, self.upper_half_node, self.lower_half_node, in_dim, hidden_dim, cls_p,
                                     cls_h, cls_f)

    def forward(self, p_node_list, h_node_list, f_node_list, xp, xh, xf, p_node_att_list, h_node_att_list, f_node_att_list):
        # for full body node
        # f_node_new_list = f_node_list
        f_node_new_list, comp_f_att = self.full_infer(f_node_list, h_node_list, p_node_list, xf, h_node_att_list)
        # for half body node
        # h_node_list_new = h_node_list
        h_node_list_new, decomp_att_fh, comp_u_att, comp_l_att = self.half_infer(f_node_list, h_node_list, p_node_list, xh, f_node_att_list, p_node_att_list)
        # for part node
        p_node_list_new, decomp_att_up, decomp_att_lp, Fdep_att_list = self.part_infer(f_node_list, h_node_list, p_node_list, xp, h_node_att_list, p_node_att_list)
        
        # return p_node_list_new, h_node_list_new, f_node_new_list, decomp_att_fh, decomp_att_up, decomp_att_lp
        return p_node_list_new, h_node_list_new, f_node_new_list, decomp_att_fh, decomp_att_up, decomp_att_lp, comp_f_att, comp_u_att, comp_l_att, Fdep_att_list, 


class GNN_infer(nn.Module):
    def __init__(self, adj_matrix, upper_half_node=[1, 2, 3, 4], lower_half_node=[5, 6], in_dim=256, hidden_dim=64,
                 cls_p=7, cls_h=3, cls_f=2):
        super(GNN_infer, self).__init__()
        self.cls_p = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # node feature transform 
        self.p_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim * cls_p, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim * cls_p), nn.ReLU(inplace=False))
        self.h_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim * cls_h, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim * cls_h), nn.ReLU(inplace=False))
        self.f_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim * cls_f, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim * cls_f), nn.ReLU(inplace=False))

        # gnn infer
        self.gnn = nn.ModuleList([GNN(adj_matrix, upper_half_node, lower_half_node, self.in_dim, self.hidden_dim, self.cls_p,
                       self.cls_h, self.cls_f) for i in range(2)])

        # node supervision
        self.node_seg = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, xp, xh, xf):
        # gnn inference at stride 8
        # feature transform
        f_node_list = list(torch.split(self.f_conv(xf), self.hidden_dim, dim=1))
        h_node_list = list(torch.split(self.h_conv(xh), self.hidden_dim, dim=1))
        p_node_list = list(torch.split(self.p_conv(xp), self.hidden_dim, dim=1))

        # node supervision
        f_seg = []
        h_seg = []
        p_seg = []
        decomp_att_fh = []
        decomp_att_up = []
        decomp_att_lp = []
        comp_f_att = []
        comp_u_att = [] 
        comp_l_att = []
        Fdep_att_list = []
        f_seg.append(torch.cat([self.node_seg(node) for node in f_node_list], dim=1))
        h_seg.append(torch.cat([self.node_seg(node) for node in h_node_list], dim=1))
        p_seg.append(torch.cat([self.node_seg(node) for node in p_node_list], dim=1))
        f_att_list = list(torch.split(torch.softmax(f_seg[0], 1), 1, dim=1))
        h_att_list = list(torch.split(torch.softmax(h_seg[0], 1), 1, dim=1))
        p_att_list = list(torch.split(torch.softmax(p_seg[0], 1), 1, dim=1))

        for iter in range(2):
            # gnn infer
            p_node_list_new, h_node_list_new, f_node_list_new, decomp_att_fh_new, decomp_att_up_new, decomp_att_lp_new, comp_f_att_new, comp_u_att_new, comp_l_att_new, Fdep_att_list_new = self.gnn[iter](p_node_list, h_node_list, f_node_list, xp, xh, xf, p_att_list, h_att_list, f_att_list)
            # node supervision new
            f_seg.append(torch.cat([self.node_seg(node) for node in f_node_list_new], dim=1))
            h_seg.append(torch.cat([self.node_seg(node) for node in h_node_list_new], dim=1))
            p_seg.append(torch.cat([self.node_seg(node) for node in p_node_list_new], dim=1))
            decomp_att_fh.append(decomp_att_fh_new)
            decomp_att_up.append(decomp_att_up_new)
            decomp_att_lp.append(decomp_att_lp_new)
            comp_f_att.append(comp_f_att_new)
            comp_u_att.append(comp_u_att_new)
            comp_l_att.append(comp_l_att_new)
            Fdep_att_list.append(Fdep_att_list_new)
            f_att_list = list(torch.split(torch.softmax(f_seg[-1], 1), 1, dim=1))
            h_att_list = list(torch.split(torch.softmax(h_seg[-1], 1), 1, dim=1))
            p_att_list = list(torch.split(torch.softmax(p_seg[-1], 1), 1, dim=1))
            p_node_list = p_node_list_new
            h_node_list = h_node_list_new
            f_node_list = f_node_list_new

        return p_seg, h_seg, f_seg, decomp_att_fh, decomp_att_up, decomp_att_lp, comp_f_att, comp_u_att, comp_l_att,   Fdep_att_list


class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        self.layer5 = ASPPModule(2048, 512)
        self.layer_part = DecoderModule(num_classes)
        self.layer_half = DecoderModule(hbody_cls)
        self.layer_full = DecoderModule(fbody_cls)
        
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(256), nn.ReLU(inplace=False),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.skip = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False),
                                   )
        self.fuse = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))
        
        # adjacent matrix for pascal person 
        self.adj_matrix = torch.tensor(
            [[0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0]], requires_grad=False)
        
        # infer with hierarchical person graph
        self.gnn_infer = GNN_infer(adj_matrix=self.adj_matrix, upper_half_node=[1, 2, 3, 4], lower_half_node=[5, 6],
                                   in_dim=256, hidden_dim=32, cls_p=7, cls_h=3, cls_f=2)
        # aux layer
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(256), nn.ReLU(inplace=False),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        _,_,h,w = x[1].size()
        context = self.layer5(x[-1])
        context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)
        context = self.fuse(torch.cat([self.skip(x[1]), context], dim=1))

        p_fea = self.layer_part(context)
        h_fea = self.layer_half(context)
        f_fea = self.layer_full(context)

        # gnn infer
        p_seg, h_seg, f_seg, decomp_att_fh, decomp_att_up, decomp_att_lp, comp_f_att, comp_u_att, comp_l_att,Fdep_att_list = self.gnn_infer(p_fea, h_fea, f_fea)

        return p_seg, h_seg, f_seg, decomp_att_fh, decomp_att_up, decomp_att_lp, comp_f_att, comp_u_att, comp_l_att,Fdep_att_list, x_dsn

class OCNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(OCNet, self).__init__()
        self.encoder = ResGridNet(block, layers)
        self.decoder = Decoder(num_classes=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, InPlaceABNSync):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_model(num_classes=20):
    model = OCNet(Bottleneck, [3, 4, 23, 3], num_classes)  # 101
    return model
