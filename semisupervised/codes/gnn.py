import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from layer import GraphConvolution

class GNNq(nn.Module):
    def __init__(self, opt, adj):
        super(GNNq, self).__init__()
        self.opt = opt
        self.adj = adj

        # input feature
        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x

class GNNq_feature(nn.Module):
    def __init__(self, opt, adj):
        super(GNNq, self).__init__()
        self.opt = opt
        self.adj = adj

        # input feature
        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        h = F.relu(x)
        x = F.dropout(h, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x, h

class GNNp(nn.Module):
    def __init__(self, opt, adj):
        super(GNNp, self).__init__()
        self.opt = opt
        self.adj = adj

        # input pseudo label
        opt_ = dict([('in', opt['num_class']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x

class GNNp_concat(nn.Module):
    def __init__(self, opt, adj):
        super(GNNp_concat, self).__init__()
        self.opt = opt
        self.adj = adj

        # concat
        opt_ = dict([('in', opt['num_class']+opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x

class GNNp_singelGCN(nn.Module):
    def __init__(self, opt, adj):
        super(GNNp_concat, self).__init__()
        self.opt = opt
        self.adj = adj

        # concat
        opt_ = dict([('in', opt['num_class']+opt['hidden_dim']), ('out', opt['num_class'])])
        self.m1 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()

    def forward(self, x, h):
        x = torch.cat((x, h), 1)
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        return x