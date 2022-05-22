from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


# ----------TODO------------
# Implement the PointNet 
# ----------TODO------------

class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, d=1024):
        super(PointNetfeat, self).__init__()

        self.d = d
        self.global_feat = global_feat

        self.layer1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, self.d),
            nn.ReLU()
        )

    def forward(self, x):
        # x: 32*3*1024
        n_pts = x.size()[2]  # 点的个数
        out1 = self.layer1(x.transpose(1, 2))  # 32*1024*64
        out2 = self.layer2(out1)  # 32*1024*128
        out3 = self.layer3(out2)  # 32*1024*d
        features = nn.MaxPool1d(n_pts)(out3.transpose(1, 2)).view(-1, self.d)  # 32*d

        if self.global_feat:
            return features
        else:
            features = features.view(-1, 1, self.d).repeat(1, n_pts, 1)
            ret = torch.cat((out1, features), 2)
            return ret


class PointNetCls1024D(nn.Module):
    def __init__(self, k=2):
        super(PointNetCls1024D, self).__init__()
        self.k = k
        self.get_feature = PointNetfeat(d=1024)
        # DEBUGGING LOG: 不要把seq写在forward里......
        self.seq = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.k)
        )

    def forward(self, x):
        x = self.get_feature(x)
        x = self.seq(x)
        vis_feature = 0
        return F.log_softmax(x, dim=1), vis_feature
        # vis_feature only for visualization, your can use other ways to obtain the vis_feature


class PointNetCls256D(nn.Module):
    def __init__(self, k=2):
        super(PointNetCls256D, self).__init__()
        self.k = k
        self.get_feature = PointNetfeat(d=256)
        self.seq = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.k)
        )

    def forward(self, x):
        x = self.get_feature(x)
        x = self.seq(x)
        return F.log_softmax(x, dim=1)


class PointNetDenseCls(nn.Module):
    def __init__(self, k=2):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.get_feature = PointNetfeat(global_feat=False, d=1024)
        self.seq = nn.Sequential(
            nn.Linear(64 + 1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.k)
        )

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = self.get_feature(x)
        x = self.seq(x)
        return F.log_softmax(x, dim=1)
