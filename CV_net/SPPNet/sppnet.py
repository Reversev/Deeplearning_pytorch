#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2023/7/13 12:29
# @Author : ''
# @FileName: sppnet.py
# modified from https://github.com/yifanjiang19/sppnet-pytorch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    """
    previous_conv: a tensor vector of previous convolution layer
    num_sample: the number of image in the batch
    previous_conv_size: the vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    """
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        h_wid = math.ceil(previous_conv_size[0] / out_pool_size[i])
        w_wid = math.ceil(previous_conv_size[1] / out_pool_size[i])
        h_str = math.floor(previous_conv_size[0] / out_pool_size[i])
        w_str = math.floor(previous_conv_size[1] / out_pool_size[i])
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_str, w_str))
        x = maxpool(previous_conv)
        # print("x.size ", i, ":", x.size())
        if i == 0:
            spp = x.view(num_sample, -1)
            # print("0spp size: ", spp.size())
        else:
            # print("size: ", i, " ", spp.size())
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp


class SPPNet(nn.Module):
    def __init__(self, ndf=64, num_classes=1000):
        super(SPPNet, self).__init__()
        self.output_num = [4, 2, 1]

        self.conv1 = nn.Conv2d(3, ndf, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, 64, 4, 1, 0, bias=False)
        self.fc1 = nn.Linear(64 * (self.output_num[0]**2 + self.output_num[1]**2 + self.output_num[2]**2), 1024)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(self.bn1(x))

        x = self.conv3(x)
        x = F.leaky_relu(self.bn2(x))

        x = self.conv4(x)
        x = F.leaky_relu(self.bn3(x))
        x = self.conv5(x)
        # print(x.shape)  # 1 64 106 106

        spp = spatial_pyramid_pool(x, x.size(0), [int(x.size(2)), int(x.size(3))], self.output_num)   # [106, 106]
        # print(spp.size())  # 1 1344

        fc1 = self.dropout(self.fc1(spp))
        fc2 = self.fc2(fc1)
        out = torch.sigmoid(fc2)
        return out


model_list = {
    "sppnet": SPPNet,
}


if __name__ == '__main__':
    x = torch.randn((2, 3, 224, 224))
    net = model_list["sppnet"](num_classes=1000)
    y = net(x)
    print(y.shape)
