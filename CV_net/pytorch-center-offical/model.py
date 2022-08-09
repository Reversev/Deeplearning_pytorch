#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/9/20 15:11
# @Author : 'IReverser'
# @FileName: model.py
from abc import ABC

import torch
import torch.nn as nn


class TrunkNet(nn.Module, ABC):
    def __init__(self, num_classes=1000):
        super(TrunkNet, self).__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            Basicblock(32, 64),
            self._make_resblock(64, 64, num_layers=1),
            Basicblock(64, 128),
            self._make_resblock(128, 128, num_layers=2),
            Basicblock(128, 256),
            self._make_resblock(256, 256, num_layers=5),
            Basicblock(256, 512),
            self._make_resblock(512, 512, num_layers=3),
        )
        self.fc1 = nn.Linear(512 * 7 * 6, 512)   # 512 * h * w = 25088
        # self.fc1 = nn.Linear(512 * 14 * 14, 512)  # 512 * h * w = 25088

        self.classify = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_resblock(self, in_channels, out_channels, num_layers):
        block = []
        for _ in range(num_layers):
            block.append(Bottleneck(in_channels, out_channels))
        return nn.Sequential(*block)

    def forward(self, x):
        x = self.trunk(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)
        x2 = self.classify(x1)
        return x1, x2


class Basicblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Basicblock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(self.prelu(x))
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        return x


class Branch(nn.Module, ABC):
    def __init__(self, num_classes=1000):
        super(Branch, self).__init__()
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.classify = nn.Linear(512, num_classes, bias=True)

    def forward(self, fea):
        fea = self.fc1(fea)
        # print(x.shape)
        fea_cm = self.classify(fea)
        return fea, fea_cm


__factory = {
    'dcr-trunk': TrunkNet,
}


def create(name, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes)


if __name__ == '__main__':
    pass

# test
# x = torch.rand((1, 3, 224, 224))
# num_classes = 10
# net = TrunkNet(num_classes=num_classes)
# print(net)
# [center_y, soft_x] = net(x)
# print(center_y.shape, soft_x.shape)
