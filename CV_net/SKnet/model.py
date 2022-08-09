#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/11/29 9:01
# @Author : 'IReverser'
# @FileName: model.py
# Selective Kernel Networks https://arxiv.org/pdf/1903.06586.pdf
import torch.nn as nn
from functools import reduce


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        """
        :param in_channels: the dimension of input channel
        :param out_channels: the dimension of output channel
        :param stride: stride, default: 1
        :param M: the number of branch in the split phase
        :param r: the length of features z and calculate ratio of the dimension d.
        In the paper, features s -> z is reduce process and so you need to define lower bounding in the reduction.
        :param L: the lower bounding of feature z in the paper, default: 32
        """
        super(SKConv, self).__init__()
        d = max(in_channels//r, L)  # calculate the length of the features z
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(self.M):
            self.conv.append(
                nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=32, bias=False),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # adaptively pool to defined dimension, here define to 1 for achieving GAP.
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, kernel_size=1, bias=False),  # reduce dimensional
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(d, out_channels * self.M, kernel_size=1, stride=1, bias=False)  # increase dimensional
        self.softmax = nn.Softmax(dim=1)  # define dim=1 to ensure a + b = 1 in the linear layer corresponding location

    def forward(self, input):
        batch_size = input.size(0)
        output = []
        # the part of split
        for i, conv in enumerate(self.conv):
            # print(i, conv(x).size())
            output.append(conv(input))
        # the part of fusion
        U = reduce(lambda x, y: x + y, output)  # element-wise addition to build mixed matrix
        s = self.global_pool(U)
        z = self.fc1(s)  # s->z reduce dimensional  [B, 1, 1, 1] -> [B, 32, 1, 1]
        # print(z.shape)
        a_b = self.fc2(z)  # z -> a, b increase dimension and conv 1x1 is used in the paper, the first results is a, other is b.
        # print(a_b.shape)  # [B, 32, 1, 1] -> [B, C*M, 1, 1]
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)  # [B, MC, 1, 1] -> [B, M, C, 1]
        # print(a_b.shape)
        a_b = self.softmax(a_b)  # in the dim=1, soft max function is used.

        # the part of selection
        # print(a_b.shape)
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))  # [B, M, C, 1] -> list[[B, C, 1, 1], M]
        V = list(map(lambda x, y: x * y, output, a_b))  # weight and corresponding output U from different conv_block are used to element-wise multiplication operator
        V = reduce(lambda x, y: x + y, V)  # M weighted features are used to element-wise addition operator

        return V


class SKBlock(nn.Module):
    """
    construct SKBlock method based on ResBlock
    1x1 conv(channel num: x) + SKConv (channel num: x) + 1x1 conv(channel: 2x)
    """
    expansion = 2  # define the increased ratio of each number of channels in the each Block

    def __init__(self, inplances, planes, stride=1, shortcut=None):
        super(SKBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inplances, planes, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU(inplace=True))
        self.conv2 = SKConv(planes, planes, stride)
        self.conv3 = nn.Sequential(nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(planes * self.expansion))
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut

    def forward(self, input):
        identity = input
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        if self.shortcut is not None:
            identity = self.shortcut(input)

        output += identity
        return self.relu(output)


class SKNet(nn.Module):
    """
    construct the backbone in the paper.
    """
    def __init__(self, block=SKBlock, blocks_num=[3, 4, 6, 3], num_classes=1000, include_top=True):
        super(SKNet, self).__init__()
        self.in_channel = 64
        self.include_top = include_top

        self.conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, blocks_num[0], stride=1)
        self.layer2 = self._make_layer(block, 256, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 512, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, blocks_num[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if self.include_top:
            self.fc = nn.Linear(1024 * block.expansion, num_classes)
            self.softmax = nn.Softmax(-1)  # softmax function is used in last dimension
        self._initalize_weights()

    def forward(self, x):
        out = self.conv(x)
        out = self.maxpool(out)
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = self.avgpool(out)
        out = out.squeeze(-1).squeeze(-1)
        if self.include_top:
            out = self.fc(out)
            out = self.softmax(out)
        return out

    def _make_layer(self, block, channel, block_num, stride=1):
        shortcut = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, shortcut=shortcut, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def _initalize_weights(self):
        for m in self.modules():
            if isinstance(nn, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)


def SKNet50(num_classes=1000, include_top=True):
    return SKNet(SKBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def SKNet101(num_classes=1000, include_top=True):
    return SKNet(SKBlock, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def SKNet152(num_classes=1000, include_top=True):
    return SKNet(SKBlock, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)


# x = torch.rand(2, 3, 224, 224)
# model = SKNet50()
# y = model(x)
# print(y.shape)
