#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/9/19 17:22
# @Author : ''
# @FileName: model.py
import torch
import torch.nn as nn


def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False))
    return layer


class DenseBlock(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate

        self.net = nn.Sequential(*block)

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)  # concat from channels
        return x


def transition_layer(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channel, out_channel, kernel_size=1),
        nn.AvgPool2d(2, 2))
    return trans_layer


class DenseNet(nn.Module):
    def __init__(self, in_channel, block_layers, num_classes, growth_rate=32):
        super(DenseNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.denseblock1 = self._make_dense_block(64, growth_rate, num=block_layers[0])
        self.translayer1 = self._make_transition_layer(256)
        self.denseblock2 = self._make_dense_block(128, growth_rate, num=block_layers[1])
        self.translayer2 = self._make_transition_layer(512)
        self.denseblock3 = self._make_dense_block(256, growth_rate, num=block_layers[2])
        self.translayer3 = self._make_transition_layer(1024)
        self.denseblock4 = self._make_dense_block(512, growth_rate, num=block_layers[3])
        self.global_average = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.denseblock1(x)
        x = self.translayer1(x)
        x = self.denseblock2(x)
        x = self.translayer2(x)
        x = self.denseblock3(x)
        x = self.translayer3(x)
        x = self.denseblock4(x)
        x = self.global_average(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        return x

    def _make_dense_block(self, channels, growth_rate, num):
        block = [DenseBlock(channels, growth_rate, num)]
        channels += num * growth_rate
        return nn.Sequential(*block)

    def _make_transition_layer(self, channels):
        block = [transition_layer(channels, channels // 2)]
        return nn.Sequential(*block)


def DenseNet121(num_classes=1000):
    return DenseNet(3, [6, 12, 24, 16], num_classes=num_classes, growth_rate=32)


def DenseNet169(num_classes=1000):
    return DenseNet(3, [6, 12, 32, 32], num_classes=num_classes, growth_rate=32)


def DenseNet201(num_classes=1000):
    return DenseNet(3, [6, 12, 48, 32], num_classes=num_classes, growth_rate=32)


def DenseNet161(num_classes=1000):
    return DenseNet(3, [6, 12, 36, 24], num_classes=num_classes, growth_rate=48)

