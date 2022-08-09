#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/11/29 18:42
# @Author : ''
# @FileName: predict.py
import torch
from torch import nn
from torch.nn.parameter import Parameter


class ECA_layer(nn.Module):
    """
    Construct a ECA module
    channel: the number of channels of the input feature map
    k_size: adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:[B, C, H, W]
        # feature descriptor on the global spatial information
        out = self.avg_pool(x)

        # two different branches of ECA module
        out = self.conv(out.squeeze(-1).transpose(-1, -2))
        out = out.transpose(-1, -2).unsqueeze(-1)

        # multi-scale information fusion
        out = self.sigmoid(out)

        return x * out.expand_as(x)  # expand_as is expand out such like x

