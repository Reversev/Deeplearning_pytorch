#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/7/29 16:43
# @Author : 'IReverser'
# @FileName: model.py
import torch
import torch.nn.functional as F
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # conv2d: in_channel, out_channel, kernel_size
        self.pool1 = nn.MaxPool2d(2, 2)   # MaxPool2d: h, w
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)  # Linear: in_connection, out_connection
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)       # classes = 10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet_MNIST(nn.Module):
    """
        LeNet for mnist dataset.
    """
    def __init__(self):
        super(LeNet_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # conv2d: in_channel, out_channel, kernel_size
        self.pool1 = nn.MaxPool2d(2, 2)   # MaxPool2d: h, w
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)  # Linear: in_connection, out_connection
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)       # classes = 10

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 6*5*5，output:6*28*28
        x = self.pool1(x)          # 6*14*14
        x = F.relu(self.conv2(x))  # 16*10*10
        x = self.pool2(x)          # 16*5*5
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# test
# x = torch.rand((1, 1, 28, 28))
# net = LeNet_MNIST()
# y = net(x)
# print(y.shape)
