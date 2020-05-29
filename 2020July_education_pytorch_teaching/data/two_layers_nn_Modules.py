#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2020/2/25
# @Author ：''
# @FileName: two_layers_nn_Modules.py
import torch
from torch import nn


class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):

        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N is batch size;
# D_in is input dimension;
# H is hidden dimension;
# D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
learning_rate = 1e-4

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(500):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
