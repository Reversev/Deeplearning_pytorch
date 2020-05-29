#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author = 'IReverser'
"""
    这是一个利用了回归处理的神经网络教程
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy
import time

# unsqueeze use to transfor one dimension to two dimension
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor),shape(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

start_time = time.time()
plt.ion()

for t in range(500):
    print(t)
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()   # set gradient of parameters to zero
    loss.backward()         # calculate gradient of per node
    optimizer.step()    # optimize gradient

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

end_time = time.time()
print('The total cost time:', str(end_time - start_time) + 'sec')
plt.ioff()
plt.show()




























