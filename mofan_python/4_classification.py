#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author = 'IReverser'
"""
    这是一个利用了分类处理的神经网络教程
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time

n_data = torch.ones([100, 2])
x0 = torch.normal(2*n_data, 1)   # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)            # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)  # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)             # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)   # LongTensor = 64-bit integer


# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)
#
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
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


net = Net(2, 10, 2)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

start_time = time.time()
plt.ion()

for t in range(150):
    print(t)
    out = net(x)    # [-2, -.12, 20] F,softmax(out) [.1, 0.2, .7]

    loss = loss_func(out, y)   # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # set gradient of parameters to zero
    loss.backward()         # calculate gradient of per node
    optimizer.step()    # optimize gradient

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

end_time = time.time()
print('The total cost time:', str(end_time - start_time) + 'sec')
plt.ioff()
plt.show()




























