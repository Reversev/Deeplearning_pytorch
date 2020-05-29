#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2020/2/24
# @Author ：'IReverser'
# @FileName: two_layers_tensor_autograd.py

import torch

dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0")

# N is batch size;
# D_in is input dimension;
# H is hidden dimension;
# D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
learning_rate = 1e-6

# 创建随机的Tensor来保存输入和输出
# 设定requires_grad=False表示在反向传播的时候我们不需要计算gradient
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 创建随机的Tensor和权重。
# 设置requires_grad=True表示我们希望反向传播的时候计算Tensor的gradient
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

for t in range(500):
    # 前向传播:通过Tensor预测y；这个和普通的神经网络的前向传播没有任何不同，
    # 但是我们不需要保存网络的中间运算结果，因为我们不需要手动计算反向传播。
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 通过前向传播计算loss
    # loss是一个形状为(1，)的Tensor
    # loss.item()可以给我们返回一个loss的scalar
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # PyTorch给我们提供了autograd的方法做反向传播。如果一个Tensor的requires_grad=True，
    # backward会自动计算loss相对于每个Tensor的gradient。
    # 在backward之后，w1.grad和w2.grad会包含两个loss相对于两个Tensor的gradient信息。
    loss.backward()

    # 用torch.no_grad()包含以下statements，因为w1和w2都是requires_grad=True，
    # 但是在更新weights之后我们并不需要再做autograd。
    # 另一种方法是在weight.data和weight.grad.data上做操作，这样就不会对grad产生影响。
    # tensor.data会我们一个tensor，这个tensor和原来的tensor指向相同的内存空间，但是不会记录计算图的历史。
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()


