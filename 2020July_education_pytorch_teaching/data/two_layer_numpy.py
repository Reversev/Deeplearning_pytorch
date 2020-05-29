#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2020/2/24
# @Author ：'IReverser'
# @FileName: two_layer_numpy.py
from __future__ import print_function
import numpy as np

# N is batch size;
# D_in is input dimension;
# H is hidden dimension;
# D_out is output dimension.
N, D_in, H, D_out = 64, 100, 100, 10
learning_rate = 1e-5

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

for i in range(500):
    # Forward pass: compute predicted y
    # .dot don't use in float if you change x(y) with normal.
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred-y).sum()
    print(i, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss.
    # loss = (y_pred - y) ** 2
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2






