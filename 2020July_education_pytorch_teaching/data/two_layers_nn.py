#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2020/2/25
# @Author ：'IReverser'
# @FileName: two_layers_nn.py
import torch
from torch import nn

# N is batch size;
# D_in is input dimension;
# H is hidden dimension;
# D_out is output dimension.
N, D_in, H, D_out = 64, 100, 100, 10
learning_rate = 1e-3

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers.
# nn.Sequential is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a linear function,
# and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out)
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = nn.MSELoss(reduction='sum')

for i in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true values of y,
    # and the loss function returns a Tensor containing the loss.
    loss = loss_fn(y_pred, y)
    print(i, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable parameters of the model.
    # Internally, the parameters of each Module are stored in Tensors with requires_grad=True, so this call will
    # compute gradients for all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad