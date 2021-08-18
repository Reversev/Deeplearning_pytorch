#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/7/29 16:44
# @Author : 'IReverser'
# @FileName: train.py

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import LeNet_MNIST
import os

BATCH_SIZE = 36
RATE_LEARNING = 0.0001
EPOCH = 7
PRINT_EPOCH = 100
DATA_PATH = './data'
SAVE_PATH = './model/'
DOWNLOAD = True  # if you want to download data, it sets 'True', otherwise.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Use device: ', device)
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])
# load train set and test set
train_samples = torchvision.datasets.MNIST(root=DATA_PATH, train=True,
                                           download=DOWNLOAD, transform=transform)
val_samples = torchvision.datasets.MNIST(root=DATA_PATH, train=False,
                                         download=DOWNLOAD, transform=transform)
# data format transform
train_loader = torch.utils.data.DataLoader(train_samples, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_samples, batch_size=10000,
                                         shuffle=True, num_workers=0)

# category labels
# classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# train set are load for classification
val_iter = iter(val_loader)
val_images, val_labels = val_iter.next()
val_images = val_images.to(device)
val_labels = val_labels.to(device)

# initialize net
net = LeNet_MNIST()
net = net.to(device)

# design loss function
loss_function = nn.CrossEntropyLoss()
# design optimizer
optimizer = optim.Adam(net.parameters(), lr=RATE_LEARNING)

# start training
for epoch in range(EPOCH):
    run_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = net(inputs)
        loss = loss_function(output, labels)
        # Update Visualization
        loss.backward()
        optimizer.step()

        run_loss += loss.item()
        if step % PRINT_EPOCH == (PRINT_EPOCH - 1):
            with torch.no_grad():
                outputs = net(val_images)
                pred_y = torch.max(outputs, dim=1)[1].to(device)
                acc_rate = torch.eq(pred_y, val_labels).sum().item() / val_labels.size(0)
                print('Epoch: %d, iter: %d, train_loss: %.3f, test_accuracy: %.3f'
                      % (epoch, step, run_loss/PRINT_EPOCH, acc_rate))
                run_loss = 0.0

print('Finished Training!')
# save model
save_path = SAVE_PATH + 'LeNet_mnist.pth'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
torch.save(net.state_dict(), save_path)
# save onnx model format
# dummy_input = torch.randn(1, 1, 28, 28, requires_grad=True).to(device)
# torch.onnx.export(net, dummy_input, SAVE_PATH + "lenet_mnist.onnx")
