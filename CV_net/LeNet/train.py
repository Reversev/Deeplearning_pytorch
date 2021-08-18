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
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
from model import LeNet
writer = SummaryWriter()   # start tensorboardx monitor

BATCH_SIZE = 64
RATE_LEARNING = 1e-5
EPOCH = 5
PRINT_EPOCH = 100
DATA_PATH = './data'     # dataset save path
SAVE_PATH = './model/'   # trained model path
DOWNLOAD = False         # if you want to download data, it sets 'True', otherwise.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Use device: ', device)  # print used device during the training
# data preprocess
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# load train set and test set
train_samples = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                             download=DOWNLOAD, transform=transform)
val_samples = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                           download=DOWNLOAD, transform=transform)
# data format transform
train_loader = torch.utils.data.DataLoader(train_samples, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_samples, batch_size=4,
                                         shuffle=True, num_workers=0)

# category labels
classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# train set are load for classification
val_iter = iter(val_loader)
val_images, val_labels = val_iter.next()

# visualize validation samples
# # print labels
# print(' '.join('%5s' % classes[val_labels[j]] for j in range(4)))
# # imshow images
# img = torchvision.utils.make_grid(val_images)
# img = img / 2 + 0.5      # unnormalize
# npimg = img.numpy()
# plt.imshow(np.transpose(npimg, (1, 2, 0)))
# plt.show()

val_images = val_images.to(device)
val_labels = val_labels.to(device)
# initialize net
net = LeNet()
print(net)
net = net.to(device)

# design loss function
loss_function = nn.CrossEntropyLoss()
# design optimizer
optimizer = optim.Adam(net.parameters(), lr=RATE_LEARNING)

# start training
for epoch in range(EPOCH):
    run_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data   # build samples and labels from data

        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
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
                # visualize for training process
                writer.add_scalar('Training loss', run_loss/PRINT_EPOCH, global_step=step)
                writer.add_scalar('Valdation accuracy', acc_rate, global_step=step)
                run_loss = 0.0
            writer.close()

print('Finished Training!')
# save model
save_path = SAVE_PATH + 'LeNet1.pth'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
torch.save(net.state_dict(), save_path)
# save onnx model format
# dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True).to(device)
# torch.onnx.export(net, dummy_input, SAVE_PATH + "lenet1.onnx")
