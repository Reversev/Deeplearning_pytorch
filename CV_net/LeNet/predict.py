#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/7/29 16:44
# @Author : ''
# @FileName: predict.py

import torch
import torchvision.transforms as transform
from PIL import Image
import glob
from model import LeNet


MODEL_PATH = './model/LeNet1.pth'
TEST_PATH = './results/'
# category labels
classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Use device: ', device)
# pre-processing
transform = transform.Compose(
    [transform.Resize((32, 32)),
     transform.ToTensor(),
     transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

net = LeNet()
net = net.to(device)
net.load_state_dict(torch.load(MODEL_PATH))

for im_path in glob.glob(TEST_PATH + '*.jpg'):
    im = Image.open(im_path)
    im = transform(im)    # [H, W, C] -> [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [C, H, W] -> [N, C, H, W]
    im = im.to(device)

    with torch.no_grad():
        output = net(im)
        # confidence = torch.max(output, dim=1)[0].cpu().data.numpy()[0]  # option
        confidence = torch.max(torch.softmax(output, dim=1)).cpu().data.numpy()
        predict = torch.max(output, dim=1)[1].cpu().data.numpy()

    print('Vertification picture:', im_path.split('/')[-1],
          'Recognition result:',  classes[int(predict)],
          'Recognition confidence:', str(confidence))
