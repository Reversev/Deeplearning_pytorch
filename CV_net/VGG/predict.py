#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/8/25 16:14
# @Author : ''
# @FileName: predict.py
import os
import json
import glob
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import vgg

model_name = 'vgg16'
MODEL_PATH = './model/' + model_name + '_flowers.pth'
TEST_PATH = './results/'
CLASS_NUM = 5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Use device: ', device)
# pre-processing
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# read class_dictionary
json_path = './classes_indices.json'
assert os.path.exists(json_path), "file: '{}' does not exists.".format(json_path)

json_file = open(json_path, "r")
classes_indict = json.load(json_file)
print(classes_indict)
# create model
net = vgg(model_name=model_name, n_classes=CLASS_NUM)
net = net.to(device)

# load model weights
assert os.path.exists(MODEL_PATH), "file: '{}' does not exist.".format(MODEL_PATH)
net.load_state_dict(torch.load(MODEL_PATH))

for im_path in glob.glob(TEST_PATH + '*.jpg'):
    # load data
    im = Image.open(im_path)
    im = transform(im)  # [H, W, C] -> [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [C, H, W] -> [N, C, H, W]
    im = im.to(device)

    net.eval()
    with torch.no_grad():
        output = net(im)
        # confidence = torch.max(output, dim=1)[0].cpu().data.numpy()[0]  # option
        confidence = torch.max(torch.softmax(output, dim=1)).cpu().data.numpy()
        predict = torch.max(output, dim=1)[1].cpu().data.numpy()

    print('Vertification picture:', im_path.split('/')[-1], '\t',
          'Recognition result:', classes_indict[str(int(predict))], '\t',
          'Recognition confidence:', str(confidence))

