#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/8/9 22:57
# @Author : 'IReverser'
# @FileName: train.py


import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import torch.optim as optim
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from model import AlexNet


def main():
    dataset_name = 'flowers'
    BATCH_SIZE = 32
    CLASS_NUM = 5
    EPOCH = 10
    SAVE_PATH = './model/AlexNet_' + dataset_name + '.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use device: ', device)  # print used device during the training

    data_transform = {
        "train": transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    image_path = "../datasets/" + dataset_name + "/"
    assert os.path.join(image_path), "{} path does not exist!".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    # print('train number: ', train_num)

    # transform labels
    dataset_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in dataset_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open("classes_indices.json", "w") as f:
        f.write(json_str)

    net_count = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of network
    print("Use {} dataloader workers every process".format(net_count))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=net_count)
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform["val"])
    val_num = len(val_dataset)
    # print('val number: ', val_num)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=net_count)

    print("Use {} images for training, {} images for validation.".format(train_num, val_num))

    # visualization for validation when the batch size is set to 4 in val_loader
    # val_data_iter = iter(val_loader)
    # val_image, val_label = val_data_iter.next()
    # def imshow(im):
    #     im = im / 2 + 0.5  # denormalize
    #     npimg = im.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    # print(''.join('%5s' % cla_dict[val_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(val_image))

    # define network
    net = AlexNet(n_classes=CLASS_NUM, init_weights=False)
    net.to(device)
    print(net)
    loss_function = nn.CrossEntropyLoss()
    # para = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    best_acc = 0.0
    train_step = len(train_loader)
    for epoch in range(EPOCH):
        # train process
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = 'train epoch [{}/{}] loss: {:3f}'.format(epoch + 1, EPOCH, loss)

        # validation
        net.eval()
        acc = 0.0   # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                pred_y = torch.max(outputs, dim=1)[1]

                acc += torch.eq(pred_y, val_labels.to(device)).sum().item()

            val_acc = acc / val_num
            print('[epoch %d] train_loss: %.3f, validation accuracy:%.3f\n' % (epoch + 1, running_loss / train_num, val_acc))

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(net.state_dict(), SAVE_PATH)

    print('Finished Training!')


if __name__ == '__main__':
    main()



