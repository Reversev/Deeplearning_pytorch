#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/8/25 16:14
# @Author : 'IReverser'
# @FileName: train.py
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from model import vgg


def main():
    model_name = 'vgg16'   # vgg11, vgg11L, vgg13, vgg16, vgg19
    dataset_name = 'flowers'
    BATCH_SIZE = 32
    CLASS_NUM = 5
    EPOCH = 10
    SAVE_PATH = './model/' + model_name + '_' + dataset_name + '.pth'
    if not os.path.exists(os.path.dirname(SAVE_PATH)):
        os.mkdir(os.path.dirname(SAVE_PATH))
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

    net = vgg(model_name=model_name, n_classes=CLASS_NUM, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    best_acc = 0.0
    train_steps = len(train_loader)

    for epoch in range(EPOCH):
        # training process
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

            # print info
            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}], loss:{:.3f}".format(epoch+1, EPOCH, loss)

        # evaluation process
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
        print('[epoch %d]: training loss: %.3f, val_accuracy: %.3f' % (epoch+1, running_loss/train_steps, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), SAVE_PATH)
    print('Training best accuracy: ', best_acc)
    print('Finished Training!')


if __name__ == '__main__':
    main()

    # official pretrain weights
    model_urls = {
        "vgg11": 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        "vgg13": 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        "vgg16": 'https://download.pytorch.org/models/vgg16-397923af.pth',
        "vgg19": 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    }

