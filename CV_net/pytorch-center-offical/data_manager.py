#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/12/16 9:47
# @Author : ''
# @FileName: data_manager.py
import os
import cv2
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_transform1(mode):
    def subtract_mean(x):
        mean_vector = [91.4953, 103.8827, 131.0912]
        x *= 225
        x[0] -= mean_vector[0]
        x[1] -= mean_vector[1]
        x[2] -= mean_vector[2]

    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop((112, 96)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: subtract_mean(x))])

    if mode == 'val':
        return transforms.Compose([
            transforms.Resize((112, 96)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: subtract_mean(x))])
    else:
        assert "The {} of get_transform is not define!".format(mode)


def get_transform(mode):
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop((112, 96)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if mode == 'val':
        return transforms.Compose([
            transforms.Resize((112, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        assert "The {} of get_transform is not define!".format(mode)


class MyDataset(Dataset):
    def __init__(self, txt_path, resize_tuple=((40, 40), (6, 6)), train=True, transform=None, interpolation=Image.BICUBIC):
        super(MyDataset, self).__init__()
        self._txt_path = txt_path
        self._root = os.path.dirname(self._txt_path)
        self._resize_tuple = resize_tuple
        self._train = train
        self._transform = transform
        self._interpolation = interpolation
        self._class_to_idx, self._classes = self._find_classes()
        self._imgs_path = self._read_imgs()
        self._loader = self._get_loader

    def _find_classes(self):
        classes = [d for d in os.listdir(os.path.join(self._root, 'train')) if os.path.isdir(os.path.join(self._root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {i: classes[i] for i in range(len(classes))}
        json_str = json.dumps(idx_to_class, indent=4)
        with open("classes_indices.json", "w") as f:
            f.write(json_str)
        return class_to_idx, classes

    def _read_imgs(self):
        imgs = []
        if self._train:
            with open(os.path.join(self._txt_path, 'CASIA_train.txt'), 'r') as fh:
                for line in fh:
                    line = line.strip('\n')
                    line = line.rstrip()
                    # words = line.split('/')[-2].split('\\')[1]
                    words = line.split('/')[-2]
                    imgs.append((line, words))  # imgs中包含有train图像路径和标签
        else:
            with open(os.path.join(self._txt_path, 'CASIA_val.txt'), 'r') as fh:
                for line in fh:
                    line = line.strip('\n')
                    line = line.rstrip()
                    words = line.split('/')[-2]
                    # words = line.split('\\')[-2:-1]
                    imgs.append((line, words))  # imgs中包含有val图像路径和标签
        return imgs

    def _lower_resolution(self, img, resolution):
        w_i, h_i = img.size
        r = h_i/float(w_i)
        if self._train:
            res = torch.rand(1).item()
            res = 3 + 5*res
            res = 2**int(res)
        else:
            res = resolution
        if res >= w_i or res >= h_i:
            return img
        if h_i < w_i:
            h_n = res
            w_n = h_n/float(r)
        else:
            w_n = res
            h_n = w_n*float(r)
        img2 = img.resize((int(w_n), int(h_n)), self._interpolation)
        img2 = img2.resize((w_i, h_i), self._interpolation)
        return img2

    @ staticmethod
    def _get_loader(path):
        return Image.fromarray(cv2.imread(path))

    def __getitem__(self, index):
        fn, lable_name = self._imgs_path[index]
        label = self._class_to_idx[lable_name]
        orgin_img = self._loader(fn)
        if self._train:
            img1 = self._lower_resolution(orgin_img, self._resize_tuple[0][0])
            img2 = self._lower_resolution(orgin_img, self._resize_tuple[1][1])
            if self._transform is not None:
                orgin_img = self._transform(orgin_img)
                img1 = self._transform(img1)
                img2 = self._transform(img2)
            else:
                assert "transform method is not defined!"
            return orgin_img, img1, img2, label

        else:
            orgin_img = self._transform(orgin_img)
        return orgin_img, label

    def __len__(self):
        return len(self._imgs_path)


class MyDatasetManger():
    def __init__(self, dataset_path, batch_size=32, num_of_workers=0, use_gpu=True):
        self._batch_size = batch_size
        self._num_of_workers = num_of_workers
        self.pin_memory = True if use_gpu else False
        self._datasets_path = dataset_path
        self._datasets = self._init_datasets()
        self._data_loaders = self._init_data_loaders()

    def _init_datasets(self):
        train_data = MyDataset(self._datasets_path,
                               resize_tuple=((40, 40), (6, 6)),
                               train=True,
                               transform=get_transform(mode='train'),
                               interpolation=Image.BICUBIC)

        val_data = MyDataset(self._datasets_path,
                             resize_tuple=((40, 40), (6, 6)),
                             train=False,
                             transform=get_transform(mode='val'),
                             interpolation=Image.BICUBIC)

        return train_data, val_data

    def _init_data_loaders(self):

        train_data_loader = DataLoader(dataset=self._datasets[0],
                                       batch_size=self._batch_size,
                                       shuffle=True,
                                       num_workers=self._num_of_workers,
                                       pin_memory=self.pin_memory)

        val_data_loader = DataLoader(dataset=self._datasets[1],
                                     batch_size=4,
                                     shuffle=False,
                                     num_workers=self._num_of_workers,
                                     pin_memory=self.pin_memory)

        return train_data_loader, val_data_loader

    def get_loaders(self):
        return self._data_loaders