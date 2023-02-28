# ！/usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2023/2/25 20:19
# @Author: 'IReverser'
# @ FileName: datasets.py
import os

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import warnings
warnings.filterwarnings("ignore")


def build_transform(is_train, args):
    resize_im = args.input_size > 32

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(input_size=args.input_size,
                                     is_training=True,
                                     color_jitter=0.3,
                                     auto_augment='rand-m9-mstd0.5-inc1',
                                     interpolation='bicubic',
                                     re_prob=0.25,
                                     re_mode="pixel",
                                     re_count=1)
        # transform = transforms.Compose([transforms.RandomResizedCrop(224),
        #                                   transforms.RandomHorizontalFlip(),
        #                                   transforms.ToTensor(),
        #                                   transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)

        return transform

    t = []
    if resize_im:
        size = int(args.input_size / 0.875)  # 224 / 0.875 = 256
        t.append(transforms.Resize(size, interpolation=InterpolationMode.BICUBIC))  # 3: InterpolationMode.BICUBIC
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset(train, args):
    transform = build_transform(train, args)
    if args.data == "CIFAR10":
        dataset = datasets.CIFAR10(args.data_path, train=train, transform=transform)
        num_classes = 10
    elif args.data == "CIFAR100":
        dataset = datasets.CIFAR100(args.data_path, train=train, transform=transform)
        num_classes = 100
    elif args.data == 'IMNET':
        root = os.path.join(args.data_path, 'train' if train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        num_classes = 1000
    elif args.data == 'flowers':
        root = os.path.join(args.data_path, 'train' if train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        num_classes = 5
    return dataset, num_classes


if __name__ == '__main__':
    import torch
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    root = os.path.join("../datasets/flowers", 'train')
    dataset = datasets.ImageFolder(root, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
                                               num_workers=0, pin_memory=True, drop_last=True)
    for i, (im, target) in enumerate(train_loader):
        print(im.shape, target.shape)
