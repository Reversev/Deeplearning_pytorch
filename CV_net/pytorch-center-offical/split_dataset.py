#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/8/9 23:16
# @Author : ''
# @FileName: split_dataset.py
# download: https://www.kaggle.com/alxmamaev/flowers-recognition

import os
from shutil import copy, rmtree
import random
random.seed(0)
split_ratio = 0.2
DATASET = 'datasets'
ORIGIN_PATH = 'flowers1'
# ORIGIN_PATH = 'LFW'


def make_file(path):
    if os.path.exists(path):
        rmtree(path)  # the directory of path is rebuild when it exist
    os.makedirs(path)


def main():
    cwd = os.getcwd()
    data_root = os.path.join(cwd, DATASET)
    print(data_root)
    origin_path = os.path.join(data_root, ORIGIN_PATH)
    assert os.path.exists(origin_path), 'file path: {} is not exist'.format(origin_path)
    classes = [cla for cla in os.listdir(origin_path) if os.path.join(origin_path, cla)]

    # build and save training samples and directory
    train_root = os.path.join(data_root, os.path.join(ORIGIN_PATH, 'train'))
    make_file(train_root)
    for cla in classes:
        make_file(os.path.join(train_root, cla))

    # build and save test samples and directory
    val_root = os.path.join('../', data_root, os.path.join(ORIGIN_PATH, 'val'))
    make_file(val_root)
    for cla in classes:
        make_file(os.path.join(val_root, cla))  # build directory for each categories

    for cla in classes:
        cla_path = os.path.join(origin_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # randomly sampling index of val dataset
        eval_index = random.sample(images, k=int(num*split_ratio))
        for index, image in enumerate(images):
            if image in eval_index:
                # contribute eval samples
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
                with open(os.path.join(origin_path, 'val.txt'), 'a') as f:
                    f.write(os.path.join(new_path, image) + '\n')
            else:
                # contribute train samples
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
                with open(os.path.join(origin_path, 'train.txt'), 'a') as f:
                    f.write(os.path.join(new_path, image) + '\n')
            print('\r[{}] processing [{}/{}]'.format(cla, index+1, num), end="")  # processing bar
    print("\nSplit dataset Finish!")


if __name__ == '__main__':
    main()












