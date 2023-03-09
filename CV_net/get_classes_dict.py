# ！/usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2023/3/6 22:53
# @Author: 'IReverser'
# @ FileName: get_classes_dict.py
import json
from torchvision import datasets

file_path = "../datasets/flowers/train/"
json_name = "classes_dict.json"

train_dataset = datasets.ImageFolder(root=file_path)
# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)
