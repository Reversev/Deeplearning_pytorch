#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/11/26 17:12
# @Author : 'IReverser'
# @FileName: train.py
from __future__ import print_function, absolute_import
import os
import json
import random
import argparse
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.misc import setup_seed, format_time, Logger, AverageMeter, accuracy, save_checkpoint
from nonlocalnn import model_list


def argspar():
    parser = argparse.ArgumentParser(description='Swin Transformer PyTorch Training')
    parser.add_argument('--dataset-name', '-d', default='flowers')
    parser.add_argument('--img_path', '-p', default='../datasets/', help="images path")
    parser.add_argument('--manual-seed', default=0, type=int, help="random seed in range [1, 10000] if not define")

    parser.add_argument('--batch_size', type=int, default=60,
                        help='input batch size for training and eval(default: 60)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--step-size', type=int, default=0,
                        help='lr schedule with step_size penalty (default: 10), if step_size=0 not use schedule')
    parser.add_argument('--logs', default='logs', type=str, metavar='PATH', help='path to save logs')

    parser.add_argument('--model-name', type=str, default='resnet56',
                        help='model name: resnet14, resnet20, resnet56')

    parser.add_argument('--drop_rate', type=float, default=0.0,
                        help='if or not use dropout to regularize (default: 0.0)')

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default=False, type=bool,
                        help='if not load checkpoint (default: False)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--load_model', default='./resnet_224.pth', type=str,
                        help='file name of save checkpoint')
    args = parser.parse_args()
    return args


@torch.no_grad()
def eval(net, val_loader, device):
    # evaluation process
    net.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    val_bar = tqdm(val_loader)
    for val_data in val_bar:
        val_images, val_labels = val_data
        val_images, val_labels = val_images.to(device), val_labels.to(device)
        outputs = net(val_images)
        prec1, prec5 = accuracy(outputs, val_labels, topk=(1, 5))
        top1.update(prec1[0], val_images.size(0))
        top5.update(prec5[0], val_images.size(0))
    return top1, top5


def main():
    args = argspar()
    logger = Logger(args=args,
                    fpath=os.path.join(args.logs, 'log_' + args.model_name + '_' + args.dataset_name + '.txt'))
    print(args)
    seed = args.manual_seed if args.manual_seed else random.randint(1, 1000)
    setup_seed(seed)

    SAVE_PATH = os.path.join(args.checkpoint, args.model_name + '_' + args.dataset_name + '_best.pth')
    if not os.path.exists(os.path.dirname(SAVE_PATH)):
        os.mkdir(os.path.dirname(SAVE_PATH))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.write('Use device: {}\n'.format(device))  # print used device during the training

    data_transform = {
        "train": transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
                        transforms.Resize(256),  # resize from smallest edge
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = os.path.join(args.img_path, args.dataset_name)
    assert os.path.join(image_path), "{} path does not exist!".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])

    net_count = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of network
    logger.write("Use {} dataloader workers every process \n".format(net_count))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True, num_workers=net_count)

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=net_count)

    logger.write("Use {} images for training, {} images for validation.\n".
                 format(len(train_dataset), len(val_dataset)))

    # transform labels
    dataset_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in dataset_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open("classes_indices.json", "w") as f:
        f.write(json_str)

    net = model_list.get(args.model_name)(num_classes=len(train_dataset.classes))

    start_epoch = args.start_epoch
    net = net.to(device)
    logger.write('Total params: {:.2f}M \n'.format(sum(p.numel() for p in net.parameters()) / (1000 * 1000)))
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(),
                            lr=args.lr,
                            weight_decay=5e-04)
    best_acc = 0.0

    LOAD_PATH = os.path.join(args.checkpoint, args.load_model)
    if os.path.exists(LOAD_PATH) and args.resume:
        assert os.path.isfile(LOAD_PATH), 'Error: no checkpoint directory found!'
        logger.write("load weights file: {}\n".format(LOAD_PATH))
        checkpoint = torch.load(LOAD_PATH)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        best_acc = checkpoint['best_acc']
        optimizer.load_state_dict(checkpoint['optimizer'])

    if os.path.exists(LOAD_PATH) and args.freeze_layers:
        for name, para in net.named_parameters():
            if 'head' not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    if args.step_size > 0:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode="min",
                                                   factor=0.5,
                                                   patience=args.step_size,
                                                   verbose=True,
                                                   min_lr=1.e-7,
                                                   threshold=0.1)
    # train process
    st_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # training process
        net.train()
        loss_avg = {}
        train_correct = AverageMeter()
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            losses = {}
            outputs = net(images)
            loss = loss_function(outputs, labels)
            losses['cls_loss'] = loss
            loss.backward()
            optimizer.step()

            # print info
            for key in losses:
                if not key in loss_avg:
                    loss_avg[key] = AverageMeter()
                else:
                    loss_avg[key].update(losses[key])
            # calculate running average of accuracy
            train_top1 = accuracy(outputs, labels, topk=(1, ))
            train_correct.update(train_top1[0], labels.size(0))

        top1, top5 = eval(net, val_loader, device)
        if args.step_size > 0:
            scheduler.step(loss_avg['cls_loss'].avg)

        # save results
        if top1.avg > best_acc:
            best_acc = top1.avg
            torch.save(net.state_dict(), SAVE_PATH)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(args.checkpoint, 'checkpoint_' + args.model_name + '_best.pt'))

        loss_avg = {k: loss_avg[k].val for k in loss_avg}
        print_bar = {"epoch": str(epoch),
                     "train_acc": '%.2f' % train_correct.avg,
                     "test1_acc": '%.2f' % top1.avg,
                     "test5_acc": '%.2f' % top5.avg,
                     "best_acc": '%.3f' % best_acc.data,
                     "loss": '%.5f' % (sum(loss_avg.values())),
                     "lr": '%.5f' % (optimizer.state_dict()['param_groups'][0]['lr'])}
        loss_avg = {k: '%.5f' % loss_avg[k] for k in loss_avg}
        print_bar.update(loss_avg)
        print_bar.update({"time": format_time(time.time() - st_time),
                          "eta": format_time((time.time() - st_time) / (epoch + 1) * (args.epochs - epoch - 1))})
        print(print_bar)
        logger.write_dict(print_bar)

    logger.write('Training best accuracy: {}\n'.format(best_acc.data))
    logger.write('Finished Training!\n')


if __name__ == '__main__':
    main()


