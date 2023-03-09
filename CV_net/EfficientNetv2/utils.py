# ！/usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2023/2/12 10:12
# @Author: 'IReverser'
# @ FileName: utils.py
import os
import torch


def make_dir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger():
    def __init__(self, args, filename='log.txt'):

        self.filename = filename
        self.file = open(filename, 'a')
        # Write model configuration at top of file
        for arg in vars(args):
            self.file.write(arg+': '+str(getattr(args, arg))+'\n')
        self.file.flush()

    def writerow(self, row):
        for k in row:
            self.file.write(k+': '+row[k]+'  ')
        self.file.write('\n')
        self.file.flush()

    def close(self):
        self.file.close()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # print(output)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        # print(pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100./batch_size))
        return res

# class confision_matrix(object):
