#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2022/2/10 12:43
# @Author : ''
# @FileName: misc.py
from __future__ import absolute_import
import os
import sys
import torch
import torch.backends.cudnn


def setup_seed(an_int):
    torch.manual_seed(an_int)
    torch.cuda.manual_seed(an_int)
    torch.cuda.manual_seed_all(an_int)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mkdir_file(path):
    if not os.path.exists(path):
        os.makedirs(path)


def format_time(seconds):
    days = int(seconds / (3600 * 24))
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

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
    if secondsf > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, args, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_file(os.path.dirname(fpath))
            self.file = open(fpath, 'w')
        # write model configure at the top of file
        for arg in vars(args):
            self.file.write(arg + ': ' + str(getattr(args, arg)) + '\n')
        self.file.flush()

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
            self.file.flush()

    def write_dict(self, dict):
        if self.file is not None:
            for k in dict:
                self.file.write(k + ': ' + dict[k] + ' ')
            self.file.write('\n')
            self.file.flush()

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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


def save_checkpoint(state, filename='checkpoint.pt'):
    torch.save(state, filename)


@torch.no_grad()
def accuracy(output, target, topk=(1, )):
    """"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(axis=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))  # element-wise multiply in matrices
    return res
