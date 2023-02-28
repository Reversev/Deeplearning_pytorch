# ！/usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2023/2/10 16:05
# @Author: 'IReverser'
# @ FileName: train.py
import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data.distributed
from timm.models import create_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import make_dir, AverageMeter, Logger, accuracy, format_time
from vit import create_vit
from deit import create_deit
from datasets import build_dataset
from losses import DistillationLoss

try:
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser("Pytorch Training for DeiT")
    parser.add_argument('--data_path', default="../datasets/flowers", type=str, help='path to dataset')
    parser.add_argument('--data', default="flowers", type=str, help='path to dataset')
    parser.add_argument('--arch', metavar='ARCH', default='deit_tiny', type=str,
                        help='deit_tiny, deit_small, deit_base, deit_base_384')
    parser.add_argument('--input_size', default=224, type=int, help='number of total epochs to training')
    parser.add_argument('--teacher-model', default=None, type=str,
                        help='Name of teacher model to train (default: "resnet34"')

    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to training')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restart)')
    parser.add_argument('-b', '---batch-size', default=64, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total'
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate',
                        dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default:4)')
    parser.add_argument('--model-path', default=' ', type=str, help='loading pretraining model')
    parser.add_argument('--teacher-path', default=' ', type=str, help='loading teacher model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url seed to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id to used')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.9, type=float, help="")
    parser.add_argument('--distillation-tau', default=4.0, type=float, help="")

    parser.add_argument('--suffix', default='', type=str, help='path suffix')

    return parser.parse_args()


def sgd_optimizer(model, lr, momentum, weight_decay):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr
        if 'bias' in key or 'bn' in key:
            apply_weight_decay = 0
            # print('set weight decay=0 for {}'.format(key))
        if 'bias' in key:
            apply_lr = 2 * lr  # Just a Caffe-style common practice. Made no difference.
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer


def train(train_loader, model, criterion, optimizer, args, lr_scheduler):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        output = output[0]
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

    return losses.avg, top1.avg, top5.avg


def train_distill(train_loader, model, criterion, optimizer, args, lr_scheduler):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    for i, (images, target) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(images, output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy((output[0] + output[1]) / 2, target, topk=(1, 5))  # output[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    return losses.avg, top1.avg, top5.avg


def main_worker(gpu, ngpus_per_node, log, args):
    best_acc1 = 0
    st_time = time.time()
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprogressing_distributed:
            # For multiprogressing distributed training, rank needs to be the global rank among all the progress
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # data loader
    train_dataset, args.num_classes = build_dataset(train=True, args=args)
    val_dataset, _ = build_dataset(train=False, args=args)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, drop_last=False)

    print(len(train_loader), len(val_loader))

    # load model or teacher
    if "deit" in args.arch:
        model = create_deit(model_name=args.arch, num_classes=args.num_classes)
    else:
        model = create_vit(model_name=args.arch, num_classes=args.num_classes)

    if not torch.cuda.is_available():
        print("Using CPU, this will be slow")

    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParalled will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # load pretrain model
    if args.model_path != " ":
        print("loading pretrained model from " + args.model_path)
        model.load_state_dict(torch.load(args.model_path))
    model_n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("params: {:3f} M".format(model_n_parameters / 1024 / 1024))

    # define optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = sgd_optimizer(model, args.lr, args.momentum, args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer,
                                     T_max=args.epochs * IMAGENET_TRAINSET_SIZE // args.batch_size // ngpus_per_node)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # map model to be loaded to specified single gpu
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)

            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
            print("=> loaded chekpoint '{}' (epoch: {})".format(args.resume, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # load teacher model
    if args.teacher_model is not None:
        teacher = create_model(args.teacher_model,
                               pretrained=True,
                               num_classes=args.num_classes,
                               global_pool="avg")
        # map model to be loaded to specified single gpu
        loc = "cuda:{}".format(args.gpu)
        # load pretrain model
        if args.teacher_path != " ":
            print("loading pretrained model from " + args.teacher_path)
            if args.teacher_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.teacher_path, map_location=loc, check_hash=True)
            else:
                checkpoint = torch.load(args.teacher_path, map_location=loc)
            print("loading pretrained teacher model from " + args.teacher_path)
            teacher.load_state_dict(checkpoint['model'])
        teacher.eval()
    else:
        teacher = None
    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    if teacher is not None:
        criterion_kd = DistillationLoss(criterion,
                                        teacher,
                                        args.distillation_type,
                                        args.distillation_alpha,
                                        args.distillation_tau).cuda(args.gpu)

    # eval only
    if args.evaluate:
        val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, args)
        row = {"Test_Loss": "%.5f" % val_loss, "Test_Acc@1": "%.3f" % val_acc1, "Test_Acc@5": "%.3f" % val_acc5}
        log.writerow(row)
        print(row)
        return

    # train and eval
    print("=> Start training...")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if teacher is not None:
            train_loss, train_acc1, train_acc5 = train_distill(train_loader, model, criterion_kd, optimizer, args,
                                                               lr_scheduler)
        else:
            train_loss, train_acc1, train_acc5 = train(train_loader, model, criterion, optimizer, args, lr_scheduler)

        # eval
        val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, args)

        row = {"Epoch": str(epoch),
               "Train_Loss": "%.5f" % train_loss, "Train_Acc@1": "%.3f" % train_acc1,
               "Train_Acc@5": "%.3f" % train_acc5,
               "Test_Loss": "%.5f" % val_loss, "Test_Acc@1": "%.3f" % val_acc1, "Test_Acc@5": "%.3f" % val_acc5,
               "Test_best_Acc@1": "%.3f" % best_acc1, "lr": "%.5f" % optimizer.param_groups[0]['lr']}
        row.update({
            'time': format_time(time.time() - st_time),
            'eta': format_time((time.time() - st_time) / (epoch + 1) * (args.epochs - epoch - 1)),
        })
        print(row)
        log.writerow(row)

        if best_acc1 < val_acc1:
            best_acc1 = val_acc1
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                make_dir('./checkpoints/')
                torch.save({'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.state_dict(),
                            'best_acc1': best_acc1,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': lr_scheduler.state_dict()},
                           './checkpoints/model_{}_seed{}_best.pt'.format(args.arch, str(args.seed)))
        else:
            pass
    print("=> Training Finish!")


def main():
    args = parse_option()

    if args.seed is not None:
        if args.seed == 0:
            args.seed = np.random.randint(1000)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True

    args.model_name = '{}_{}_{}_seed{}'.format(args.arch, args.data, args.suffix, str(args.seed))
    make_dir('./logs/')
    if args.evaluate:
        log = Logger(args=args, filename='./logs/' + args.model_name + "_test.txt")
    else:
        log = Logger(args=args, filename='./logs/' + args.model_name + ".txt")
    print(args)

    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, log, args)


if __name__ == '__main__':
    IMAGENET_TRAINSET_SIZE = 3520
    main()
