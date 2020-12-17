import time
import os
import torch
from utils.util import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from losses import DiceLoss, BinaryDiceLoss
from models.UNet.model import UNet3D as UNet
from collections import OrderedDict


class AvgMeter(object):
    """
    Acc meter class, use the update to add the current acc
    and self.avg to get the avg acc
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def train(args, train_loader, model, optimizer, criterion):
    losses = AvgMeter()
    model.train()
    for batch_idx, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item())

    log = OrderedDict([
        ('loss', losses.avg),
    ])
    return log


def validate(args, val_loader, model, criterion):
    losses = AvgMeter()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item())

    log = OrderedDict([
        ('loss', losses.avg),
    ])
    return log


def main(args):
    ckpt_path = os.path.join(args.output_path, "Checkpoint")
    log_path = os.path.join(args.output_path, "Log")
    min_pixel = int(args.min_pixel * ((args.patch_size[0] * args.patch_size[1] * args.patch_size[2]) / 100))
    check_dir(args.output_path)
    check_dir(log_path)
    check_dir(ckpt_path)

    if args.do_you_wanna_train is True:
        train_list = create_list(args.data_path)
        val_list = create_list(args.val_path)
        test_list = create_list(args.test_path)
