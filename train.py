import time
import os
import torch
from utils.util import *
from utils.VerseDataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from losses import DiceLoss, BinaryDiceLoss
from models.UNet.model import UNet3D as UNet
from collections import OrderedDict
import pandas as pd

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


def train(train_loader, model, optimizer, criterion):
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


def validate(val_loader, model, criterion):
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
        # 数据
        train_list = create_list(args.data_path)
        val_list = create_list(args.val_path)
        test_list = create_list(args.test_path)

        # 模型和权重
        if args.multi_gpu is True:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
            net = torch.nn.DataParallel((UNet(residual='pool')).cuda())
        else:
            torch.cuda.set_device(args.gpu_id)
            net = UNet(residual='pool').cuda()

        if args.do_you_wanna_load_weight is True:
            net.load_state_dict(torch.load(args.load_path))

        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = BinaryDiceLoss()
        best_loss = 1
        trigger = 0
        for epoch in range(args.num_epoch):
            print('Epoch [%d/%d]' % (epoch, args.epochs))
            train_log = train(train_loader, net, optimizer, criterion)
            val_log = validate(test_loader, net, optimizer)
            print('loss %.4f - val_loss %.4f'
                  % (train_log['loss'],  val_log['loss']))
            tmp = pd.Series([
                epoch,
                args.lr,
                train_log['loss'],
                val_log['loss'],
            ], index=['epoch', 'lr', 'loss', 'val_loss'])
            log.to_csv('trained_models/%s/log.csv' % args.name, index=False)
            log = log.append(tmp, ignore_index=True)
            log.to_csv('trained_models/%s/log.csv' % args.name, index=False)
            trigger += 1

            if val_log['loss'] < best_loss:
                torch.save(net.state_dict(), 'trained_models/%s/model.pth' % args.name)
                best_loss = val_log['loss']
                print('=> saved best model')
                # 并保持当前最好的保存checkpoint
                checkpoint = {"model_state_dict": net.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch}
                path_checkpoint = "trained_models/%s/checkpoint_%d_epoch.pkl" % (args.name, epoch)
                torch.save(checkpoint, path_checkpoint)
                trigger = 0
            torch.cuda.empty_cache()
