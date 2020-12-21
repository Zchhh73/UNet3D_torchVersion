import time
import os
import torch
from utils.util import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from losses import DiceLoss, BinaryDiceLoss
from models.UNet.model import UNet3D as UNet
from collections import OrderedDict
from utils.VerseDataset import *
import utils.VerseDataset as VerseDataset
from metrics import dice_coeff
from init import InitParser


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


def train(model, train_loader, optimizer, criterion):
    model.train()
    losses = AvgMeter()
    for batch_idx, (input, target) in enumerate(train_loader):
        input = Variable(input.cuda())
        target = Variable(target.cuda())
        output = model(input)
        loss = criterion(output, target)
        output = output.squeeze().data.cpu().numpy()
        label = label.squeeze().cpu().numpy()
        dice = dice_coeff(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item())

        if batch_idx % 10 == 0:
            print("Train Batch {} || Loss: {:.4f} | Training Dice: {:.4f}".format(str(batch_idx).zfill(4), losses.val,
                                                                                  dice))
    return losses.avg


def validate(model, val_loader):
    model.eval()
    losses = AvgMeter()
    for i, (input, label) in enumerate(val_loader):
        data = Variable(data.cuda())
        output = model(data)
        output = output.squeeze().data.cpu().numpy()
        label = label.squeeze().cpu().numpy()
        losses.update(dice_coeff(output, label))
        # print("Test {} || Dice: {:.4f}".format(str(batch_idx).zfill(4), test_dice_meter.val))
    return losses.avg


def main(args):
    output_path = os.path.join(args.output_path, args.name)
    ckpt_path = os.path.join(output_path, "Checkpoint")
    log_path = os.path.join(output_path, "Log")
    min_pixel = int(args.min_pixel * ((args.patch_size[0] * args.patch_size[1] * args.patch_size[2]) / 100))
    check_dir(output_path)
    check_dir(log_path)
    check_dir(ckpt_path)

    print('Config --------')
    for arg in vars(args):
        print('%s,%s' % (arg, getattr(args, arg)))
    print('---------------')

    with open(os.path.join(output_path, "model_args.txt"), 'w') as f:
        for arg in vars(args):
            print('%s,%s' % (arg, getattr(args, arg)), file=f)

    if args.do_you_wanna_train is True:
        train_list = create_list(args.data_path)
        val_list = create_list(args.val_path)
        test_list = create_list(args.test_path)

        for i in range(args.increase_factor_data):
            train_list.extend(train_list)
            val_list.extend(val_list)
            test_list.extend(test_list)

        print('Numbers of train patches per epoch:', len(train_list))
        print('Numbers of val patches per epoch:', len(val_list))
        print('Numbers of test patches per epoch:', len(test_list))

        trainTransforms = [
            VerseDataset.Resample(args.new_resolution, args.resample),
            VerseDataset.Augmentation(),
            VerseDataset.Padding((args.patch_size[0], args.patch_size[1], args.patch_size[2])),
            VerseDataset.RandomCrop((args.patch_size[0], args.patch_size[1], args.patch_size[2]), args.drop_ratio,
                                    min_pixel),

        ]
        valTransforms = [
            VerseDataset.Resample(args.new_resolution, args.resample),
            VerseDataset.Padding((args.patch_size[0], args.patch_size[1], args.patch_size[2])),
            VerseDataset.RandomCrop((args.patch_size[0], args.patch_size[1], args.patch_size[2]), args.drop_ratio,
                                    min_pixel),

        ]

        trainSet = VerseDataset.VerseDataset(train_list, transforms=trainTransforms, train=True)
        valSet = VerseDataset.VerseDataset(val_list, transforms=valTransforms, test=True)
        testSet = VerseDataset.VerseDataset(test_list, transforms=valTransforms, test=True)

        trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=True)
        valLoader = DataLoader(valSet, batch_size=args.batch_size, shuffle=False)
        testLoader = DataLoader(testSet, batch_size=args.batch_size, shuffle=False)

        if args.multi_gpu is True:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
            model = torch.nn.DataParallel((UNet(residual='pool')).cuda())
        else:
            torch.cuda.set_device(args.gpu_id)
            model = UNet(residual='pool').cuda()
        if args.do_you_wanna_load_weights is True:
            model.load_state_dict(torch.load(args.load_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = BinaryDiceLoss()
        best_dice = 0.
        for epoch in range(args.init_epoch, args.init_epoch + args.num_epoch):
            start_time = time.time()
            epoch_loss = train(model, trainLoader, optimizer, criterion)
            epoch_dice_val = validate(model, valLoader)
            epoch_dice_test = validate(model, testLoader)
            epoch_time = time.time() - start_time

            info_line = "Epoch {} || Loss: {:.4f} | Time(min): {:.2f} |Validation Dice: {:.4f} | Testing Dice: {:.4f}" \
                .format(str(epoch).zfill(3), epoch_loss, epoch_time / 60, epoch_dice_val, epoch_dice_test)

            open(os.path.join(log_path, 'train_log.txt'), 'a').write(info_line + '\n')

            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(ckpt_path, "Network_{}.pth.gz".format(epoch)))
            if epoch_dice_val > best_dice:
                best_dice = epoch_dice_val
                torch.save(model.state_dict(), os.path.join(ckpt_path, "Best_Dice.pth.gz"))
        '''
        if args.do_you_wanna_check_accuracy is True:

            if args.multi_gpu is True:
                os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # Multi-gpu selector for training
                model = torch.nn.DataParallel((UNet(residual='pool')).cuda())  # load the network Unet

            else:
                torch.cuda.set_device(args.gpu_id)
                model = UNet(residual='pool').cuda()

            model.load_state_dict(torch.load('History/Checkpoint/Best_Dice.pth.gz'))

            train_list = create_list(args.data_path)
            val_list = create_list(args.val_path)
            test_list = create_list(args.test_path)

            print("Checking accuracy on validation set")
            Dice_val = check_accuracy_model(model, val_list, args.resample, args.new_resolution, args.patch_size[0],
                                            args.patch_size[1], args.patch_size[2],
                                            args.stride_inplane, args.stride_layer)

            print("Checking accuracy on testing set")
            Dice_test = check_accuracy_model(model, test_list, args.resample, args.new_resolution, args.patch_size[0],
                                             args.patch_size[1], args.patch_size[2],
                                             args.stride_inplane, args.stride_layer)

            print("Checking accuracy on training set")
            Dice_train = check_accuracy_model(model, train_list, args.resample, args.new_resolution, args.patch_size[0],
                                              args.patch_size[1], args.patch_size[2],
                                              args.stride_inplane, args.stride_layer)

            print("Dice_val:", Dice_val, "Dice_test:", Dice_test, "Dice_train:", Dice_train)
            '''


if __name__ == '__main__':
    parsers = InitParser()
    main(parsers)
