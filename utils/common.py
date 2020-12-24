import os
import argparse


# if folder does not exist, create it
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# 参数统计
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
