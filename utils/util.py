import os


# if folder does not exist, create it
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 参数统计
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)