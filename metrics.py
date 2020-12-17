import numpy as np


def dice_coeff(seg, gt, ratio=0.5):
    """
    function to calculate the dice score
    """
    seg = seg.flatten()
    gt = gt.flatten()
    seg[seg > ratio] = np.float32(1)
    seg[seg < ratio] = np.float32(0)
    dice = float(2 * (gt * seg).sum()) / float(gt.sum() + seg.sum())
    return dice


def rel_abs_vol_diff(y_true, y_pred):
    return np.abs((y_pred.sum() / y_true.sum() - 1) * 100)