import SimpleITK as sitk
import os
import re
import numpy as np
import random
import glob
import scipy.ndimage.interpolation as interpolation
from sklearn.model_selection import train_test_split
import scipy
import torch
import torch.utils.data

interpolator_image = sitk.sitkLinear
interpolator_mask = sitk.sitkLinear

_interpolator_image = 'linear'
_interpolator_mask = 'linear'

Segmentation = True


def create_list(data_path):
    data_list = glob.glob(os.path.join(data_path, '*'))
    mask_name = 'mask.nii.gz'
    img_name = 'image.nii.gz'
    data_list.sort()
    list_all = [{
        'image': os.path.join(path, img_name),
        'mask': os.path.join(path, mask_name)
    } for path in data_list]
    return list_all


def resize(img, new_size, interpolator):
    dimension = img.GetDimension()
