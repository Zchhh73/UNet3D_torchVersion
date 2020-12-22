from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class VerseDataset(Dataset):
    def __init__(self, crop_size, resize_scale, dataset_path, mode=None):
        self.crop_size = crop_size
        self.resize_scale = resize_scale
        self.dataset_path = dataset_path
        self.n_labels = 3

        if mode == 'train':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'train_name_list.txt'))
        elif mode == 'val':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'val_name_list.txt'))
        else:
            raise TypeError('Dataset mode error!!! ')

    def __getitem__(self, index):
        data, target = self.get_train_batch_by_index(crop_size=self.crop_size, index=index,
                                                     resize_scale=self.resize_scale)
        return torch.from_numpy(data), torch.from_numpy(target)

    def __len__(self):
        return len(self.filename_list)

    def get_train_batch_by_index(self, crop_size, index, resize_scale=1):
        img, label = self.get_np_data_3d(self.filename_list[index], resize_scale=resize_scale)
        img, label = random_crop_3d(img, label, crop_size)
        return np.expand_dims(img, axis=0), label

    def get_np_data_3d(self, filename, resize_scale=1):
        data_np = sitk_read_raw(self.dataset_path + '/data/' + filename,
                                resize_scale=resize_scale)
        data_np = norm_img(data_np)
        label_np = sitk_read_raw(self.dataset_path + '/label/' + filename.replace('image', 'mask'),
                                 resize_scale=resize_scale)
        return data_np, label_np


def main():
    fix_path = r'D:\python_workplace\UNet3D_torchVersion\dataset\fixed_data\test'
    dataset = VerseDataset([8, 64, 64], 0.5, fix_path, mode='train')
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1, shuffle=True)
    for i, (data, target) in enumerate(dataloader):
        target = to_one_hot_3d(target.long())
        print(data.shape, target.shape)
        plt.subplot(121)
        plt.imshow(data[0, 0, 0])
        plt.subplot(122)
        plt.imshow(target[0, 1, 0])
        plt.show()


if __name__ == '__main__':
    main()
