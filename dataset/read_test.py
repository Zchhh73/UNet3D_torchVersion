import numpy as np
import nibabel as nib
import os

data_path = r'D:\3Ddata\fixed_data\train\data'
mask_path = r'D:\3Ddata\fixed_data\train\label'


def read_data(data_path):
    data_all = []
    label_all = []

    for data in os.listdir(data_path):
        data_all.append(data)
        mask = data.replace("image", "mask")
        label_all.append(mask)

    print(data_all)
    print(label_all)
    data_read = []
    label_read = []
    a_list = []
    b_list = []
    c_list = []
    for i in range(len(data_all)):
        data_read.append(nib.load(os.path.join(data_path, data_all[i])).get_data())
        label_read.append(nib.load(os.path.join(mask_path, label_all[i])).get_data())
        print("******verse data compare******")
        print("num{}, data_name:{}: data shape:{}, label shape:{}; Pixel -- data_max:{}, data_min:{}, "
              "label_max:{}, label_min:{} "
              .format(i, data_all[i].split('.')[0],
                      data_read[i].shape, label_read[i].shape,
                      data_read[i].max(), data_read[i].min(),
                      label_read[i].max(), label_read[i].min()))
        print("===***===" * 10)

        a, b, c = data_read[i].shape
        la, lb, lc = label_read[i].shape
        if a != la or b != lb or c != lc:
            print('data different label size :{}'.format(data_all[i].split('/')[-1].split(".")[0]))

        if label_read[i].max() == 25:
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&label is 25 data'.format(
                data_all[i].split('/')[-1].split(".")[0]))
    return data_read, label_read


if __name__ == '__main__':
    print("read start")
    read_data(data_path)
    print("read stop")
