import os
from skimage import transform
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

data_path = r'F:\Verse_Data\3Ddata\fixed_data\train\data'
mask_path = r'F:\Verse_Data\3Ddata\fixed_data\train\label'


# read data
def read_data(datapath, maskpath):
    data_all = []
    label_all = []

    for data in os.listdir(datapath):
        data_all.append(data)
        mask = data.replace("image", "mask")
        label_all.append(mask)

    # print(data[0], label[0])
    new_data = []
    new_label = []

    for i in range(len(data_all)):
        new_data.append(sitk.GetArrayFromImage(sitk.ReadImage(datapath + '/' + data_all[i])))
        new_label.append(sitk.GetArrayFromImage(sitk.ReadImage(maskpath + '/' + label_all[i])))
        print("******verse data compare******")
        print("num{}, data_name:{}: data shape:{}, label shape:{}; Pixel -- data_max:{}, data_min:{}, "
              "label_max:{}, label_min:{} "
              .format(i, data_all[i].split('.')[0],
                      new_data[i].shape, new_label[i].shape,
                      new_data[i].max(), new_data[i].min(),
                      new_label[i].max(), new_label[i].min()))
        print("===***===" * 10)

        a, b, c = new_data[i].shape
        la, lb, lc = new_label[i].shape
        if a != la or b != lb or c != lc:
            print('data different label size :{}'.format(data_all[i].split('/')[-1].split(".")[0]))

        if new_label[i].max() == 25:
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&label is 25 data'.format(
                data_all[i].split('/')[-1].split(".")[0]))

    return new_data, new_label


def normalize(image, label):
    MIN_BOUND = -100.0
    MAX_BOUND = 400.0
    # image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    # image[image > 1] = 1
    # image[image < 0] = 0
    for i in range(len(image)):
        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
        # image[i] = (image[i] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        # image[i][image[i] > 1] = 1
        # image[i][image[i] < 0] = 0
        image[i] = transform.resize(image[i].astype(np.float32), (64, 128, 128))  # size need to set up d
        label[i] = transform.resize(label[i].astype(np.uint8), (64, 128, 128))

    image = (image - np.mean(image)) / (np.std(image))

    image = np.array(image)  # B,D,H,W
    label = np.array(label)
    # image = np.transpose(image, (0, 2, 3, 1))  # B,H,W,D
    # label = np.transpose(label, (0, 2, 3, 1))
    print(image.shape)
    print(label.shape)

    image = (image * 255).astype(np.float32)
    image = image[np.newaxis, :]  # C,B,D,H,W
    image = np.transpose(image, (1, 0, 2, 3, 4))  # B,C,D,H,W

    label = label[np.newaxis, :]  # C,B,D,H,W
    label = np.transpose(label, (1, 0, 2, 3, 4))  # B,C,D,H,W
    label = (label * 255).astype(np.uint8)
    # label[label == 25] = 24  # one-hot encode the target
    # binary class
    # if label.all() == 0:
    #     label = 0
    # else:
    #     label = 1
    # label[label > 0] = 1
    shp = label.shape[0]
    print(shp)
    # label = label.reshape(-1)
    # print(label)
    label = torch.nn.functional.one_hot(label, 25)
    # label = label.reshape(shp, 128, 128, 64, 25)
    label = label.reshape(shp, 128, 128, 64, 25)  # 0 2 3 1
    # label = label.reshape(shp, 128, 128, 64, 2)  # 0 2 3 1

    np.save("x_training", image.astype(np.float32))  # save data  np.float32
    np.save("y_training", label.astype(np.uint8))  # save label

    return image, label


def concatenate(x1, x2, x3, y1, y2, y3):
    shuffle = list(zip(x1, x2, x3, y1, y2, y3))
    np.random.shuffle(shuffle)
    x = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3), axis=0)

    np.save("x_training.npy", x.astype(np.float32))
    np.save("y_training.npy", y.astype(np.uint8))

    return x, y


if __name__ == "__main__":
    # all train data set
    print("***********save data start***********")
    a, b = read_data(data_path, mask_path)
    normalize(a, b)
    print("***********save data end***********")
    #
    # print("***********concatenate data start***********")
    # x1 = np.load("x_training1.npy").astype(np.float32)
    # x2 = np.load("x_training2.npy").astype(np.float32)
    # x3 = np.load("x_training3.npy").astype(np.float32)
    # y1 = np.load("y_training1.npy").astype(np.uint8)
    # y2 = np.load("y_training2.npy").astype(np.uint8)
    # y3 = np.load("y_training3.npy").astype(np.uint8)
    # concatenate(x1, x2, x3, y1, y2, y3)
    # print("***********concatenate data end***********")

    # print("***********Visualization normalization data start***********")
    # Visualization()
    # print("***********Visualization normalization data end***********")
