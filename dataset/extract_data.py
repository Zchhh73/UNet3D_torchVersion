import os
from skimage import transform
import nibabel as nib
import numpy as np
import SimpleITK as sitk


data_path = r'D:\3Ddata\Verse_batch1\data'
mask_path = r'D:\3Ddata\Verse_batch1\label'


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
        image[i] = transform.resize(image[i].astype(np.float32), (64, 128, 96))  # size need to set up d

        label[i] = transform.resize(label[i].astype(np.uint8), (64, 128, 96))

    image = (image - np.mean(image)) / (np.std(image))

    image = np.array(image)
    label = np.array(label)
    image = np.transpose(image, (0, 2, 3, 1))
    label = np.transpose(label, (0, 2, 3, 1))

    image = (image * 255).astype(np.float32)
    image = image[np.newaxis, :]
    image = np.transpose(image, (1, 2, 3, 4, 0))

    label = label[np.newaxis, :]
    label = np.transpose(label, (1, 2, 3, 4, 0))
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
    label = label.reshape(-1)
    print(label)
    label = np_utils.to_categorical(label).astype(np.uint8)
    # label = label.reshape(shp, 128, 96, 64, 25)
    label = label.reshape(shp, 128, 96, 64, 26)  # 0 2 3 1
    # label = label.reshape(shp, 128, 96, 64, 2)  # 0 2 3 1

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
