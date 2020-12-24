import numpy as np
import SimpleITK as sitk
import os
from utils.common import check_dir

root_path = r'/hdd/chenkecheng/zchhh_data/3Dtrain/fixed_data/train/data'
root_mask_path = r'/hdd/chenkecheng/zchhh_data/3Dtrain/fixed_data/train/label'

trainImage = r"/hdd/chenkecheng/zchhh_data/VerseData/trainImage"
trainMask = r"/hdd/chenkecheng/zchhh_data/VerseData/trainMask"


def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9
        return tmp


def crop_ceter(img, croph, cropw):
    # for n_slice in range(img.shape[0]):
    height, width = img[0].shape
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    return img[:, starth:starth + croph, startw:startw + cropw]


def preprocess():
    check_dir(trainImage)
    check_dir(trainMask)
    BLOCK_SIZE = BLOCKSIZE = (32, 160, 160)
    for index in range(len(os.listdir(root_path))):
        print(os.path.join(root_path, "image" + str(index) + ".nii.gz"))
        # 1、读取数据
        img_path = os.path.join(root_path, "image" + str(index) + ".nii.gz")
        mask_path = os.path.join(root_mask_path, "mask" + str(index) + ".nii.gz")
        img_src = sitk.ReadImage(img_path, sitk.sitkInt16)
        mask_src = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        img_array = sitk.GetArrayFromImage(img_src)
        mask_array = sitk.GetArrayFromImage(mask_src)
        # 2、对四个模态分别进行标准化
        img_array_nor = normalize(img_array)
        # 3、裁剪
        img_crop = crop_ceter(img_array_nor, 160, 160)
        mask_crop = crop_ceter(mask_array, 160, 160)
        # 4、分块处理
        patch_block_size = BLOCKSIZE
        numberxy = patch_block_size[1]
        numberz = 8  # patch_block_size[0]
        width = np.shape(img_crop)[1]
        height = np.shape(img_crop)[2]
        imagez = np.shape(img_crop)[0]
        block_width = np.array(patch_block_size)[1]
        block_height = np.array(patch_block_size)[2]
        blockz = np.array(patch_block_size)[0]
        stridewidth = (width - block_width) // numberxy
        strideheight = (height - block_height) // numberxy
        stridez = (imagez - blockz) // numberz
        step_width = width - (stridewidth * numberxy + block_width)
        step_width = step_width // 2
        step_height = height - (strideheight * numberxy + block_height)
        step_height = step_height // 2
        step_z = imagez - (stridez * numberz + blockz)
        step_z = step_z // 2
        hr_img_samples_list = []
        hr_mask_samples_list = []
        patchnum = []
        for z in range(step_z, numberz * (stridez + 1) + step_z, numberz):
            for x in range(step_width, numberxy * (stridewidth + 1) + step_width, numberxy):
                for y in range(step_height, numberxy * (strideheight + 1) + step_height, numberxy):
                    if np.max(mask_crop[z:z + blockz, x:x + block_width, y:y + block_height]) != 0:
                        print("切%d" % z)
                        patchnum.append(z)
                        hr_img_samples_list.append(img_crop[z:z + blockz, x:x + block_width, y:y + block_height])
                        hr_mask_samples_list.append(mask_crop[z:z + blockz, x:x + block_width, y:y + block_height])
        samples_img = np.array(hr_img_samples_list).reshape(
            (len(hr_img_samples_list), blockz, block_width, block_height))
        mask_samples = np.array(hr_mask_samples_list).reshape(
            (len(hr_mask_samples_list), blockz, block_width, block_height))
        samples, imagez, height, width = np.shape(samples_img)[0], np.shape(samples_img)[1], \
                                         np.shape(samples_img)[2], np.shape(samples_img)[3]
        print("samples:" + str(samples))
        print("imagez:" + str(imagez))
        print("height:" + str(height))
        print("width:" + str(width))

        for j in range(samples):
            imagearray = np.zeros((imagez, height, width, 3), np.float)
            datapath = os.path.join(trainImage, "image" + str(index) + "_" + str(patchnum[j]) + ".npy")
            maskpath = os.path.join(trainMask, "mask" + str(index) + "_" + str(patchnum[j]) + ".npy")
            image = samples_img[j, :, :, :]
            image = image.astype(np.float)
            np.save(datapath, imagearray)
            print(datapath + "处理完成")
            MaskArray = np.zeros((imagez, height, width, 3), np.uint8)
            mask_one_sample = mask_samples[j, :, :, :]
            np.save(maskpath, MaskArray)
            print(maskpath + "处理完成")

    print("Done!")


if __name__ == '__main__':
    preprocess()
