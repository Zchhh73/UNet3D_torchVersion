import os
from shutil import copyfile

img_output_dir = r"F:\Verse_Data\Data\Verse_train\data"
label_output_dir = r"F:\Verse_Data\Data\Verse_train\label"

img2_output_dir = r"F:\Verse_Data\Data\Verse_test\data"
label2_output_dir = r"F:\Verse_Data\Data\Verse_test\label"

root1_dir = r'F:\Verse_Data\3Ddata\Verse_train'
root2_dir = r'F:\Verse_Data\3Ddata\Verse_test'
img = 'image.nii.gz'
label = 'mask.nii.gz'


def main(mode):
    if mode == 'train':
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)
        if not os.path.exists(label_output_dir):
            os.makedirs(label_output_dir)
        for root, dirs, files in os.walk(root1_dir):
            for i in range(len(dirs)):
                old_img_path = os.path.join(root, dirs[i], img)
                old_mask_path = os.path.join(root, dirs[i], label)
                new_img_name = img.split('.')[0] + str(i) + '.nii.gz'
                new_label_name = label.split('.')[0] + str(i) + '.nii.gz'
                new_img_path = os.path.join(img_output_dir, new_img_name)
                new_mask_path = os.path.join(label_output_dir, new_label_name)
                copyfile(old_img_path, new_img_path)
                copyfile(old_mask_path, new_mask_path)
                print(dirs[i] + "复制成功")
    elif mode == 'test':
        if not os.path.exists(img2_output_dir):
            os.makedirs(img2_output_dir)
        if not os.path.exists(label2_output_dir):
            os.makedirs(label2_output_dir)
        for root, dirs, files in os.walk(root2_dir):
            for i in range(len(dirs)):
                old_img_path = os.path.join(root, dirs[i], img)
                old_mask_path = os.path.join(root, dirs[i], label)
                new_img_name = img.split('.')[0] + str(i) + '.nii.gz'
                new_label_name = label.split('.')[0] + str(i) + '.nii.gz'
                new_img_path = os.path.join(img_output_dir, new_img_name)
                new_mask_path = os.path.join(label_output_dir, new_label_name)
                copyfile(old_img_path, new_img_path)
                copyfile(old_mask_path, new_mask_path)
                print(dirs[i] + "复制成功")
    else:
        print("None")


if __name__ == '__main__':
    main('train')
