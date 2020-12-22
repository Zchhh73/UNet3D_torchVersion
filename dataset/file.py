import os
from shutil import copyfile

img_output_dir = "D:\\3Ddata\\Verse_batch1\\data"
label_output_dir = "D:\\3Ddata\\Verse_batch1\\label"

img2_output_dir = "D:\\3Ddata\\Verse_batch2\\data"
label2_output_dir = "D:\\3Ddata\\Verse_batch2\\label"

root1_dir = 'D:\\Data\\Verse_batch1'
root2_dir = 'D:\\Data\\Verse_batch2'
img = 'image.nii.gz'
label = 'mask.nii.gz'

if __name__ == '__main__':
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
