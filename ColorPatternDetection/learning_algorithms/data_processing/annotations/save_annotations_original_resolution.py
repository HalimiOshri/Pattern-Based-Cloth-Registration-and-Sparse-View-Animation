import os.path

import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from learning_algorithms.data_processing.text_files.util_func import get_file_as_np_int_array, save_np_array_array_to_file

path_dataset_file = '/mnt/home/oshrihalimi/Data/ColorPatternAnnotations/CornersAndCenters/train_list.txt'
path_512_patches = '/mnt/home/oshrihalimi/Data/ColorPatternAnnotations/CornersAndCenters/patches_512_subset'
path_512_annotations = '/mnt/home/oshrihalimi/Data/ColorPatternAnnotations/CornersAndCenters/annotations'
path_patches_original_res = '/mnt/home/oshrihalimi/Data/ColorPatternAnnotations/CornersAndCenters/patches_short_subset'
path_save = '/mnt/home/oshrihalimi/Data/ColorPatternAnnotations/CornersAndCenters/annotations_short'

if __name__ == '__main__':
    ia.seed(1)

    with open(path_dataset_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        patch_path, centers_path, corners_path = line.split()
        patch = cv2.imread(patch_path)
        patch_original_res = cv2.imread(os.path.join(path_patches_original_res, os.path.basename(patch_path)))
        try:
            centers = get_file_as_np_int_array(centers_path)
            corners = get_file_as_np_int_array(corners_path)
        except OSError: #some of the patches are not annotated
            continue
        if centers.shape[0] == 0 or corners.shape[0] == 0:
            continue

        centers[:, :2] = centers[:, :2] - 1 # save as zero based annotations
        corners[:, :2] = corners[:, :2] - 1  # save as zero based annotations

        centers_kpts = KeypointsOnImage([Keypoint(x=center[0], y=center[1]) for center in centers], shape=patch.shape)
        corners_kpts = KeypointsOnImage([Keypoint(x=corner[0], y=corner[1]) for corner in corners], shape=patch.shape)

        seq = iaa.Sequential([
            iaa.Resize({"height": patch_original_res.shape[0], "width": patch_original_res.shape[1]})
        ])

        # Augment keypoints and images.
        centers_kpts_aug = seq(keypoints=centers_kpts)
        corners_kpts_aug = seq(keypoints=corners_kpts)

        # # print coordinates before/after augmentation (see below)
        # # use after.x_int and after.y_int to get rounded integer coordinates
        # for i in range(len(kps.keypoints)):
        #     before = kps.keypoints[i]
        #     after = kps_aug.keypoints[i]
        #     print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
        #         i, before.x, before.y, after.x, after.y)
        #           )

        # image with keypoints before/after augmentation (shown below)
        debug_dir = os.path.join(path_save, 'verify_correctness')
        os.makedirs(debug_dir, exist_ok=True)

        before_dir = os.path.join(debug_dir, 'before')
        os.makedirs(before_dir, exist_ok=True)

        after_dir = os.path.join(debug_dir, 'after')
        os.makedirs(after_dir, exist_ok=True)

        patch = centers_kpts.draw_on_image(patch, size=10)
        patch = corners_kpts.draw_on_image(patch, size=10)
        cv2.imwrite(os.path.join(before_dir, os.path.basename(patch_path)), patch)

        patch_original_res = centers_kpts_aug.draw_on_image(patch_original_res, size=1)
        patch_original_res = corners_kpts_aug.draw_on_image(patch_original_res, size=1)
        cv2.imwrite(os.path.join(after_dir, os.path.basename(patch_path)), patch_original_res)

        original_res_centers = np.stack((np.array([kpt.x_int for kpt in centers_kpts_aug.keypoints]),
                                               np.array([kpt.y_int for kpt in centers_kpts_aug.keypoints]),
                                               centers[:, 2]), axis=1) # concatenate new coords with label
        original_res_corners = np.stack((np.array([kpt.x_int for kpt in corners_kpts_aug.keypoints]),
                                               np.array([kpt.y_int for kpt in corners_kpts_aug.keypoints])), axis=1)

        save_np_array_array_to_file(path=os.path.join(path_save, os.path.basename(centers_path)), array=original_res_centers)
        save_np_array_array_to_file(path=os.path.join(path_save, os.path.basename(corners_path)), array=original_res_corners)