import sys
sys.path.append('/mnt/home/oshrihalimi/KinematicAlignment/utils/')
import tensorIO
import numpy as np
import os
import cv2

def save_color_image(detections_path, save_path):
    colormap_color = np.array([
        [0, 0, 0],  # black
        [0, 0, 255],  # red
        [0, 255, 0],  # green
        [0, 255, 255],  # yellow
        [255, 0, 0],  # blue
        [255, 0, 255],  # magenta
        [255, 255, 0],  # cyan
        [255, 255, 255]  # white
    ])

    color_labels = tensorIO.ReadTensorFromBinaryFile(detections_path)
    cv2.imwrite(save_path, np.flip(colormap_color[color_labels], axis=2))


if __name__ == '__main__':
    #color_detections_path = '/mnt/home/oshrihalimi/Data/CornerTriangulation/color_detector'
    #save_path = '/Users/oshrihalimi/Data/CornerTriangulation/visualization/color_detections'

    color_detections_path = '/mnt/home/oshrihalimi/capture/color_detector_results/FINAL_clean_detector_sparse_cross_entropy_400_random_keypoints_with_resize_aug_0_v1/cam400872'
    #os.makedirs(save_path, exist_ok=True)

    for file in ['3000.bt']: #os.listdir(color_detections_path):
        save_color_image(detections_path=os.path.join(color_detections_path, file), save_path=os.path.join(color_detections_path, file.split('.')[0] + '.png'))