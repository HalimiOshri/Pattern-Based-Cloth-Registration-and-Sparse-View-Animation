import sys
sys.path.append('/mnt/home/oshrihalimi/KinematicAlignment/utils/')
import tensorIO
import numpy as np
from scipy.special import softmax
import cv2
import time

example_path = '/mnt/home/oshrihalimi/capture/color_detector_results/FINAL_clean_detector_sparse_cross_entropy_400_random_keypoints_with_resize_aug_0_v1/cam400929/0.bt'
save_path = '/mnt/home/oshrihalimi/capture/color_detector_results/FINAL_clean_detector_sparse_cross_entropy_400_random_keypoints_with_resize_aug_0_v1/cam400929/0.png'

if __name__ == '__main__':
    rgb_map = np.array([
        [0, 0, 0],  # black
        [0, 0, 255],  # red
        [0, 255, 0],  # green
        [0, 255, 255],  # yellow
        [255, 0, 0],  # blue
        [255, 0, 255],  # magenta
        [255, 255, 0],  # cyan
        [255, 255, 255]  # white
    ])
    start_time = time.time()
    labels_tensor = tensorIO.ReadTensorFromBinaryFile(example_path)
    labels_np = labels_tensor.detach().numpy()
    print("--- %s seconds ---" % (time.time() - start_time))

    cv2.imwrite(save_path, rgb_map[labels_np.astype(np.uint8), :])

    print("Hi")