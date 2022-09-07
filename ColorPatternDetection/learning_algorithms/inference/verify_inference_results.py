import sys
sys.path.append('/mnt/home/oshrihalimi/KinematicAlignment/utils/')
import tensorIO
import numpy as np
from scipy.special import softmax
import cv2

example_path = '/mnt/home/oshrihalimi/capture/keypoint_location_detector_results/corners_centers_detector_unet-5_double_conv_4_conv_layers_cross_entropy_1_sparse_and_1_dense_input_gray_aug_v1/cam400143/3.bt'
save_path = '/mnt/home/oshrihalimi/capture/keypoint_location_detector_results/corners_centers_detector_unet-5_double_conv_4_conv_layers_cross_entropy_1_sparse_and_1_dense_input_gray_aug_v1/cam400143/3.png'

if __name__ == '__main__':
    colormap = np.array([[0, 0, 0], # black - background
                [0, 0, 255], # red - corner
                [0, 255, 0],]) # green - center
    result_tensor = tensorIO.ReadTensorFromBinaryFile(example_path)
    result_np = result_tensor[0].detach().permute(1, 2, 0).numpy()
    labels = np.argmax(result_np, axis=2)
    cv2.imwrite(save_path, colormap[labels.astype(np.uint8), :])
    print("Hi")