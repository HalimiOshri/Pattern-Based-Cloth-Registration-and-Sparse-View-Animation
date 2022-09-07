import sys
sys.path.append('/mnt/home/oshrihalimi/KinematicAlignment/utils/')
import tensorIO
import numpy as np
import os
import cv2

def save_keypoints_image(detections_path, save_path):
    type_result_tensor = tensorIO.ReadTensorFromBinaryFile(detections_path)
    type_result_np = type_result_tensor[0].detach().permute(1, 2, 0).numpy()
    type_labels = np.argmax(type_result_np, axis=2)

    # get corners & centers
    corners_mask = 1 * (type_labels == 1)
    centers_mask = 1 * (type_labels == 2)

    output_image = np.zeros(corners_mask.shape + (3,))
    output_image[:, :, 2] = 255 * corners_mask
    output_image[:, :, 1] = 255 * centers_mask

    cv2.imwrite(save_path, output_image)

if __name__ == '__main__':
    # keypoint_detections_path = '/Users/oshrihalimi/Data/CornerTriangulation/corners_center_detector'
    # save_path = '/Users/oshrihalimi/Data/CornerTriangulation/visualization/keypoint_detections'
    #
    # os.makedirs(save_path, exist_ok=True)

    keypoint_detections_path = '/mnt/home/oshrihalimi/capture/keypoint_location_detector_results/corners_centers_detector_unet-5_double_conv_4_conv_layers_cross_entropy_1_sparse_and_1_dense_input_gray_aug_v1/cam400872/'
    save_path = keypoint_detections_path
    for file in ['3000.bt']: #os.listdir(keypoint_detections_path):
        save_keypoints_image(detections_path=os.path.join(keypoint_detections_path, file), save_path=os.path.join(save_path, file.split('.')[0] + '.png'))