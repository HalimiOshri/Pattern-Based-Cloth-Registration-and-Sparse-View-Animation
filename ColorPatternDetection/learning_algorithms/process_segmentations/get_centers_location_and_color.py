import os
import sys
sys.path.append('/mnt/home/oshrihalimi/KinematicAlignment/utils/')
import tensorIO
import numpy as np
from scipy.special import softmax
import cv2

save_path = '/mnt/home/oshrihalimi/capture/keypoints_sets/explore/'

camera = 400936 #400143 #400894 #400936
frame = 1000

path_segmentations = '/mnt/captures/studies/pilots/sociopticon/Aug18/s--20210823--1323--0000000--pilot--patternCloth/experimental/segmentation_part/predictions/segmentation/'
path_color_detections = '/mnt/home/oshrihalimi/capture/color_detector_results/FINAL_clean_detector_sparse_cross_entropy_400_random_keypoints_with_resize_aug_0_v1/'
path_type_detections = '/mnt/home/oshrihalimi/capture/keypoint_location_detector_results/corners_centers_detector_unet-5_double_conv_4_conv_layers_cross_entropy_1_sparse_and_1_dense_input_gray_aug_v1'

# produce 2 sets of reliable detected corners & centers with reliable color attribute for the centers
# 211 cameras x 12k frames < 10s per frame on 1 cpu

segmentation_example_path = os.path.join(path_segmentations, f'cam{camera}', f'image{frame:04d}.png')
color_example_path = os.path.join(path_color_detections, f'cam{camera}', f'{frame}.bt')
type_example_path = os.path.join(path_type_detections, f'cam{camera}', f'{frame}.bt')

save_path = f'/mnt/home/oshrihalimi/capture/keypoints_sets/explore/cam_{camera}_frame_{frame}'

# PARAMETERS
corners_size_thresh_relative_median = 5
centers_size_thresh_relative_median = 5

if __name__ == '__main__':
    # get type
    colormap_type = np.array([[0, 0, 0], # black - background
                [0, 0, 255], # red - corner
                [0, 255, 0],]) # green - center
    type_result_tensor = tensorIO.ReadTensorFromBinaryFile(type_example_path)
    type_result_np = type_result_tensor[0].detach().permute(1, 2, 0).numpy()
    type_labels = np.argmax(type_result_np, axis=2)
    type_image = colormap_type[type_labels.astype(np.uint8), :]
    cv2.imwrite(save_path + '_type.png', type_image)

    # get color
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
    color_labels_tensor = tensorIO.ReadTensorFromBinaryFile(color_example_path)
    color_labels_np = color_labels_tensor.detach().numpy().astype(np.uint8)
    color_labels_image = colormap_color[color_labels_np, :]
    cv2.imwrite(save_path + '_color.png', color_labels_image)

    # get segmentation
    segmentation_np = cv2.imread(segmentation_example_path) # 0 - background, 1 - cloth, 2 - body
    cloth_mask = 1 * (segmentation_np == 1)
    segmentation_image = 255 * cloth_mask
    cv2.imwrite(save_path + '_segmentation.png', segmentation_image)
    cv2.imwrite(save_path + '_segmented_color.png', cloth_mask * color_labels_image)
    cv2.imwrite(save_path + '_segmented_type.png', cloth_mask * type_image)

    # get colored pixels
    colored_mask = (1 * (color_labels_np != 0))[:, :, None]
    black_colored_mask = (1 * (color_labels_np == 0))[:, :, None]

    # get color homogeneous pixels = surrounded by same color
    kernel = np.ones((3, 3), np.uint8) # defines the pixel neighborhood
    min_filter = cv2.erode(color_labels_np, kernel)
    max_filter = cv2.dilate(color_labels_np, kernel)
    homogeoneous_color_mask = 1 * (min_filter == max_filter)
    homogeoneous_color_image = 255 * homogeoneous_color_mask[:, :, None]
    cv2.imwrite(save_path + '_segmented_homogeous_color.png', cloth_mask * homogeoneous_color_image)

    cv2.imwrite(save_path + '_segmented_homogeous_color_colored.png', cloth_mask * colored_mask * homogeoneous_color_image)
    cv2.imwrite(save_path + '_segmented_homogeous_color_black_colored.png', cloth_mask * black_colored_mask * homogeoneous_color_image)

    # get corners
    corners_mask = 1 * (type_labels[:, :, None] == 1) * cloth_mask
    corners_image = (255 * corners_mask).astype(np.uint8)
    cv2.imwrite(save_path + '_segmented_corners.png', corners_image)

    # filter corners size
    output = cv2.connectedComponentsWithStats(
        corners_image[:, :, 0], 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output # igonre the background cc
    cc_size = stats[:, 4]
    median_size = np.median(cc_size[1:])
    mean_size = np.mean(cc_size[1:])
    std_size = np.std(cc_size[1:])
    # meant to remove big corner clusters
    size_filter_mask =np.logical_or(cc_size > corners_size_thresh_relative_median * median_size, cc_size > mean_size + 2 * std_size)
    size_filteres_image = np.logical_not(np.isin(labels, np.where(size_filter_mask)[0]))
    size_filteres_image = 255 * size_filteres_image[:, :, None]
    cv2.imwrite(save_path + '_segmented_corners_size_filteres.png', size_filteres_image)

    corners = centroids[np.logical_not(size_filter_mask), :].astype(np.uint8)

    # get centers
    centers_mask = 1 * (type_labels[:, :, None] == 2) * cloth_mask
    centers_image = (255 * centers_mask).astype(np.uint8)
    cv2.imwrite(save_path + '_segmented_centers_.png', centers_image)

    # filter centers size
    output = cv2.connectedComponentsWithStats(
        centers_image[:, :, 0], 8, cv2.CV_32S)
    (numLabels, center_labels, stats, centroids) = output # igonre the background cc
    cc_size = stats[:, 4]
    median_size = np.median(cc_size[1:])
    mean_size = np.mean(cc_size[1:])
    std_size = np.std(cc_size[1:])
    # meant to remove big corner clusters
    size_filter_mask = np.logical_or(cc_size > centers_size_thresh_relative_median * median_size, cc_size > mean_size + 2 * std_size)
    size_filteres_image_centers = np.logical_not(np.isin(center_labels, np.where(size_filter_mask)[0]))
    size_filteres_image_centers = size_filteres_image_centers[:, :, None]
    cv2.imwrite(save_path + '_segmented_centers_size_filteres.png', 255 * size_filteres_image_centers)

    centers = centroids[np.logical_not(size_filter_mask), :].astype(np.uint8)

    
    print("Hi")