from external.KinematicAlignment.utils.tensorIO import ReadTensorFromBinaryFile
import torch
import numpy as np
import cv2
import os

camera = 400872
frame = 3752
path = f'/mnt/home/oshrihalimi/cloth_completion/data_processing/gt_geometry_pixel_location_pattern_domain/{camera}/skinned-{frame:06d}_pixel_location.bt'
debug_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/debug/'
image_path = f'/mnt/captures/studies/pilots/sociopticon/Aug18/s--20210823--1323--0000000--pilot--patternCloth/undistorted/cam{camera}/image{frame:04d}.png'
if __name__ == '__main__':
    pixel_location_gt = ReadTensorFromBinaryFile(path)
    valid_inds_pattern_domain = torch.all(pixel_location_gt != -np.inf, dim=-1)
    horizontal_edges_inds = torch.where(valid_inds_pattern_domain[:, :-1] * valid_inds_pattern_domain[:, 1:])
    vertical_edges_inds = torch.where(valid_inds_pattern_domain[:-1, :] * valid_inds_pattern_domain[1:, :])

    original_image = cv2.imread(image_path).transpose(1, 0, 2)
    image = np.zeros((2668, 4096, 3))
    color = (255, 255, 255)
    thickness = 1

    for i in range(horizontal_edges_inds[0].shape[0]):
        x = horizontal_edges_inds[0][i]
        y = horizontal_edges_inds[1][i]
        start_point = np.flip(pixel_location_gt[x, y, :2].to(dtype=torch.int).numpy())
        end_point = np.flip(pixel_location_gt[x, y + 1, :2].to(dtype=torch.int).numpy())
        image = cv2.line(image, start_point, end_point, color, thickness)

    for i in range(vertical_edges_inds[0].shape[0]):
        x = vertical_edges_inds[0][i]
        y = vertical_edges_inds[1][i]
        start_point = np.flip(pixel_location_gt[x, y, :2].to(dtype=torch.int).numpy())
        end_point = np.flip(pixel_location_gt[x + 1, y, :2].to(dtype=torch.int).numpy())
        image = cv2.line(image, start_point, end_point, color, thickness)

    image = image + original_image
    image[image > 255] = 255
    cv2.imwrite(os.path.join(debug_dir, 'pixel_location_gt.png'), np.transpose(image, (1, 0, 2)))
