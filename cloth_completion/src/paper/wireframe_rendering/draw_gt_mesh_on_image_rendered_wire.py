from external.KinematicAlignment.utils.tensorIO import ReadTensorFromBinaryFile
import torch
import numpy as np
import cv2
import os

camera = 400872
frame = 2934
path = f'/mnt/home/oshrihalimi/cloth_completion/data_processing/gt_geometry_pixel_location_pattern_domain/{camera}/skinned-{frame:06d}_rendered_wire.png'
debug_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/debug/'
image_path = f'/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/qualitative_evaluation_registration_wireframe/image2934.png'
if __name__ == '__main__':
    original_image = cv2.imread(image_path).transpose(1, 0, 2)
    wire_image = 255 * (cv2.imread(path).transpose(1, 0, 2) > 0)

    original_image[wire_image[:, :, 0] == 255, 0] = 238
    original_image[wire_image[:, :, 0] == 255, 1] = 175
    original_image[wire_image[:, :, 0] == 255, 2] = 3
    image = original_image
    cv2.imwrite(os.path.join(debug_dir, 'pixel_location_gt_rendered_wire.png'), np.transpose(image, (1, 0, 2)))

