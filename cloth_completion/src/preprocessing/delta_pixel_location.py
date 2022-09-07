import os
import torch
from external.KinematicAlignment.utils.tensorIO import ReadTensorFromBinaryFile
import matplotlib.pyplot as plt
import numpy as np
import cv2
#TODO: plot pixel location

if __name__ == '__main__':
    debug_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/debug/'

    frame = 3500
    camera = 400889

    pixel_location_detections_path = f'/mnt/home/oshrihalimi/capture/small_pattern_hash_detection_results_uv_domain_cleaned/{camera}/{frame}.bt'
    pixel_location_LBS_posed_path = f'/mnt/home/oshrihalimi/cloth_completion/data_processing/posedLBS_pixel_location_uv/{camera}/{frame:06d}.bt'


    pixel_location_detections = ReadTensorFromBinaryFile(pixel_location_detections_path)
    # pixel_location_detections = torch.flip(pixel_location_detections, dims=[2])
    detections_mask = 1 - 1 * torch.any(torch.isinf(pixel_location_detections), dim=-1)
    inf_idx = torch.where(torch.isinf(pixel_location_detections))
    pixel_location_detections[inf_idx[0], inf_idx[1], inf_idx[2]] = 0

    pixel_location_LBS_posed = ReadTensorFromBinaryFile(pixel_location_LBS_posed_path)
    pixel_location_LBS_posed[pixel_location_LBS_posed == -np.inf] = 0

    delta_signal = detections_mask[:, :, None] * torch.abs((pixel_location_detections - pixel_location_LBS_posed[:, :, :2]))
    delta_signal = delta_signal.numpy()
    delta_image = np.sqrt(delta_signal[:, :, 0] ** 2 + delta_signal[:, :, 1] ** 2)

    plt.imshow(delta_image)
    plt.colorbar()
    plt.savefig(os.path.join(debug_dir, 'delta_pixel_lcation_uv.png'))

    pixel_set_lbs = torch.unique((pixel_location_LBS_posed * detections_mask[:, :, None]).to(dtype=torch.int).view(-1, 3), dim=0)[:, :2].numpy()
    pixel_set_detections = torch.unique((pixel_location_detections * detections_mask[:, :, None]).to(dtype=torch.int).view(-1, 2), dim=0).numpy()

    image = np.zeros((2668, 4096, 3))
    image[pixel_set_lbs[:, 0], pixel_set_lbs[:, 1], 0] = 255
    image[pixel_set_detections[:, 0], pixel_set_detections[:, 1], 2] = 255

    cv2.imwrite(os.path.join(debug_dir, 'pixel_lcation_image.png'), np.transpose(image, (1, 0, 2)))
    print("Hi")


