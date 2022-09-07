import os
import torch
from external.KinematicAlignment.utils.tensorIO import ReadTensorFromBinaryFile, WriteTensorToBinaryFile
import matplotlib.pyplot as plt
import numpy as np
import cv2
#TODO: plot pixel location

if __name__ == '__main__':
    uv_size = 1024
    save_path = '/mnt/home/oshrihalimi/cloth_completion/data_processing/delta_pixel_location_uv_detection_minus_zero/'
    driving_cameras_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/drivingCamsSelected.txt'

    os.makedirs(save_path, exist_ok=True)

    with open(driving_cameras_path) as f:
        lines = f.readlines()
        camera_ids = np.array([line.strip() for line in lines])

    num_cameras = len(camera_ids)
    for frame in range(488, 4451):
        delta_tensor = torch.zeros((num_cameras, uv_size, uv_size, 2))

        for cam_id, camera in enumerate(camera_ids):
            pixel_location_detections_path = f'/mnt/home/oshrihalimi/capture/small_pattern_hash_detection_results_uv_domain_cleaned/{camera}/{frame}.bt'
            # pixel_location_LBS_posed_path = f'/mnt/home/oshrihalimi/cloth_completion/data_processing/posedLBS_pixel_location_uv/{camera}/{frame:06d}.bt'
            #pixel_location_LBS_posed_path = f'/mnt/home/oshrihalimi/cloth_completion/data_processing/posed_bimonocular_kinematic_model_pixel_location_uv/{camera}/{frame:06d}.pt'
            # pixel_location_LBS_posed_path = f'/mnt/home/oshrihalimi/cloth_completion/data_processing/posed_bimonocular_kinematic_model_sequential_alignment_accel_0_lowRes_pixel_location_uv/{camera}/{frame:06d}.pt'


            pixel_location_detections = ReadTensorFromBinaryFile(pixel_location_detections_path)
            # detections_mask = 1 - 1 * torch.any(torch.isinf(pixel_location_detections), dim=-1)
            # inf_idx = torch.where(torch.isinf(pixel_location_detections))
            # pixel_location_detections[inf_idx[0], inf_idx[1], inf_idx[2]] = 0

            # pixel_location_LBS_posed = torch.load(pixel_location_LBS_posed_path)
            # inf_idx = torch.where(torch.isinf(pixel_location_LBS_posed))
            # pixel_location_LBS_posed[inf_idx[0], inf_idx[1], inf_idx[2]] = 0

            # delta_signal = detections_mask[:, :, None] * (pixel_location_detections - pixel_location_LBS_posed[:, :, :2])
            # delta_signal = detections_mask[:, :, None] * pixel_location_detections
            delta_tensor[cam_id, ...] = pixel_location_detections #delta_signal

        tensor_filename = os.path.join(save_path, f'{frame}.pt')
        torch.save(delta_tensor, tensor_filename)
        print(tensor_filename)


