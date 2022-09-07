from external.KinematicAlignment.utils.tensorIO import ReadTensorFromBinaryFile, WriteTensorToBinaryFile
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import cv2

max_pix = np.array([2668, 4096])
if __name__ == '__main__':
    for frame in range(0, 4451):
        for camera in [400883, 400889, 400895, 400926, 400929, 401538]:
            triangulation_pattern_domain_path = f'/mnt/home/oshrihalimi/capture/registrations_pattern_domain_small/post_triangulation_cleaning_with_final_cloth_mask/pattern_domain_signal_camera_projection/{camera}/{frame}.bt'
            detection_pattern_domain_path = f'/mnt/home/oshrihalimi/capture/small_pattern_hash_detection_results_pattern_domain/{camera}/{frame}.bt'
            save_path = f'/mnt/home/oshrihalimi/capture/small_pattern_hash_detection_results_pattern_domain_cleaned/{camera}/'

            os.makedirs(save_path, exist_ok=True)
            triangulation_pattern_domain = ReadTensorFromBinaryFile(triangulation_pattern_domain_path)
            detection_pattern_domain = ReadTensorFromBinaryFile(detection_pattern_domain_path)
            detection_pattern_domain = torch.flip(detection_pattern_domain, dims=[2])

            detection_mask = 1 - 1 * torch.any(detection_pattern_domain == -np.inf, dim=-1)
            triangulation_mask = 1 - 1 * torch.any(triangulation_pattern_domain == -np.inf, dim=-1)

            detection_pattern_domain[detection_pattern_domain == -np.inf] = 0
            triangulation_pattern_domain[triangulation_pattern_domain == -np.inf] = 0

            delta = triangulation_pattern_domain - detection_pattern_domain
            small_delta_mask = 1 * torch.sqrt(torch.sum(delta ** 2, dim=-1)) < 50

            detection_pattern_domain_image = detection_pattern_domain.numpy() * 255 / max_pix
            cv2.imwrite(os.path.join(save_path, f'{frame}_original.png'), detection_pattern_domain_image[:, :, 0].astype(np.uint8))

            # filter
            valid_mask = small_delta_mask[:, :, None] * triangulation_mask[:, :, None] * detection_mask[:, :, None]
            detection_pattern_domain_masked_image = valid_mask.numpy() * detection_pattern_domain.numpy() * 255 / max_pix
            cv2.imwrite(os.path.join(save_path, f'{frame}_cleaned.png'), detection_pattern_domain_masked_image[:, :, 0].astype(np.uint8))

            invalid_mask = 1 - valid_mask
            invalid_inds = torch.where(invalid_mask)
            detection_pattern_domain[invalid_inds[0], invalid_inds[1], :] = -np.inf
            WriteTensorToBinaryFile(detection_pattern_domain, os.path.join(save_path, f'{frame}.bt'))
            print(frame, camera)
