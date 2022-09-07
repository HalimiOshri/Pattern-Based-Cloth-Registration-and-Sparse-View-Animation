import os
import numpy as np
import cv2
from external.KinematicAlignment.utils.tensorIO import ReadTensorFromBinaryFile, WriteTensorToBinaryFile
from registration_analyzer import RegistrationAnalyser

# filtering camera detection outliers in pattern domain based on triangulation projection

camera = 400889
frame = 4000
uv_size = 1024

input_path = f'/mnt/home/oshrihalimi/capture/small_pattern_hash_detection_results_uv_domain/{camera}/{frame}.bt'
save_path = f'/mnt/home/oshrihalimi/capture/small_pattern_hash_detection_results_uv_domain_cleaned/{camera}/{frame}.bt'

path_mask_uv = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/uvToMesh_1024/uvGrid_to_mesh_active.png'

if __name__ == '__main__':
    detections = ReadTensorFromBinaryFile(input_path).numpy()
    detections_mask = 1 - 1 * np.any(detections == -np.inf, axis=-1)
    detections[detections == -np.inf] = 0
    cloth_mask = np.any(cv2.imread(path_mask_uv) > 0, axis=-1)

    detections_mask = detections_mask[:, :, None]
    cloth_mask = cloth_mask[:, :, None]

    analyser = RegistrationAnalyser(registrations_dict = {"detection_mask": detections_mask, "cloth_mask": cloth_mask, "location": detections})
    outlier_mask, median = analyser.filter_outliers()
    print("Hi")