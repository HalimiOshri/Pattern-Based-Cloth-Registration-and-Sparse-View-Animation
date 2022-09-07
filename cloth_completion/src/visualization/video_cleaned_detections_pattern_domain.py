import cv2
import numpy as np
import os
import torch
from external.KinematicAlignment.utils.tensorIO import ReadTensorFromBinaryFile, WriteTensorToBinaryFile

camera = 400889
W, H = 900, 300

cleaned_path = f'/mnt/home/oshrihalimi/capture/small_pattern_hash_detection_results_pattern_domain_cleaned/{camera}/'
save_path = os.path.join(cleaned_path, 'cleaned_tensors.mp4')
max_pix = np.array([2668, 4096, 1])
if __name__ == '__main__':
    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (W, H))

    for frame in range(0, 4451):
        tensor = ReadTensorFromBinaryFile(os.path.join(cleaned_path, f'{frame}.bt'))
        tensor[tensor == -np.inf] = 0
        image = np.pad(tensor.numpy(), ((0, 0), (0, 0), (0, 1)))
        image = image * 255 / max_pix
        image = image.astype(np.uint8)
        video.write(image)

    video.release()