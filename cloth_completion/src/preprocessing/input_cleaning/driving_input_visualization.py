import os
import numpy as np
import cv2
from external.KinematicAlignment.utils.tensorIO import ReadTensorFromBinaryFile, WriteTensorToBinaryFile

camera = 400889
uv_size = 1024
input_path = f'/mnt/home/oshrihalimi/capture/small_pattern_hash_detection_results_uv_domain/{camera}/'
delta_input_path = f'http://hewen.sb.facebook.net:8082/visualize/mnt/home/oshrihalimi/cloth_completion/data_processing/delta_pixel_location_uv_detection_minus_LBS_posed/
save_path = os.path.join(input_path, 'video.mp4')
if __name__ == '__main__':
    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (uv_size, uv_size))
    #low = -100
    #high = 100
    low = np.array([0, 0])
    high = np.array([4096, 2668])

    for frame in range(0, 4551):
        filename = os.path.join(input_path, f'{frame}.bt')
        delta_filename = os.path.join(delta_input_path, f'{frame}.bt')
        image = ReadTensorFromBinaryFile(filename).numpy()
        # image[image > high] = high
        # image[image < low] = low
        # image = 255 * (image - low) / (high - low)

        image[image == -np.inf] = 0
        image = 255 * (image - low) / (high - low)

        image = np.pad(image, ((0, 0), (0, 0), (0, 1))).astype(np.uint8)
        video.write(image)
    video.release()