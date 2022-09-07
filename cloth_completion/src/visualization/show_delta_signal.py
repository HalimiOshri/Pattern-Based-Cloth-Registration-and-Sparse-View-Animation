from external.KinematicAlignment.utils.tensorIO import ReadTensorFromBinaryFile, WriteTensorToBinaryFile
import cv2
import torch
import numpy as np

path_delta_signal = '/mnt/home/oshrihalimi/cloth_completion/data_processing/gt_geometry_pixel_location_uv/400872/skinned-004041.bt'
save_path = '/mnt/home/oshrihalimi/cloth_completion/data_processing/gt_geometry_pixel_location_uv/400872/skinned-004041.png'
cam_id = 1

if __name__ == '__main__':
    delta_signal = ReadTensorFromBinaryFile(path_delta_signal)
    #delta_signal = torch.load(path_delta_signal)[cam_id]
    delta_signal[delta_signal==-np.inf] = 0
    min_delta = torch.min(delta_signal.view(-1, 3), dim=0)[0]
    max_delta = torch.max(delta_signal.view(-1, 3), dim=0)[0]

    image = 255 * (delta_signal - min_delta[None, None, :]) / (max_delta[None, None, :] - min_delta[None, None, :])
    image = image.numpy().astype(np.uint8)
    #image = np.pad(image, ((0, 0), (0, 0), (0, 1)))
    save_filename = save_path.split('.')[0] + f'_{cam_id}.png'
    cv2.imwrite(save_filename, image)