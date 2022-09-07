import os
import ffmpeg
import torch
import numpy as np
import cv2
from external.KinematicAlignment.utils.tensorIO import ReadTensorFromBinaryFile, WriteTensorToBinaryFile
import subprocess

def generate_video(img, save_path):
    for i in range(img.shape[0]):
        if i == 0:
            video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (img.shape[1], img.shape[2]))

        frame = 255 * np.all(img[i] != 0, axis=2)
        frame = np.pad(frame[:, :, None], ((0, 0), (0, 0), (0, 2)))
        frame = frame.astype(np.uint8)
        video.write(frame)

    video.release()

def generate_video_subset(img, save_path):
    for i in range(img.shape[0]):
        if i == 0:
            video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (img.shape[2], img.shape[3]))

        frame = np.sum(50 * np.all(img[i] != 0, axis=3), axis=0)
        frame = np.stack((frame,) * 3, axis=2)
        frame = frame.astype(np.uint8)
        video.write(frame)

    video.release()

if __name__ == '__main__':
    tensor_path = '/mnt/home/oshrihalimi/cloth_completion/data_processing/delta_pixel_location_uv_detection_minus_LBS_posed_cleaned/'
    driving_cameras_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/drivingCamsSelected.txt'

    with open(driving_cameras_path) as f:
        lines = f.readlines()
        camera_ids = np.array([line.strip() for line in lines])

    tensors = []
    for frame in range(488, 4451, 1):
        filename = os.path.join(tensor_path, f'{frame}.bt')
        tensor = ReadTensorFromBinaryFile(filename)
        tensors.append(tensor)

    frame_camera_uv_delta_pixel = torch.stack(tensors)

    # for cam_ind in range(len(camera_ids)):
    #     camera_frames = frame_camera_uv_delta_pixel[:, cam_ind, :, :].numpy()
    #     generate_video(camera_frames, os.path.join(tensor_path, f'camera{camera_ids[cam_ind]}.mp4'))

    # i1 = np.where(camera_ids == '400889')[0].item() #*
    # i2 = np.where(camera_ids == '400926')[0].item() #*
    # # i3 = np.where(camera_ids == '400895')[0].item()
    # # i4 = np.where(camera_ids == '400883')[0].item()

    # i1 = np.where(camera_ids == '401538')[0].item()
    # i2 = np.where(camera_ids == '400895')[0].item()
    # i3 = np.where(camera_ids == '400929')[0].item()

    i1 = np.where(camera_ids == '400889')[0].item() #*
    frames_cam_subset = frame_camera_uv_delta_pixel[:, (i1, ), :, :, :].numpy()
    generate_video_subset(frames_cam_subset, os.path.join(tensor_path, f'camera_{i1}.mp4'))

