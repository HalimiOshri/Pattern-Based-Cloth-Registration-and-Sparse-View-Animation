import numpy as np
import os
from preprocessing.pattern_mesh_uv_domain_transition import PatternMeshUVConvertor
import matplotlib.pyplot as plt
from multiprocessing import Pool
from external.KinematicAlignment.utils.tensorIO import WriteTensorToBinaryFile
import torch

def process_camera_frame(pattern_size, frame, camera_dir, save_dir):

    image_registration_path = os.path.join(camera_dir, frame)
    image_registration_data = np.load(image_registration_path, allow_pickle=True).item()
    board_locations = np.array(
        [image_registration_data['board_location'][idx] for idx in image_registration_data['board_location'].keys()])
    image_locations = np.array(
        [image_registration_data['image_location'][idx] for idx in image_registration_data['board_location'].keys()])
    board_locations = board_locations.astype(int)

    pixel_location_pattern_domain = -np.inf * np.ones((pattern_size[0], pattern_size[1], 2))
    pixel_location_pattern_domain[board_locations[:, 0], board_locations[:, 1], :] = image_locations

    save_filename = os.path.join(save_dir, frame.split('.')[0] + '.bt')
    WriteTensorToBinaryFile(torch.Tensor(pixel_location_pattern_domain), save_filename)

    plt.imshow(pixel_location_pattern_domain[:, :, 0])
    plt.savefig(save_filename.split('.')[0] + '.png')
    plt.close()
    print(save_filename)

if __name__ == '__main__':
    pattern_size = np.array([300, 900])
    path_pixel_location = '/mnt/home/oshrihalimi/capture/small_pattern_hash_detection_results/'
    driving_cameras_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/drivingCamsSelected.txt'
    save_path = '/mnt/home/oshrihalimi/capture/small_pattern_hash_detection_results_pattern_domain/'

    with open(driving_cameras_path) as f:
        lines = f.readlines()
        camera_ids = np.array([line.strip() for line in lines])

    samples = []
    for cam_id in camera_ids:
        camera_dir = os.path.join(path_pixel_location, 'cam' + cam_id)
        save_dir = os.path.join(save_path, cam_id)
        os.makedirs(save_dir, exist_ok=True)
        frames = [f for f in os.listdir(camera_dir) if f.endswith('.npy')]
        samples += [(frame, camera_dir, save_dir) for frame in frames]

    with Pool(processes=200) as pool:
        pool.starmap(process_camera_frame, ((pattern_size, sample[0], sample[1], sample[2])
                                            for sample in samples))

    # for sample in samples:
    #     process_camera_frame(pattern_size, sample[0], sample[1], sample[2])
    #
    #
