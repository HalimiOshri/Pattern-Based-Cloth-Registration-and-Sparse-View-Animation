import os
import numpy as np
import torch
from external.KinematicAlignment.utils.tensorIO import WriteTensorToBinaryFile
import matplotlib.pyplot as plt

triangulation_path = '/mnt/home/oshrihalimi/capture/registrations_pattern_domain_small/post_triangulation_cleaning_with_final_cloth_mask/tensors_3D_300x900_cleaned/'
save_path = '/mnt/home/oshrihalimi/capture/registrations_pattern_domain_small/post_triangulation_cleaning_with_final_cloth_mask/pattern_domain_signal/'

if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)

    for frame in range(0, 4451):
        filename = os.path.join(triangulation_path, f'{frame}.pkl')
        d = np.load(filename, allow_pickle=True)

        detection = d['location']
        off_pixels = np.where(d['detection_mask'] == 0)
        detection[off_pixels[0], off_pixels[1], :] = -np.inf

        save_filename = os.path.join(save_path, f'{frame}.bt')
        WriteTensorToBinaryFile(torch.Tensor(detection), save_filename)

        plt.imshow(detection[:, :, 0])
        plt.savefig(save_filename.split('.')[0] + '.png')
        plt.close()
        print(save_filename)