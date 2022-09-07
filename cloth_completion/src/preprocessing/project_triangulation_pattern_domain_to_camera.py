import os
from external.KinematicAlignment.utils.tensorIO import ReadTensorFromBinaryFile, WriteTensorToBinaryFile
from pyutils.cameraTools import loadCameraSetInfo
from drtk.renderlayer import project_points
import numpy as np
import torch

save_path = '/mnt/home/oshrihalimi/capture/registrations_pattern_domain_small/post_triangulation_cleaning_with_final_cloth_mask/pattern_domain_signal_camera_projection/'
pattern_size = [300, 900]

path_krt = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/raw_sequence_dir/KRTnodoor'
path_camera_indices = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/drivingCamsSelected.txt'
camera_image_width = 2668
camera_image_height = 4096
rendering_scale = 1.0

if __name__ == '__main__':
    for frame in range(0, 4451):
        triangulation_pattern_domain_path = f'/mnt/home/oshrihalimi/capture/registrations_pattern_domain_small/post_triangulation_cleaning_with_final_cloth_mask/pattern_domain_signal/{frame}.bt'
        triangulation_pattern_domain = ReadTensorFromBinaryFile(triangulation_pattern_domain_path)
        off_triangulation_mask = torch.any(triangulation_pattern_domain == -np.inf, dim=-1)
        off_triangulation_inds = torch.where(off_triangulation_mask)
        [cam_focal, cam_princpt, cam_intrinsic, camera_rotation, camera_translation, camera_position, w, h, cam_indices] = \
            loadCameraSetInfo(path_krt, path_camera_indices, camera_image_width, camera_image_height, rendering_scale, to_cuda=True)

        for cam in cam_indices:
            os.makedirs(os.path.join(save_path, str(cam)), exist_ok=True)

        v = triangulation_pattern_domain.view(-1, 3).to(device='cuda')
        v[v==-np.inf] = 0
        for i in range(len(cam_indices)):
            v_pix, _ = project_points(v[None, ...], camera_position[i][None, ...], camera_rotation[i][None, ...], cam_focal[i][None, ...], cam_princpt[i][None, ...])
            v_pix = v_pix.view(pattern_size[0], pattern_size[1], 3)
            v_pix = v_pix[:, :, :2]
            v_pix[off_triangulation_inds[0], off_triangulation_inds[1], :] = -np.inf
            filename = os.path.join(save_path, str(cam_indices[i]), f'{frame}.bt')
            WriteTensorToBinaryFile(v_pix.cpu(), filename)
            print(filename)
        print("Hi")

