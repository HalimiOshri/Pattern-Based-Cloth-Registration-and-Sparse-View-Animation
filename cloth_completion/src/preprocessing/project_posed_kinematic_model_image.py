from pyutils.io import load_obj
from drtk.renderlayer import RenderLayer
import trimesh
import torch
from pyutils.cameraTools import loadCameraSetInfo
import cv2
import os
from preprocessing.pattern_mesh_uv_domain_transition import PatternMeshUVConvertor
import matplotlib.pyplot as plt
import time
from external.KinematicAlignment.utils.tensorIO import WriteTensorToBinaryFile, ReadTensorFromBinaryFile
import numpy as np

if __name__ == '__main__':
    debug_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/debug/'
    #save_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/posed_kinematic_model_pixel_location_uv/'
    #save_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/posed_generative_kinematic_model_pixel_location_uv/'
    #save_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/posed_bimonocular_kinematic_model_pixel_location_uv/'
    save_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/posed_bimonocular_kinematic_model_sequential_alignment_accel_0_lowRes_pixel_location_uv/'

    camera_image_width = 2668
    camera_image_height = 4096
    rendering_scale = 1.0
    # posedLBS_path = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/ik_momentum/refitting_non_sequential/Meshes-03/'
    # posedLBS_path = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/kinematic_alignment/sequential_alignment_accel_0/{frame:06d}/optimum_skinned.ply'
    #posedLBS_path = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/binocular_registration/projective_fitting_non_sequential/Meshes-04/skinned-{frame:06d}-400889-400926.ply'
    posedLBS_path = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/kinematic_alignment/sequential_alignment_accel_0_lowRes/{frame:06d}/new_skinned.ply'
    template_obj_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured.obj'
    template_ply_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured.ply'
    barycentric_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/uvToMesh_1024/uvGrid_to_mesh_bc.bt'
    tIndex_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/uvToMesh_1024/uvGrid_to_mesh_tIndex.bt'
    path_krt = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/raw_sequence_dir/KRTnodoor'
    path_camera_indices = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/drivingCamsSelected.txt'
    texture_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/texture_cats.jpeg'
    [cam_focal, cam_princpt, cam_intrinsic, camera_rotation, camera_translation, camera_position, w, h, cam_indices] = \
        loadCameraSetInfo(path_krt, path_camera_indices, camera_image_width, camera_image_height, rendering_scale, to_cuda=True)

    v, vt, vi, vti = load_obj(template_obj_path)
    v = torch.Tensor(v).to(device='cuda')
    vt = torch.Tensor(vt).to(device='cuda')
    vi = torch.Tensor(vi).to(device='cuda')
    vti = torch.Tensor(vti).to(device='cuda')

    rl = RenderLayer(h, w, vt, vi, vti, boundary_aware=False, flip_uvs=True)


    convertor = PatternMeshUVConvertor(mesh_path=template_ply_path,
                                       barycentric_path=barycentric_path, tIndex_path=tIndex_path)
    texture = torch.tensor(cv2.imread(texture_path), dtype=torch.float32).permute(2, 0, 1)[None, ...].to(device='cuda')

    os.makedirs(save_dir, exist_ok=True)
    for cam_id in cam_indices:
        os.makedirs(os.path.join(save_dir, str(cam_id)), exist_ok=True)

    for frame in range(488, 4451):
        posedLBS_filename = posedLBS_path.format(frame=frame)
        mesh = trimesh.load_mesh(posedLBS_filename, process=False, validate=False)
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)[None, ...].to(device='cuda')


        for i in range(len(cam_indices)):
            output = rl(vertices, texture,
                        camera_position[i][None, ...], camera_rotation[i][None, ...], cam_focal[i][None, ...], cam_princpt[i][None, ...],
                        ksize=None, output_filters=["v_pix"])
                        # output_filters=["index_img", "render", "mask", "bary_img", "v_pix"])  # originally ksize=3

            pixel_location = output["v_pix"][0].detach().cpu()
            pixel_location_uv = convertor.mesh2uv(pixel_location)
            save_filename = os.path.join(save_dir, str(cam_indices[i]), f'{frame:06d}.pt')
            print(pixel_location_uv.min())
            print(pixel_location_uv.max())
            print(torch.sum(torch.isnan(pixel_location_uv)))
            print(pixel_location_uv.device)
            print(pixel_location_uv.dtype)
            #WriteTensorToBinaryFile(pixel_location_uv, save_filename)
            #tmp = ReadTensorFromBinaryFile(save_filename)
            torch.save(pixel_location_uv, save_filename)
            tmp = torch.load(save_filename)

            # mask = output["mask"][:, None, :, :]  # add a singleton dimension in the place of RGB dimension --> B x 1 x H x W
            # rendered = output["render"]
            # cv2.imwrite(os.path.join(debug_dir, 'rendered_LBSposed.png'), rendered[0].permute(1, 2, 0).cpu().numpy())

            # pixel_location_uv_image = pixel_location_uv.numpy()
            # pixel_location_uv_image[pixel_location_uv_image ==-np.inf] = 0
            # max_value = pixel_location_uv_image.max()
            # min_value = pixel_location_uv_image.min()
            # pixel_location_uv_image = 255 * (pixel_location_uv_image - min_value) / (max_value - min_value)
            # pixel_location_uv_image = pixel_location_uv_image.astype(np.uint8)
            # cv2.imwrite(os.path.join(debug_dir, f'generative_kinematic_model_posed_pixel_location_uv_{frame}.png'), pixel_location_uv_image)

            print(save_filename)
