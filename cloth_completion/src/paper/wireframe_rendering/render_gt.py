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
import numpy as np
from external.KinematicAlignment.utils.tensorIO import WriteTensorToBinaryFile

if __name__ == '__main__':
    debug_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/debug/'
    save_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/gt_geometry_pixel_location_pattern_domain/'

    pattern_size = np.array([300, 900])
    camera_image_width = 2668
    camera_image_height = 4096
    rendering_scale = 1.0
    gt_geometry_path = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/surface_deformable/incremental_registration/Meshes/'
    template_obj_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured.obj'
    template_ply_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured.ply'
    barycentric_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/uvToMesh_1024/uvGrid_to_mesh_bc.bt'
    tIndex_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/uvToMesh_1024/uvGrid_to_mesh_tIndex.bt'
    grid_to_mesh_index_map_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/gridToMeshMap.txt'
    path_krt = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/raw_sequence_dir/KRTnodoor'
    path_camera_indices = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/drivingCams.txt'
    texture_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/board_rendering_normalized_10k.png'
    [cam_focal, cam_princpt, cam_intrinsic, camera_rotation, camera_translation, camera_position, w, h, cam_indices] = \
        loadCameraSetInfo(path_krt, path_camera_indices, camera_image_width, camera_image_height, rendering_scale, to_cuda=True)

    v, vt, vi, vti = load_obj(template_obj_path)
    v = torch.Tensor(v).to(device='cuda')
    vt = torch.Tensor(vt).to(device='cuda')
    vi = torch.Tensor(vi).to(device='cuda')
    vti = torch.Tensor(vti).to(device='cuda')

    rl = RenderLayer(h, w, vt, vi, vti, boundary_aware=False, flip_uvs=True, backface_culling=True)


    convertor = PatternMeshUVConvertor(pattern_size=pattern_size, grid_to_mesh_index_map_path=grid_to_mesh_index_map_path, mesh_path=template_ply_path,
                                       barycentric_path=barycentric_path, tIndex_path=tIndex_path)
    texture = cv2.imread(texture_path)
    texture = cv2.resize(texture, (100000, 100000), interpolation=cv2.INTER_AREA)
    texture = torch.tensor(cv2.imread(texture_path), dtype=torch.float32).permute(2, 0, 1)
    texture_wire = torch.where(torch.all(texture == 0, dim=0))
    texture = 0 * texture
    texture[:, texture_wire[0], texture_wire[1]] = 255

    os.makedirs(save_dir, exist_ok=True)
    for cam_id in cam_indices:
        os.makedirs(os.path.join(save_dir, str(cam_id)), exist_ok=True)

    os.listdir(gt_geometry_path)
    for frame in os.listdir(gt_geometry_path):
        posedLBS_filename = os.path.join(gt_geometry_path, frame)
        mesh = trimesh.load_mesh(posedLBS_filename, process=False, validate=False)
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)


        for i in range(len(cam_indices)):
            output = rl(vertices[None, ...].to(device='cuda'), texture[None, ...].to(device='cuda'),
                        camera_position[i][None, ...], camera_rotation[i][None, ...], cam_focal[i][None, ...], cam_princpt[i][None, ...],
                        ksize=None, output_filters=["render"])
                        # output_filters=["index_img", "render", "mask", "bary_img", "v_pix"])  # originally ksize=3



            rendered = output["render"][0].cpu().numpy()

            save_filename = os.path.join(save_dir, str(cam_indices[i]), frame.split('.')[0] + '_rendered_wire.png')
            cv2.imwrite(save_filename, rendered.transpose(1, 2, 0))

            print(save_filename)
