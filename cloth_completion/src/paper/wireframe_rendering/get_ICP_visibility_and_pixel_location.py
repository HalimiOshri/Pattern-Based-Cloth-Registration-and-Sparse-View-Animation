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
    save_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/ICP_geometry_pixel_location_pattern_domain/'

    pattern_size = np.array([300, 900])
    camera_image_width = 2668
    camera_image_height = 4096
    rendering_scale = 1.0
    gt_geometry_path = '/mnt/home/donglaix/s--20210823--1323--0000000--pilot--patternCloth/ICPAfterSmoothing_selfAdjustLpl/'
    template_obj_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured.obj'
    template_ply_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured.ply'
    barycentric_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/uvToMesh_1024/uvGrid_to_mesh_bc.bt'
    tIndex_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/uvToMesh_1024/uvGrid_to_mesh_tIndex.bt'
    grid_to_mesh_index_map_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/gridToMeshMap.txt'
    path_krt = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/raw_sequence_dir/KRTnodoor'
    path_camera_indices = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/drivingCams.txt'
    texture_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/texture_cats.jpeg'
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
    texture = torch.tensor(cv2.imread(texture_path), dtype=torch.float32).permute(2, 0, 1)

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
                        ksize=None, output_filters=["v_pix", "index_img", "mask"])
                        # output_filters=["index_img", "render", "mask", "bary_img", "v_pix"])  # originally ksize=3



            faces_idx = output["index_img"][0].cpu()
            render_mask = output["mask"][0].cpu()
            visible_faces = torch.unique(faces_idx[(faces_idx != -1) * render_mask])
            visible_vertices = torch.unique(vi[visible_faces.long(), :].view(-1))

            vertex_invisibility_mask = torch.ones(v.shape[0])
            vertex_invisibility_mask[visible_vertices.long()] = 0
            pixel_location = output["v_pix"][0].cpu()
            pixel_location[vertex_invisibility_mask.bool(), :] = -np.inf
            pixel_location_pattern_domain = convertor.mesh2pattern(pixel_location)

            save_filename = os.path.join(save_dir, str(cam_indices[i]), frame.split('.')[0] + '_pixel_location.bt')
            WriteTensorToBinaryFile(pixel_location_pattern_domain, save_filename)


            # mask = output["mask"][:, None, :, :]  # add a singleton dimension in the place of RGB dimension --> B x 1 x H x W
            # rendered = output["render"]
            # cv2.imwrite(os.path.join(debug_dir, 'rendered_LBSposed.png'), rendered[0].permute(1, 2, 0).cpu().numpy())

            # cv2.imwrite(os.path.join(debug_dir, 'LBSposed_pixel_location_uv.png'), pixel_location_uv.numpy())
            # plt.imshow(pixel_location_uv[:, :, 0].numpy())
            # plt.show()

            print(save_filename)
