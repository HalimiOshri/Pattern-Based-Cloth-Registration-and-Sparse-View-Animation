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
from external.KinematicAlignment.utils.tensorIO import WriteTensorToBinaryFile
import numpy as np

if __name__ == '__main__':
    debug_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/debug/'
    # save_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/posedLBS_3d_location_uv/'
    save_dir = '/mnt/home/oshrihalimi/compare_methods_paper/geodesic_distortion/coordinates_metric_uv_space/pattern_registration'

    # posedLBS_path = '/mnt/home/oshrihalimi/cloth_completion/data_processing/posedLBS/'
    posedLBS_path = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/surface_deformable/incremental_registration/Meshes/'
    template_obj_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured.obj'
    template_ply_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured.ply'
    barycentric_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/uvToMesh_1024/uvGrid_to_mesh_bc.bt'
    tIndex_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/uvToMesh_1024/uvGrid_to_mesh_tIndex.bt'

    convertor = PatternMeshUVConvertor(mesh_path=template_ply_path,
                                       barycentric_path=barycentric_path, tIndex_path=tIndex_path)

    os.makedirs(save_dir, exist_ok=True)
    os.listdir(posedLBS_path)

    for x in range(1265, 4450):
        frame = f'skinned-{x:06d}.ply'
        try:
            posedLBS_filename = os.path.join(posedLBS_path, frame)
            mesh = trimesh.load_mesh(posedLBS_filename, process=False, validate=False)
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32)

            location_3d_uv = convertor.mesh2uv(torch.tensor(vertices))

            inf_idx = torch.where(location_3d_uv == -np.inf)
            location_3d_uv[inf_idx[0], inf_idx[1], inf_idx[2]] = 0

            save_filename = os.path.join(save_dir, frame.split('.')[0] + '.bt')
            WriteTensorToBinaryFile(location_3d_uv, save_filename)

            # plt.imshow(location_3d_uv)
            # plt.colorbar()
            # plt.savefig(os.path.join(save_dir, frame.split('.')[0] + '.png'))
            # plt.close()
            print(save_filename)
        except:
            print(f'Failed with frame: {frame}')
