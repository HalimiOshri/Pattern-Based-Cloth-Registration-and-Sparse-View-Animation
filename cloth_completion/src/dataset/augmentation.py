from src.pyutils.io import load_ply
from pyutils.io import load_obj
from drtk.renderlayer import RenderLayer
import trimesh
import torch
from pyutils.cameraTools import loadCameraSetInfo
import cv2
import os
from preprocessing.pattern_mesh_uv_domain_transition import PatternMeshUVConvertor

class KinemaitcModelCameraProjector(torch.nn.Module):
    def __init__(self, camera_image_width, camera_image_height, template_obj_path,
                 template_ply_path, barycentric_path, tIndex_path, path_krt, path_camera_indices, projected_cam_indices, texture_path):
        # path_krt & path_camera_indices, contain the absolute camera indices and KRT in the same order, exactly corresponding values
        # projected_cam_indices holds relative indices to use for projection
        super(KinemaitcModelCameraProjector, self).__init__()

        rendering_scale = 1.0
        [cam_focal, cam_princpt, cam_intrinsic, camera_rotation, camera_translation, camera_position, w, h, cam_indices] =\
            loadCameraSetInfo(path_krt, path_camera_indices, camera_image_width, camera_image_height, rendering_scale, to_cuda=False)

        self.cam_indices = cam_indices
        self.projected_cam_indices = projected_cam_indices
        self.register_buffer('cam_focal', torch.Tensor(cam_focal))
        self.register_buffer('cam_princpt', torch.Tensor(cam_princpt))
        self.register_buffer('cam_intrinsic', torch.Tensor(cam_intrinsic))
        self.register_buffer('camera_rotation', torch.Tensor(camera_rotation))
        self.register_buffer('camera_translation', torch.Tensor(camera_translation))
        self.register_buffer('camera_position', torch.Tensor(camera_position))


        v, vt, vi, vti = load_obj(template_obj_path)
        self.register_buffer('v', torch.Tensor(v))
        self.register_buffer('vt', torch.Tensor(vt))
        self.register_buffer('vi', torch.Tensor(vi))
        self.register_buffer('vti', torch.Tensor(vti))

        self.rl = RenderLayer(h, w, self.vt, self.vi, self.vti, boundary_aware=False, flip_uvs=True)

        self.convertor = PatternMeshUVConvertor(mesh_path=template_ply_path,
                                           barycentric_path=barycentric_path, tIndex_path=tIndex_path)
        self.uv_size = self.convertor.uv_size
        self.register_buffer('texture', torch.tensor(cv2.imread(texture_path), dtype=torch.float32).permute(2, 0, 1)[None, ...])

    def project_kinematic_model(self, posed_kinematic_meshes):
        # posed_kinematic_meshes = [load_ply(path) for path in paths_posed_kinematic_model]
        num_cameras = len(self.projected_cam_indices)
        num_meshes = len(posed_kinematic_meshes)
        pixel_tensor = torch.zeros((num_meshes, num_cameras, self.uv_size[0], self.uv_size[1], 2), device=self.cam_focal.device)
        for m, mesh in enumerate(posed_kinematic_meshes):
            for i, cam_ind in enumerate(self.projected_cam_indices):
                vertices = torch.tensor(mesh, dtype=torch.float32)[None, ...].to(device=self.cam_focal.device)
                output = self.rl(vertices, self.texture,
                            self.camera_position[cam_ind][None, ...],
                            self.camera_rotation[cam_ind][None, ...],
                            self.cam_focal[cam_ind][None, ...],
                            self.cam_princpt[cam_ind][None, ...],
                            ksize=None, output_filters=["v_pix"])

                pixel_location = output["v_pix"][0].clone().detach()
                pixel_location_uv = self.convertor.mesh2uv(pixel_location)

                inf_idx = torch.where(torch.isinf(pixel_location_uv))
                pixel_location_uv[inf_idx[0], inf_idx[1], inf_idx[2]] = 0

                pixel_tensor[m, i, :, :, :] = pixel_location_uv[:, :, :2]
        return pixel_tensor


if __name__ == '__main__':
    camera_image_width = 2668
    camera_image_height = 4096
    template_obj_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured.obj'
    template_ply_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured.ply'
    barycentric_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/uvToMesh_1024/uvGrid_to_mesh_bc.bt'
    tIndex_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/uvToMesh_1024/uvGrid_to_mesh_tIndex.bt'
    path_krt = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/raw_sequence_dir/KRTnodoor'
    path_camera_indices = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/drivingCamsSelected.txt'
    texture_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/texture_cats.jpeg'

    augmentor = KinemaitcModelCameraProjector(camera_image_width=camera_image_width,
                                        camera_image_height=camera_image_height,
                                        template_obj_path=template_obj_path,
                                        template_ply_path=template_ply_path,
                                        barycentric_path=barycentric_path,
                                        tIndex_path=tIndex_path,
                                        path_krt=path_krt,
                                        path_camera_indices=path_camera_indices,
                                        texture_path=texture_path)
    augmentor = augmentor.to('cuda:0')
    paths = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/kinematic_alignment/sequential_alignment_accel_0_lowRes/{frame:06d}/new_skinned.ply'
    frames = [x for x in range(3000, 3010)]
    paths = [paths.format(frame=frame) for frame in frames]
    augmentor.project_kinematic_model(paths)


