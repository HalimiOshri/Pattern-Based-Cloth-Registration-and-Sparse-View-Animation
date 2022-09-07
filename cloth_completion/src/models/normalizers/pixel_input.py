import torch
from models.normalizers.base import Normalizer
from src.rendering.render import sample_uv, values_to_uv
from src.dataset.augmentation import KinemaitcModelCameraProjector
from models.aux.ae_geometry import GeometryBase
import numpy as np
from src.pyutils.geomutils import vert_normals

class Deformer(torch.nn.Module):
    def __init__(self, deformer_conf):
        super(Deformer, self).__init__()
        eigvecs = torch.load(deformer_conf.LBO_path)
        self.register_buffer('eigvecs', torch.Tensor(eigvecs[:, :deformer_conf.LBO_num]))
        self.LBO_num = deformer_conf.LBO_num
        self.scale_std = deformer_conf.LBO_scale
        self.cutoff_eig = deformer_conf.LBO_cutoff
        self.register_buffer('unscaled_filter', torch.Tensor(np.exp(-np.array([x for x in range(self.LBO_num)])/self.cutoff_eig)))
        pass

    def forward(self, posed_kinematic_verts, cloth_faces):
        if self.training:
            return self.deform(posed_kinematic_verts, cloth_faces)
        else:
            return posed_kinematic_verts

    def deform(self, vertices, faces):
        coeff = torch.randint(2, size=(self.LBO_num,)).to(device=vertices.device)
        scale = self.scale_std * np.random.randn()
        filter = scale * self.unscaled_filter

        n = vert_normals(vertices.view(-1, vertices.shape[2], vertices.shape[3]),
                         torch.tensor(faces[None, :, :], dtype=torch.int64, device=vertices.device))

        deformation = n.view(vertices.shape) * torch.sum(filter[None, :] * coeff[None, :] * self.eigvecs, dim=1)[:, None]
        vertices = vertices + deformation
        return vertices

class PixelInputNormalizer(Normalizer):
    def normalize(self, unnormalized):
        per_camera_delta_pixel_uv = unnormalized["per_camera_delta_pixel_uv"]
        per_camera_delta_pixel_uv = per_camera_delta_pixel_uv.permute(0, 1, 4, 2, 3)
        s = per_camera_delta_pixel_uv.shape
        per_camera_delta_pixel_uv = torch.reshape(per_camera_delta_pixel_uv, (s[0], -1, s[3], s[4]))
        return {"normalized_uv": per_camera_delta_pixel_uv}

    def unnormalize(self, normalized_output):
        delta_3d_uv = normalized_output["verts_normalized_uv"]
        posed_LBS_verts = normalized_output["posed_LBS_verts"]

        # to make the uv tensor compatible with the uv mapping
        delta_3d_uv = torch.flip(delta_3d_uv, dims=(2,))

        delta_3d_verts = sample_uv(delta_3d_uv, self.geometry_base.uv_coords, self.geometry_base.uv_mapping)
        verts = posed_LBS_verts + delta_3d_verts
        return {"verts": verts, "tex": None}

class InhousePixelProjectionPixelPixelInputNormalizer(PixelInputNormalizer):
    def normalize(self, unnormalized):
        per_camera_detection_pixel_uv = unnormalized["per_camera_detection_pixel_uv"]
        posed_kinematic_verts = unnormalized["posed_kinematic_verts"]
        cloth_faces = unnormalized["cloth_faces"]

        batch_size = posed_kinematic_verts.shape[0]
        n_seq = posed_kinematic_verts.shape[1]
        n_verts = posed_kinematic_verts.shape[2]

        posed_kinematic_verts = self.deformer(posed_kinematic_verts, cloth_faces)

        # TODO: Fix bug in axis order for n_seq > 1
        mesh_list_to_project = torch.split(posed_kinematic_verts.view(-1, n_verts, 3), split_size_or_sections=1, dim=0)
        mesh_list_to_project = [mesh[0] for mesh in mesh_list_to_project] # num meshes = batch size x sequence size
        kinematic_pixel_location_uv = self.projector.project_kinematic_model(mesh_list_to_project)

        s = kinematic_pixel_location_uv.shape # batch_size * n_seq x n_cam x U x V x 2
        kinematic_pixel_location_uv = kinematic_pixel_location_uv.view(batch_size, n_seq, s[1], s[2], s[3], s[4])
        kinematic_pixel_location_uv = kinematic_pixel_location_uv.view(batch_size, n_seq * s[1], s[2], s[3], s[4])

        per_camera_detection_pixel_uv = per_camera_detection_pixel_uv.view(batch_size, n_seq * s[1], s[2], s[3], s[4])
        # get rid of inf values
        detections_mask = 1 - 1 * torch.any(torch.isinf(per_camera_detection_pixel_uv), dim=-1)
        inf_idx = torch.where(torch.isinf(per_camera_detection_pixel_uv))
        per_camera_detection_pixel_uv[inf_idx[0], inf_idx[1], inf_idx[2], inf_idx[3], :] = 0

        inf_idx = torch.where(torch.isinf(kinematic_pixel_location_uv))
        kinematic_pixel_location_uv[inf_idx[0], inf_idx[1], inf_idx[2], inf_idx[3], :] = 0

        per_camera_delta_pixel_uv = detections_mask[:, :, :, :, None] * (per_camera_detection_pixel_uv - kinematic_pixel_location_uv)

        # stack the sequence axis in feature axis
        per_camera_delta_pixel_uv = per_camera_delta_pixel_uv.permute(0, 1, 4, 2, 3)
        s = per_camera_delta_pixel_uv.shape
        per_camera_delta_pixel_uv = torch.reshape(per_camera_delta_pixel_uv, (s[0], -1, s[3], s[4]))

        return {"normalized_uv": per_camera_delta_pixel_uv.detach(), "deformed_kinematic_mesh": posed_kinematic_verts}

    def parse(self, dataset, kinematic_model_conf):
        self.projector = KinemaitcModelCameraProjector(camera_image_width=dataset.image_width,
                                                  camera_image_height=dataset.image_height,
                                                  template_obj_path=dataset.template_obj_path,
                                                  template_ply_path=dataset.template_ply_path,
                                                  barycentric_path=dataset.barycentric_path,
                                                  tIndex_path=dataset.tIndex_path,
                                                  path_krt=dataset.krt_path,
                                                  path_camera_indices=dataset.path_camera_indices,
                                                  projected_cam_indices=dataset.driving_camera_ids,
                                                  texture_path=dataset.dummy_texture_path)
        self.deformer = Deformer(dataset.deformer)
        super(InhousePixelProjectionPixelPixelInputNormalizer, self).parse(dataset=dataset, kinematic_model_conf=kinematic_model_conf)