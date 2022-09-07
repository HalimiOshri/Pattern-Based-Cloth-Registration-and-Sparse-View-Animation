import numpy as np
import matplotlib.pyplot as plt
import torch

from external.KinematicAlignment.utils.tensorIO import ReadTensorFromBinaryFile
import trimesh
import os

class PatternMeshUVConvertor(torch.nn.Module):
    def __init__(self, pattern_size=None, mesh_to_grid_index_map_path=None, grid_to_mesh_index_map_path=None,
                 mesh_path=None, barycentric_path=None, tIndex_path=None, uv_coords_path=None):
        super(PatternMeshUVConvertor, self).__init__()

        self.mesh_to_grid_index_map_path = mesh_to_grid_index_map_path
        self.grid_to_mesh_index_map_path = grid_to_mesh_index_map_path
        self.pattern_size = pattern_size
        self.mesh_path = mesh_path
        self.barycentric_path = barycentric_path
        self.tIndex_path = tIndex_path
        self.parse_files()

    def pattern2mesh(self, signal_pattern_domain):
        if self.mesh_to_grid_index_map_path:
            signal_grid_verts = self.to_grid_vertices(signal_pattern_domain)
            signal_verts = signal_grid_verts[self.map_mesh2grid, ...]
            return signal_verts
        else:
            raise Exception("Not supported!"
                            "To support this method, you must supply mesh_to_grid_index_map_path "
                            "when creating a PatternMeshUVConvertor object.")

    def mesh2pattern(self, signal_mesh_domain):
        if self.grid_to_mesh_index_map_path:
            signal_grid_verts = -np.inf * torch.ones((self.map_grid2mesh.shape[0],) + signal_mesh_domain.shape[1:])
            vertex_inds_grid = np.where(self.map_grid2mesh != -1)[0]
            vertex_inds_mesh = self.map_grid2mesh[vertex_inds_grid]
            signal_grid_verts[vertex_inds_grid, ...] = signal_mesh_domain[vertex_inds_mesh, ...]
            signal_pattern_domain = self.to_pattern_domain(signal_grid_verts)
            return signal_pattern_domain
        else:
            raise Exception("Not supported!"
                            "To support this method, you must supply grid_to_mesh_index_map_path "
                            "when creating a PatternMeshUVConvertor object.")

    def pattern2uv(self, signal_pattern_domain):
        signal_mesh_domain = self.pattern2mesh(signal_pattern_domain)
        signal_uv_domain = self.mesh2uv(torch.Tensor(signal_mesh_domain))
        return signal_uv_domain

    def mesh2uv(self, signal_mesh_domain):
        if self.mesh_path and self.barycentric_path and self.tIndex_path:
            values_uv = self.values_to_uv(values=signal_mesh_domain, index_img=self.tindex_tensor,
                                          bary_img=self.barycentric_tensor,
                                          faces=torch.tensor(self.mesh.faces, dtype=torch.long))
            return values_uv
        else:
            raise Exception("Not supported!"
                            "To support this method, you must supply mesh_path & barycentric_path & tIndex_path "
                            "when creating a PatternMeshUVConvertor object.")

    # internal methods
    def to_grid_vertices(self, signal_pattern_domain):
        tensor = torch.Tensor(signal_pattern_domain)
        tensor = torch.rot90(torch.flip(tensor, dims=(1,)), k=2, dims=(0, 1))
        return tensor.view((-1,) + tensor.shape[2:])

    def to_pattern_domain(self, signal_grid_verts):
        tensor = signal_grid_verts.reshape((self.pattern_size[0], self.pattern_size[1]) + signal_grid_verts.shape[1:])
        tensor = torch.flip(torch.rot90(tensor, k=2, dims=(0, 1)), dims=(1,))
        return tensor

    def parse_files(self):
        if self.mesh_to_grid_index_map_path:
            with open(self.mesh_to_grid_index_map_path) as f:
                lines = f.readlines()
                self.map_mesh2grid = np.array([int(line.strip()) for line in lines])

        if self.grid_to_mesh_index_map_path:
            with open(self.grid_to_mesh_index_map_path) as f:
                lines = f.readlines()
                self.map_grid2mesh = np.array([int(line.strip()) for line in lines])

        if self.mesh_path:
            self.mesh = trimesh.load_mesh(self.mesh_path, process=False, validate=False)

        if self.barycentric_path and self.tIndex_path:
            self.register_buffer('barycentric_tensor', ReadTensorFromBinaryFile(self.barycentric_path).permute(1, 2, 0))
            self.register_buffer('tindex_tensor', ReadTensorFromBinaryFile(self.tIndex_path))
            self.uv_size = self.barycentric_tensor.shape[:2]


    def values_to_uv(self, values, index_img, bary_img, faces):
        '''

        :param values: |V| x d
        :param index_img: shape U x V
        :param bary_img: shape U x V x 3
        :return:
        '''
        uv_size = index_img.shape[0]
        index_mask = index_img != -1
        idxs_flat = faces[index_img[index_mask].to(torch.int64), :]
        bary_flat = bary_img[index_mask].to(torch.float32)
        # NOTE: here we assume

        values_faces = values[idxs_flat]
        inf_idx = torch.where(torch.isinf(values_faces))
        all_face_inf_idx = torch.where(torch.all(torch.any(torch.isinf(values_faces), dim=-1), dim=-1))
        values_faces[inf_idx[0], inf_idx[1], :] = 0 # set into arbitrary value so that 0 * value = 0

        bary_flat[inf_idx[0], inf_idx[1]] = 0 # nullify weight on vertices where the signal is not defined
        bary_flat = torch.nn.functional.normalize(bary_flat, p=1, dim=1) # normalize the barycenters
        values_flat = torch.sum(values_faces * bary_flat[:, :, None] , axis=1)
        values_uv = - np.inf * torch.ones(
            uv_size,
            uv_size,
            values.shape[-1],
            dtype=values.dtype,
            device=values.device,
        )
        values_flat[all_face_inf_idx[0], :] = -np.inf # return inf if all vertices values is inf
        values_uv[index_mask, :] = values_flat
        return values_uv


if __name__ == '__main__':
    image_registration_path = '/Users/oshrihalimi/Downloads/2069.npy'
    image_registration_data = np.load(image_registration_path, allow_pickle=True).item()
    board_locations = np.array(
        [image_registration_data['board_location'][idx] for idx in image_registration_data['board_location'].keys()])
    image_locations = np.array(
        [image_registration_data['image_location'][idx] for idx in image_registration_data['board_location'].keys()])
    board_locations = board_locations.astype(int)

    pattern_size = np.array([300, 900])
    pixel_location_pattern_domain = - np.inf * np.ones((pattern_size[0], pattern_size[1], 2))
    pixel_location_pattern_domain[board_locations[:, 0], board_locations[:, 1], :] = image_locations

    plt.imshow(pixel_location_pattern_domain[:, :, 0])
    plt.show()

    # pixel_location_pattern_domain is a signal in pattern domain, shape: [H, W, d]
    mesh_to_grid_index_map_path = '/Users/oshrihalimi/Projects/cloth_completion/assets/domain_conversion_files/seamless_to_grid_vertexMap.txt'
    grid_to_mesh_index_map_path = '/Users/oshrihalimi/Projects/cloth_completion/assets/domain_conversion_files/gridToMeshMap.txt'

    mesh_path = '/Users/oshrihalimi/Projects/cloth_completion/assets/domain_conversion_files/seamlessTextured.ply'
    barycentric_path = '/Users/oshrihalimi/Projects/cloth_completion/assets/domain_conversion_files/uvToMesh_1024/uvGrid_to_mesh_bc.bt'
    tIndex_path = '/Users/oshrihalimi/Projects/cloth_completion/assets/domain_conversion_files/uvToMesh_1024/uvGrid_to_mesh_tIndex.bt'

    convertor = PatternMeshUVConvertor(pattern_size=pattern_size,
                                       mesh_to_grid_index_map_path=mesh_to_grid_index_map_path, grid_to_mesh_index_map_path=grid_to_mesh_index_map_path,
                                       mesh_path=mesh_path,
                                       barycentric_path=barycentric_path, tIndex_path=tIndex_path)
    pixel_location_uv = convertor.pattern2uv(pixel_location_pattern_domain)
    plt.imshow(pixel_location_uv[:, :, 0])
    plt.colorbar()
    plt.savefig('/Users/oshrihalimi/Downloads/debug.png')
    plt.show()
