import torch
# from paper_comparison.map_template_topology import map_mesh_topology
import numpy as np
# from external.KinematicAlignment.utils.tensorIO import WriteTensorToBinaryFile, ReadTensorFromBinaryFile
from src.pyutils.io import load_obj
# from src.pyutils import limp_utils
import pyvista as pv
import trimesh
from src.pyutils.geomutils import vert_normals

if __name__ == '__main__':
    # Get the LBO normalized eigenvectors for low res mesh
    # template_obj_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured_lowRes.obj'
    # v, vt, vi, vti = load_obj(template_obj_path)
    # v = torch.Tensor(np.array(v))[None, :, :].to(device='cuda', dtype=torch.double)
    # vi = torch.Tensor(np.array(vi))[None, :, :].to(device='cuda', dtype=torch.double)
    # L, area_matrix, area_matrix_inv, W = limp_utils.LBO(v, vi.long())
    #
    # B = torch.bmm(torch.bmm(area_matrix_inv ** (0.5), W), area_matrix_inv ** (0.5))
    # print(torch.all(B[0] == B[0].permute(1, 0)))
    # B = 0.5 * (B[0] + B[0].permute(1, 0))
    # eigval, eigvec = torch.linalg.eigh(B)
    # L_eigvec = torch.bmm((area_matrix_inv ** (0.5)), eigvec[None])
    # # L @ L_eigvec - eigval[None, None, :] * L_eigvec ~ 0
    # scale = torch.diag((L_eigvec.permute(0, 2, 1) @ area_matrix_inv @ L_eigvec)[0])
    # normalized_L_eigvec = (scale[None, None, :] ** (-0.5)) * L_eigvec
    #
    # inner_product = normalized_L_eigvec.permute(0, 2, 1) @ area_matrix_inv @ normalized_L_eigvec
    # torch.save(normalized_L_eigvec[0].detach().cpu(),
    #            '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/LBO_eigenvectors_seamlessTextured_lowRes.pt')
    # print("Hi")

    # Lift the LBO normalized eigenvectors to high res mesh
    # source_aligned_mesh_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured_lowRes.obj'
    # target_aligned_mesh_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured.obj'
    # save_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/topology_transfer/topological_mapping.npy'
    # map_topology = map_mesh_topology(source_mesh_path=source_aligned_mesh_path, target_mesh_path=target_aligned_mesh_path, save_path=save_path)
    # topology_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/topology_transfer/topological_mapping.npy'
    # map_topology = np.load(topology_path, allow_pickle=True).item()
    #
    # source_eigenvectors = torch.load('/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/LBO_eigenvectors_seamlessTextured_lowRes.pt')
    # target_eigenvectors = np.sum(map_topology["bcc"][:, :, None] * source_eigenvectors[map_topology["vind"], :].numpy(), axis=1)
    # torch.save(target_eigenvectors[:, :1000], '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/LBO_eigenvectors_seamlessTextured.pt')

    # Show the eigenfunctions
    # eigvec = torch.load('/Users/oshrihalimi/Projects/cloth_completion/assets/LBO_eigenvectors_seamlessTextured.pt')
    #
    # mesh_path = '/Users/oshrihalimi/Projects/cloth_completion/assets/domain_conversion_files/seamlessTextured.obj'
    # v, vt, vi, vti = load_obj(mesh_path)
    # mesh = pv.PolyData(np.array(v), np.concatenate((3 * np.ones((len(vi), 1)), np.array(vi)), axis=-1).astype(int))
    # pv.global_theme.background = 'white'
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, rgb=True, colormap='jet', scalars=np.stack(3*(eigvec[:, 999],), axis=1))
    # plotter.show()

    # Random deformation
    frame = 4000
    num_eigs = 200
    cutoff_eig = 20
    scale_std = 50

    eigvec = torch.load('/Users/oshrihalimi/Projects/cloth_completion/assets/LBO_eigenvectors_seamlessTextured.pt')
    eigvec = eigvec[:, :num_eigs]

    for frame in [4046]: #[488, 1000, 3000, 2000, 4000]:
        for i in range(0, 10):
            input_path = f'/Users/oshrihalimi/Projects/cloth_completion/assets/kinematic_model_examples/{frame}.ply'
            mesh = trimesh.load_mesh(input_path, process=False, validate=False)
            n = vert_normals(torch.Tensor(mesh.vertices[None, :, :]),
                             torch.tensor(mesh.faces[None, :, :], dtype=torch.int64))[0].numpy()

            coeff = np.random.randint(2, size=num_eigs)
            scale = scale_std * np.random.randn()
            filter = scale * np.exp(-np.array([x for x in range(num_eigs)])/cutoff_eig)

            deformation = n * np.sum(filter[None, :] * coeff[None, :] * eigvec, axis=1)[:, None]
            mesh.vertices = mesh.vertices + deformation
            mesh.export(f'/Users/oshrihalimi/Projects/cloth_completion/assets/kinematic_model_examples/{frame}_deformed_scale_cutoff_{cutoff_eig}_{scale:02f}.ply')
            print("Hi")

