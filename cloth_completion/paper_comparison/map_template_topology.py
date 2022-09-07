import os

import trimesh
import numpy as np
from src.pyutils.io import load_obj
import igl
import torch as th
import imageInterpolater

def map_mesh_topology(source_mesh_path, target_mesh_path, save_path):
    v, vt, vi, vti = load_obj(target_mesh_path)
    v = np.array(v)
    vt = np.array(vt)
    vi = np.array(vi)
    vti = np.array(vti)

    faceUV = np.reshape(vt[vti, :], (-1, 6))

    v = th.tensor(v, dtype=th.float32)
    vt = th.tensor(vt, dtype=th.float32)
    vi = th.tensor(vi, dtype=th.int32)
    vti = th.tensor(vti, dtype=th.int32)
    faceUV = th.tensor(faceUV, dtype=th.float32)

    # calc mapping betwen high res icp and low res icp (containing uv)
    source_mesh_vertices, _, source_mesh_faces, _ = load_obj(source_mesh_path)
    bv = np.array(source_mesh_vertices)
    bf = np.array(source_mesh_faces)
    cv = np.array(v)

    sqrD, I, C = igl.point_mesh_squared_distance(cv, bv, bf)
    bcc = igl.barycentric_coordinates_tri(C.astype(float), bv[bf[I, 0]], bv[bf[I, 1]],
                                          bv[bf[I, 2]])  # barycentric coords
    vind = np.stack((bf[I, 0], bf[I, 1], bf[I, 2]), axis=1)

    topological_mapping = {'faceUV': faceUV, 'vt': vt, 'vi': vi, 'vti': vti, 'bcc': bcc, 'vind': vind}
    np.save(save_path, topological_mapping, allow_pickle=True)

if __name__ == '__main__':
    source_aligned_mesh_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/topology_transfer/icp-004396.obj'
    target_aligned_mesh_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/topology_transfer/our-registration-004396_textured.obj'
    save_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/topology_transfer/topological_mapping.npy'
    map_mesh_topology(source_mesh_path=source_aligned_mesh_path, target_mesh_path=target_aligned_mesh_path, save_path=save_path)

    topology_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/topology_transfer/topological_mapping.npy'
    map_topology = np.load(topology_path, allow_pickle=True).item()
    # source_path = '/mnt/home/oshrihalimi/compare_methods_paper/donglai_without_pattern_registration/data_processing/codecResClothesUV/'
    # source_path = '/mnt/home/oshrihalimi/compare_methods_paper/donglai_without_pattern_registration/train_donglai_method_icp_registration/test_output_iter_30000_patterned_cloth_train_collision_resolved/clothesRecon/recon{frame:06d}.ply'
    # target_path = '/mnt/home/oshrihalimi/compare_methods_paper/donglai_without_pattern_registration/train_donglai_method_icp_registration/test_output_iter_30000_patterned_cloth_train_collision_resolved/clothesReconUV_pattern_topology/{frame:06d}.ply'

    source_path = '/mnt/home/oshrihalimi/compare_methods_paper/donglai_without_pattern_registration/patterned_cloth_pose2clothes_temporalConv_untrackedBody_codec30k_icp_registration/temporal_skip_iter_160000_patterned_cloth_test_collision_resolved/clothesRecon/recon{frame:06d}.ply'
    target_path = '/mnt/home/oshrihalimi/compare_methods_paper/donglai_without_pattern_registration/patterned_cloth_pose2clothes_temporalConv_untrackedBody_codec30k_icp_registration/temporal_skip_iter_160000_patterned_cloth_test_collision_resolved/clothesRecon_pattern_topology/recon{frame:06d}.obj'

    os.makedirs(target_path, exist_ok=True)

    offset = 4000
    for frame in range(4000, 4451):
        source_mesh = trimesh.load_mesh(source_path.format(frame=frame - offset), process=False)
        target_vertices = np.sum(map_topology["bcc"][:, :, None] * source_mesh.vertices[map_topology["vind"]], axis=1)
        target_mesh = trimesh.Trimesh(vertices=target_vertices, faces=map_topology["vi"], process=False, validate=False)
        target_mesh.export(target_path.format(frame=frame))
        # imageInterpolater.saveplyfaceuv(target_path.format(frame=frame), th.FloatTensor(target_vertices), map_topology["vi"], map_topology["faceUV"])
        print(frame)

