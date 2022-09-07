import os
import trimesh
import numpy as np
import pyvista as pv
import open3d as o3d

path_driving_results = '/mnt/home/oshrihalimi/compare_methods_paper/donglai_without_pattern_registration/patterned_cloth_pose2clothes_temporalConv_untrackedBody_codec30k_icp_registration/temporal_skip_iter_160000_patterned_cloth_test_collision_resolved/clothesRecon/recon{frame:06d}.ply'
#path_driving_results = '/mnt/home/oshrihalimi/cloth_completion/CodecOutput/cloth_driving_registration_2_cams_cleaned_input_sequence_16/cloth_recon_meshes/iteration-50000/frame_{frame}.ply'
#path_driving_results = '/mnt/home/oshrihalimi/cloth_completion/CodecOutput/cloth_driving_registration_2_cam_cleaned_input_sequence_16_unet_6_down_up_kinematic_model/cloth_recon_meshes/iteration-50000/frame_{frame}.ply'
#path_driving_results = '/mnt/home/donglaix/Codec/CodecOutput/patterned_cloth_pose2clothes_temporalConv_untrackedBody_codec30k/temporal_skip_iter_160000_patterned_cloth_test_collision_resolved/clothesRecon//recon{frame:06d}.ply'

path_gt = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/surface_deformable/incremental_registration/Meshes/skinned-{frame:06d}.ply'

if __name__ == '__main__':
    offset = 4000
    frames = [x for x in range(4000, 4451)]
    N = len(frames)

    avg_euclidean_dist = 0
    avg_chamfer_mesh_to_gt = 0
    avg_chamfer_gt_mesh = 0
    for frame in frames:
        frame_in_file = frame - offset
        mesh_filename = path_driving_results.format(frame=frame_in_file)
        gt_filename = path_gt.format(frame=frame)

        # mesh = pv.read(mesh_filename)
        # gt_mesh = pv.read(gt_filename)
        # euclidean_dist = np.sqrt(np.sum((mesh.points - gt_mesh.points) ** 2, axis=-1)).mean()
        # avg_euclidean_dist = avg_euclidean_dist + euclidean_dist

        mesh_pcd = o3d.io.read_point_cloud(mesh_filename)
        gt_mesh_pcd = o3d.io.read_point_cloud(gt_filename)

        chamfer_mesh_to_gt = np.array(mesh_pcd.compute_point_cloud_distance(gt_mesh_pcd))
        chamfer_gt_to_mesh = np.array(gt_mesh_pcd.compute_point_cloud_distance(mesh_pcd))

        avg_chamfer_mesh_to_gt = avg_chamfer_mesh_to_gt + chamfer_mesh_to_gt.mean()
        avg_chamfer_gt_mesh = avg_chamfer_gt_mesh + chamfer_gt_to_mesh.mean()
        print(frame)

    avg_euclidean_dist = avg_euclidean_dist / N
    avg_chamfer_mesh_to_gt = avg_chamfer_mesh_to_gt / N
    avg_chamfer_gt_mesh = avg_chamfer_gt_mesh / N
    #print(f"Average Euclidean distance: {avg_euclidean_dist}")
    print(f"Average chamfer_mesh_to_gt distance: {avg_chamfer_mesh_to_gt}")
    print(f"Average avg_chamfer_gt_mesh distance: {avg_chamfer_gt_mesh}")
    print(f"Average chamfer distance: {(avg_chamfer_mesh_to_gt + avg_chamfer_gt_mesh) * 0.5}")