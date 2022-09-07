import trimesh
import PIL
from src.pyutils.io import load_obj, write_obj
import cv2
import os
import numpy as np

# geometry_path = '/mnt/home/oshrihalimi/cloth_completion/CodecOutput/cloth_driving_registration_2_cam_cleaned_input_sequence_16_unet_6_down_up_kinematic_model/cloth_recon_meshes/iteration-80000'
# geometry_path = '/mnt/home/donglaix/Codec/CodecOutput/patterned_cloth_pose2clothes_temporalConv_untrackedBody_codec30k/temporal_skip_iter_160000_patterned_cloth_test_collision_resolved/clothesRecon'
#geometry_path = '/Users/oshrihalimi/Results/driving_pixel_registration/binocular/cloth_driving_registration_2_cam_cleaned_input_sequence_16_unet_6_down_up_kinematic_model/cloth_recon_meshes/iteration-12000/frame_{frame}.ply'
#geometry_path = '/Users/oshrihalimi/Downloads/IR_optimized_train_set_donglai/test_output_iter_30000_patterned_cloth_train_collision_resolved/clothesRecon/recon{frame:06d}.ply'

# save_path = '/mnt/home/oshrihalimi/cloth_completion/CodecOutput/donglai_pose_driving/'
#save_path = '/Users/oshrihalimi/Results/driving_pixel_registration/binocular/cloth_driving_registration_2_cam_cleaned_input_sequence_16_unet_6_down_up_kinematic_model/cloth_recon_meshes/iteration-12000/frame_{frame}_textured.obj'

# template_obj_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/domain_conversion_files/seamlessTextured.obj'
template_obj_path = '/Users/oshrihalimi/Projects/cloth_completion/assets/domain_conversion_files/seamlessTextured.obj'

#geometry_path = '/Volumes/ElementsB/Paper/comparison_data/DonglaiPoseDrivingICPRegistration/patterned_cloth_pose2clothes_temporalConv_untrackedBody_codec30k_icp_registration/temporal_skip_iter_160000_patterned_cloth_test_collision_resolved/clothesRecon/recon{frame:06d}.ply'
geometry_path = '/Volumes/ElementsB/Paper/Results/fully_bimonocular_driving_with_LBO_augmentation/cloth_recon_meshes/iteration-35000/frame_{frame}.ply'
save_path = '/Volumes/ElementsB/Paper/Results/fully_bimonocular_driving_with_LBO_augmentation/cloth_recon_meshes/iteration-35000/frame_{frame}_textured.obj'
#template_obj_path = '/Users/oshrihalimi/Projects/cloth_completion/assets/topology_transfer/icp-004396.obj'
texture_path = '/Users/oshrihalimi/Projects/cloth_completion/assets/texture_cats.jpeg'

if __name__ == '__main__':
    cv2.imread(texture_path)
    v, vt, vi, vti = load_obj(template_obj_path)
    texture = cv2.imread(texture_path)
    texture = np.rot90(texture, k=-1).copy()
    material = trimesh.visual.material.SimpleMaterial(image=PIL.Image.fromarray(texture))
    color_visuals = trimesh.visual.TextureVisuals(uv=vt, image=PIL.Image.fromarray(texture), material=material)

    for frame in range(4000, 4451):
        filename = geometry_path.format(frame=frame)
        mesh = trimesh.load_mesh(filename, process=False, validate=False)
        save_filename = save_path.format(frame=frame)
        write_obj(save_filename, mesh.vertices, vt, vi, vti)
        print(save_filename)
