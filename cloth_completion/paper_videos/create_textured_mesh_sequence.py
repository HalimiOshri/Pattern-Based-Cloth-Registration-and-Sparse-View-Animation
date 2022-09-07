import pyvista as pv
import numpy as np
import pyvista as pv
import os
import trimesh
from src.pyutils.io import load_obj

import pyvista as pv
import numpy as np
import pyvista as pv
import os
import trimesh

import imageio
gif = imageio.get_reader('/Volumes/ElementsB/Paper/comparison_data/teaser data/simpson.gif', '.gif')
number_of_frames = len(gif)
gif_frames = [frame for frame in gif]
images = []
for frame in gif_frames:
    image = np.ones((1024, 1024, 3)) * 255
    image[500:500 + 362, 0:0 + 480, :] = frame[:, :, :3]
    images.append(image)



#path_meshes = '/Volumes/ElementsB/Paper/comparison_data/DonglaiPoseDrivingHighQualityRegistration/donglai_pose_driving'
#path_meshes = f'/Volumes/ElementsB/Paper/Results/driving_pixel_registration/binocular/cloth_driving_registration_2_cam_cleaned_input_sequence_16_unet_6_down_up_kinematic_model/cloth_recon_meshes/iteration-12000'
# path_meshes = '/Volumes/ElementsB/SurfaceRegistration/shirt_tracking/small_pattern/surface_deformable/incremental_registration/Meshes/'
# path_meshes = '/Users/oshrihalimi/Downloads/ICP_Registration/codecResClothesUV'
# path_meshes = '/Users/oshrihalimi/Downloads/IR_optimized_train_set_donglai/test_output_iter_30000_patterned_cloth_train_collision_resolved/clothesRecon/'
# path_meshes = '/Users/oshrihalimi/Downloads/IR_optimized_train_set_donglai/test_output_iter_30000_patterned_cloth_train_collision_resolved/clothesRecon/'
# path_meshes = '/Volumes/ElementsB/Paper/comparison_data/DonglaiPoseDrivingICPRegistration/patterned_cloth_pose2clothes_temporalConv_untrackedBody_codec30k_icp_registration/temporal_skip_iter_160000_patterned_cloth_test_collision_resolved/clothesRecon/'
# img_file = '/Volumes/ElementsB/Paper/comparison_data/video_telepresence/files single frame/chessboard_tiled.jpeg'

path_meshes = '/Volumes/ElementsB/Paper/Results/fully_bimonocular_driving_with_LBO_augmentation/cloth_recon_meshes/iteration-35000'
img_file = '/Volumes/ElementsB/Paper/comparison_data/video_telepresence/files single frame/baby_blue_texture.jpeg'
save_path = path_meshes
filename = os.path.join(save_path, f"pixel_registration_driving_back.mp4")

#meshes = [f"recon{x:06d}_textured.obj" for x in range(0, 450)]
#meshes = [f"skinned-{x:06d}_textured.obj" for x in range(488, 4450)]
#meshes = [f"skinned-{x:06d}_textured.obj" for x in range(4451, 11754)]
#meshes = [f"{x:06d}.ply" for x in range(488, 4450)]
#meshes = [f"recon{x:06d}_textured.obj" for x in range(0, 450)]
meshes = [f"frame_{x}_textured.obj" for x in range(4000, 4450)]
n_meshes = len(meshes)
start_frame = 0

if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)
    mesh = pv.read(os.path.join(path_meshes, meshes[0]))
    tex = pv.read_texture(img_file)
    #tex = pv.numpy_to_texture(images[0].astype(np.uint8))
    # If reading from obj file
    v, vt, vti, vi = load_obj(os.path.join(path_meshes, meshes[0]))
    mesh.cell_arrays["Texture Coordinates"] = vti

    pv.global_theme.background = 'white'

    plotter = pv.Plotter()
    plotter.open_movie(filename)
    actor = plotter.add_mesh(mesh, texture=tex) #texture=tex)

    plotter.show(auto_close=False)  # only necessary for an off-screen movie
    plotter.write_frame()  # write initial data

    # Update scalars on each frame
    for i in range(n_meshes):
        #plotter.remove_actor(actor)
        mesh_current = pv.read(os.path.join(path_meshes, meshes[i]))
        mesh.points = mesh_current.points
        mesh.faces = mesh_current.faces
        #curr_tex = pv.numpy_to_texture(images[i % number_of_frames].astype(np.uint8))

        #actor = plotter.add_mesh(mesh, texture=curr_tex)

        # plotter.remove_actor(actor)
        # cur_reg_mesh = pv.read(os.path.join(path_registrations, registrations[i]))
        # pc = pv.PolyData(cur_reg_mesh.points)
        # actor = plotter.add_mesh(pc, color='red', point_size=1)

        plotter.add_text(f"Pixel registration driving, Frame: {i}", name='time-label', color='black')
        plotter.write_frame()  # Write this frame
        print(i)

    # Be sure to close the plotter when finished
    plotter.close()

# path_meshes = '/Users/oshrihalimi/Results/driving_pixel_registration/binocular/cloth_driving_registration_2_cam_cleaned_input_sequence_16_unet_6_down_up_kinematic_model/cloth_recon_meshes/iteration-12000/frame_{frame}_textured.obj'
# img_file = '/Users/oshrihalimi/Downloads/video_telepresence/files single frame/cloth_texture_2.jpeg'
# save_path = '/Users/oshrihalimi/Results/driving_pixel_registration/binocular/cloth_driving_registration_2_cam_cleaned_input_sequence_16_unet_6_down_up_kinematic_model/cloth_recon_meshes/iteration-12000/textured_video.mp4'
# first_frame = 4000
# last_frame = 4450

# if __name__ == '__main__':
#     os.makedirs(save_path, exist_ok=True)
#
#     tex = pv.read_texture(img_file)
#     mesh = pv.read(path_meshes.format(frame=first_frame))
#     v, vt, vti, vi = load_obj(path_meshes.format(frame=first_frame))
#     #mesh.cell_arrays["Texture Coordinates"] = vti
#
#     pv.global_theme.background = 'white'
#
#     plotter = pv.Plotter()
#     plotter.open_movie(save_path)
#     plotter.add_mesh(mesh) #, texture=tex)
#
#     plotter.show(auto_close=False)  # only necessary for an off-screen movie
#     plotter.write_frame()  # write initial data
#     plotter.write_frame()
#     # Update scalars on each frame
#     for frame in range(first_frame, last_frame):
#         mesh_current = pv.read(path_meshes.format(frame=frame))
#         #mesh.points = mesh_current.points
#         #mesh.faces = mesh_current.faces
#         #mesh.cell_arrays["Texture Coordinates"] = vti
#
#         #plotter.add_text(f"Frame: {frame}", name='time-label')
#         plotter.write_frame()  # Write this frame
#
#     # Be sure to close the plotter when finished
#     plotter.close()


