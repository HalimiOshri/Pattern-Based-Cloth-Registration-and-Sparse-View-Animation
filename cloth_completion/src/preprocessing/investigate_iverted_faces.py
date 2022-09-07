import trimesh
import os

alignedClothesFile = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/surface_deformable/incremental_registration/Meshes/skinned-004396.ply'
templateClothesFile = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/model/seamlessTextured.obj'
debug_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/debug/'

if __name__ == '__main__':
    os.makedirs(debug_dir, exist_ok=True)
    aligned_mesh = trimesh.load_mesh(alignedClothesFile, process=False, validate=False)
    template_mesh = trimesh.load_mesh(templateClothesFile, process=False, validate=False)

    aligned_mesh.vertices = template_mesh.vertices
    aligned_mesh.export(os.path.join(debug_dir, 'aligned_mesh_topology.ply'))