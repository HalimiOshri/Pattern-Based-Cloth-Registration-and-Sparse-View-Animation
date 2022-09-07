import pygeodesic.geodesic as geodesic
import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors

# mesh_path = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/surface_deformable/incremental_registration/Meshes/skinned-{frame:06d}.ply'
mesh_path = '/mnt/home/oshrihalimi/compare_methods_paper/donglai_without_pattern_registration/train_donglai_method_icp_registration/test_output_iter_30000_patterned_cloth_train_collision_resolved/clothesRecon/recon{frame:06d}.ply'
# mesh_path = '/mnt/home/oshrihalimi/compare_methods_paper/donglai_without_pattern_registration/data_processing/codecResClothesUV/{frame:06d}.ply'
# mesh_path = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/big_pattern/surface_deformable/incremental_registration/Meshes_03/skinned-{frame:06d}.ply'

if __name__ == '__main__':
    reference_frame = 500
    # reference_frame = 4450
    # reference_frame = 4451
    num_source = 20
    offset = 488 # 0

    reference_filename = mesh_path.format(frame=reference_frame - offset)
    reference_mesh = trimesh.load_mesh(reference_filename, process=False, validate=False)
    source_indices = np.random.randint(reference_mesh.vertices.shape[0], size=num_source)
    # target_indices = np.random.randint(reference_mesh.vertices.shape[0], size=num_source)
    target_indices = None

    points = reference_mesh.vertices
    faces = reference_mesh.faces
    geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)

    # X = reference_mesh.vertices[source_indices]
    # Y = reference_mesh.vertices
    # nbrs = NearestNeighbors(n_neighbors=1001, algorithm='ball_tree').fit(Y)
    # distances, indices = nbrs.kneighbors(X)
    # #target_indices = source_indices[indices[:, 1]]
    # target_indices = indices[:, 1000]

    distances_reference = []
    for i in range(source_indices.shape[0]):
        dist, _ = geoalg.geodesicDistances(np.array([source_indices[i]]), None)
        distances_reference.append(dist)
        print(f'{i}: ref distance: {dist}')
    distances_reference = np.stack(distances_reference, axis=0)

    # frames = [x for x in range(frame_start + stride, frame_end, stride)]
    frames = [1000, 1500, 2000, 2500, 3000, 3500, 3999]
    # frames = [x for x in range(500, 4000, 500)]
    # frames = [x for x in range(4500, 8000, 500)]

    num_frames = len(frames)
    avg_distortion = 0
    N = num_frames * num_source
    for frame in frames:
        print(frame)
        filename = mesh_path.format(frame=frame - offset)
        mesh = trimesh.load_mesh(filename, process=False, validate=False)
        points = mesh.vertices
        faces = mesh.faces
        geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)

        for i in range(source_indices.shape[0]):
            dist, _ = geoalg.geodesicDistances(np.array([source_indices[i]]), None)
            delta = np.abs(dist - distances_reference[i])
            delta[delta == np.inf] = 0
            num_valid = np.sum(delta != np.inf)
            distortion = np.sum(delta) / num_valid
            print(f'{i}: ref. distance: {distances_reference[i]}, distance: {dist}, distortion: {distortion}')

            if distortion == np.inf:
                N = N - 1
                continue
            avg_distortion = avg_distortion + distortion

    avg_distortion = avg_distortion / N
    print(f'average distortion: {avg_distortion}')