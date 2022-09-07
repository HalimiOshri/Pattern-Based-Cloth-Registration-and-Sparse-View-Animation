from src.
source_aligned_mesh_path = '/Users/oshrihalimi/Projects/cloth_completion/assets/topology_transfer/icp-004396.obj'
target_aligned_mesh_path = '/Users/oshrihalimi/Projects/cloth_completion/assets/topology_transfer/our-registration-004396_textured.obj'

source_uv_signal_path = ''

if __name__ == '__main__':
    v, vt, f, face_vt = load_obj(target_aligned_mesh_path)
    v = np.array(v)
    vt = np.array(vt)

    f = np.array(f)
    face_vt = np.array(face_vt)
    faceUV = np.reshape(vt[face_vt, :], (-1, 6))

    v = th.tensor(v, dtype=th.float32)
    f = th.tensor(f, dtype=th.int32)
    faceUV = th.tensor(faceUV, dtype=th.float32)

    # calc mapping betwen high res icp and low res icp (containing uv)
    alignedHighRes = trimesh.load(alignedHighRedFile, process=False)
    bv = np.array(alignedHighRes.vertices)
    bf = np.array(alignedHighRes.faces)
    cv = np.array(v)

    sqrD, I, C = igl.point_mesh_squared_distance(cv, bv, bf)
    bcc = igl.barycentric_coordinates_tri(C.astype(float), bv[bf[I, 0]], bv[bf[I, 1]],
                                          bv[bf[I, 2]])  # barycentric coords
    vind = np.stack((bf[I, 0], bf[I, 1], bf[I, 2]), axis=1)