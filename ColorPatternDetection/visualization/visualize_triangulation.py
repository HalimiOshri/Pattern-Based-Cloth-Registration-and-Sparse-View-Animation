import numpy as np
import trimesh

if __name__ == '__main__':
    path = '/mnt/home/oshrihalimi/capture/small_pattern_triangulation/3694.pts'
    save_path = '/mnt/home/oshrihalimi/capture/small_pattern_triangulation/3694.obj'
    f = open(path, 'r')
    lines = f.readlines()

    pts = []
    cnt = []
    ind = []
    for line in lines:
        pts.append([float(x) for x in line.split()[1:4]])
        ind.append(int(line.split()[0]))
        cnt.append(float(line.split()[-1]))

    pts = np.array(pts)
    mesh = trimesh.Trimesh(vertices=pts)
    mesh.export(save_path)