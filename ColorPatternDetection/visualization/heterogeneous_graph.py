import numpy as np
import pyvista as pv
import itertools
from learning_algorithms.process_segmentations.filter_detector_results import Filter
from learning_algorithms.revisited_registration.heterogeneous_graph import HeterogeneousGraphGenerator
from learning_algorithms.process_segmentations.detections import Detections

def save_heterogeneous_graph(heterogeneous_graph, homogeneous_edges, homogeneous_graph, image_path, save_path):
    image = pv.read(image_path)

    initial_nodes = heterogeneous_graph['pixel_location']
    type = heterogeneous_graph['type']
    corners = np.pad(initial_nodes[type == 0, :2], (0, 1))
    centers = np.pad(initial_nodes[type == 1, :2], (0, 1))

    transformation = np.array([[1, 0, 0, 0],
                               [0, -1, 0, image.bounds[3]],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
    corners = pv.PolyData(corners).transform(transformation).points
    centers = pv.PolyData(centers).transform(transformation).points

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(corners, color='red', point_size=5., render_points_as_spheres=True)
    plotter.add_mesh(centers, color='green', point_size=5., render_points_as_spheres=True)

    # edges
    # edges = [[[i, j] for j in heterogeneous_graph['nbr'][i]] for i in range(initial_nodes.shape[0]) if heterogeneous_graph['nbr'][i] != []]
    # edges = np.array(list(itertools.chain.from_iterable(edges)))
    edges = homogeneous_edges
    j1 = np.pad(initial_nodes[edges[:, 0].astype(int), :2], (0, 1))
    j2 = np.pad(initial_nodes[edges[:, 1].astype(int), :2], (0, 1))
    j1 = pv.PolyData(j1).transform(transformation).points
    j2 = pv.PolyData(j2).transform(transformation).points
    plotter.add_arrows(cent=j1, direction=j2 - j1, color='blue')

    plotter.camera_position = 'xy'
    plotter.camera.roll += 0
    plotter.add_mesh(image, rgb=True)
    plotter.show(screenshot=save_path, window_size=[10000, 10000])
    plotter.close()
    print("Hi")

if __name__ == '__main__':
    color_detections_path = f'/Users/oshrihalimi/Downloads/registration_pipeline_figure/400872_3000_color_detection.bt'
    keypoint_detections_path = f'/Users/oshrihalimi/Downloads/registration_pipeline_figure/keypoints_400872_3000.bt'
    segmentation_path = '/Users/oshrihalimi/Downloads/registration_pipeline_figure/image_400872_3000_segmentation.png'
    image_path = '/Users/oshrihalimi/Downloads/registration_pipeline_figure/original_image_400872_3000.png'
    save_path = '/Users/oshrihalimi/Downloads/registration_pipeline_figure/homogeneous_graph_400872_3000.png'

    filter = Filter(path_segmentations=segmentation_path, path_type_detections=keypoint_detections_path, conf={})
    keypoints = filter.get_keypoints()

    h = HeterogeneousGraphGenerator(keypoints=keypoints)
    hetero_graph, homogeneous_edges = h.calc_final_graph()

    d = Detections(corners=np.flip(keypoints['corners'], axis=1),
                   centers=np.flip(keypoints['centers'], axis=1),
                   detected_color_path=color_detections_path,
                   debug_path=save_path)
    graph = d.get_quad_cyclic_graph()
    homo_edges = np.array(graph.edges)

    save_heterogeneous_graph(hetero_graph, homo_edges, None, image_path, save_path)