import os
import cv2
import time
import numpy as np
from heterogeneous_graph import HeterogeneousGraphGenerator
from learning_algorithms.process_segmentations.registration import Registration
from learning_algorithms.process_segmentations.filter_detector_results import Filter
from learning_algorithms.process_segmentations.detections import Detections
from visualization.heterogeneous_graph import save_heterogeneous_graph

def process_frame(color_detections_path, keypoint_detections_path, segmentation_path):
    start = time.time()
    filter = Filter(path_segmentations=segmentation_path, path_type_detections=keypoint_detections_path, conf={})
    keypoints = filter.get_keypoints()
    end = time.time()
    print(f"Filtering time [sec]: {end - start}")

    # Heterogeneous graph - revisited
    # start = time.time()
    # d = HeterogeneousGraphGenerator(keypoints=keypoints)
    # hetero_graph, homogeneous_edges = d.calc_final_graph()
    # end = time.time()
    # print(f'Graph (NEW) building time [sec]: {end - start}')

    # Original graph generation
    start = time.time()
    d = Detections(corners=np.flip(keypoints['corners'], axis=1),
                   centers=np.flip(keypoints['centers'], axis=1),
                   detected_color_path=color_detections_path,
                   debug_path=None)
    homogeneous_graph = d.get_quad_cyclic_graph()
    end = time.time()
    print(f'Graph (ORIGINAL) building time [sec]: {end - start}')

    # Registration
    start = time.time()
    r = Registration(quad_cyclic_graph=homogeneous_graph, image_path=image_path,
                     table=hash_table, color_board=color_board, save_path=registration_path,
                     save_path_image=registration_debug_image_path)
    registration = r.register_nodes(save_image=True)
    end = time.time()

    return hetero_graph, homogeneous_edges, homogeneous_graph


if __name__ == '__main__':
    visualization_path = '/Users/oshrihalimi/Data/CornerTriangulation/visualization/'
    camera_id_path = '/Users/oshrihalimi/Data/CornerTriangulation/camIds'
    save_path = None

    frame = 4460
    with open(camera_id_path) as f:
        camera_ids = [line for line in f.read().splitlines()]

    for camera in camera_ids:
        color_detections_path = f'/Users/oshrihalimi/Data/CornerTriangulation/color_detector/{camera}-{frame:06d}.bt'
        keypoint_detections_path = f'/Users/oshrihalimi/Data/CornerTriangulation/corners_center_detector/{camera}-{frame:06d}.bt'
        segmentation_path = f'/Users/oshrihalimi/Data/CornerTriangulation/segmentation/image-{camera}-{frame:06d}.png'
        image_path = f'/Users/oshrihalimi/Data/CornerTriangulation/capture/image-{camera}-{frame:06d}.png'

        hetero_graph, homogeneous_edges, homogeneous_graph = process_frame(color_detections_path, keypoint_detections_path, segmentation_path)

        # Save filtered keypoints
        # filtered_keypoints_vis_dir = os.path.join(visualization_path, 'filtered_keypoints')
        # os.makedirs(filtered_keypoints_vis_dir, exist_ok=True)
        # save_keypoints(keypoints, image_path, os.path.join(filtered_keypoints_vis_dir, f'{camera}-{frame:06d}.png'))

        # Save nearest opposites
        nearest_opposites_vis_dir = os.path.join(visualization_path, 'nearest_opposites')
        os.makedirs(nearest_opposites_vis_dir, exist_ok=True)
        save_heterogeneous_graph(hetero_graph, homogeneous_edges, homogeneous_graph,
                               image_path, os.path.join(nearest_opposites_vis_dir, f'{camera}-{frame:06d}.png'))

