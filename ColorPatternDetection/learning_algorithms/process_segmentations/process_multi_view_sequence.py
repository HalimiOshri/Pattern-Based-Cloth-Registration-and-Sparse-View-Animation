#!/usr/bin/env python3

import os
import cv2
import numpy as np
import time
from multiprocessing import Pool
from filter_detector_results import Filter
from detections import Detections
from registration import Registration
import pickle

def process_heatmap(camera, frame,
                    color_path, type_path, segmentation_path, image_path,
                    hash_table, color_board,
                    registration_path, registration_debug_image_path,
                    debug_path=None):
    start_total = time.time()

    try:
        if debug_path:
            os.makedirs(debug_path, exist_ok=True)
        # camera-frame paths
        segmentation_path = os.path.join(segmentation_path, f'{camera}', f'image{frame:04d}.png')
        color_path = os.path.join(color_path, f'{camera}', f'{frame}.bt')
        type_path = os.path.join(type_path, f'{camera}', f'{frame}.bt')
        image_path = os.path.join(image_path, f'{camera}', f'image{frame:04d}.png')
        registration_path = os.path.join(registration_path, f'{camera}', f'{frame}.npy')
        registration_debug_image_path = os.path.join(registration_debug_image_path, f'{camera}', f'image{frame:04d}.png')

        if os.path.isfile(registration_path):
            print(f"FILE EXISTS: {camera}, {frame}")
            return

        # Filtering
        start = time.time()
        filter = Filter(path_segmentations=segmentation_path, path_color_detections=color_path, path_type_detections=type_path, conf={}, debug_path=debug_path)
        keypoints = filter.get_keypoints()
        end = time.time()
        print(f"Filtering time [sec]: {end - start}")

        # Graph building
        start = time.time()
        d = Detections(corners=np.flip(keypoints['corners'], axis=1),
                       centers=np.flip(keypoints['centers'], axis=1),
                       detected_color_path=color_path,
                       debug_path=debug_path)
        graph = d.get_quad_cyclic_graph()
        end = time.time()
        print(f'Graph building time [sec]: {end - start}')

        if debug_path:
            detection_path = os.path.join(debug_path, 'detections')
            os.makedirs(detection_path, exist_ok=True)
            with open(os.path.join(detection_path, f'detections_{camera}_{frame}.pkl'), 'wb') as outp:
                pickle.dump(d, outp, pickle.HIGHEST_PROTOCOL)

        # Registration
        start = time.time()
        r = Registration(quad_cyclic_graph=graph, image_path=image_path,
                         table=hash_table, color_board=color_board, save_path=registration_path,
                         save_path_image=registration_debug_image_path)
        r.register_nodes(save_image=True)
        end = time.time()
        print(f'Registration time [sec]: {end - start}')

        end_total = time.time()
        print(f'Elapsed time [sec]: {end_total - start_total}')

    except:
        print(f"Exception occured with camera {camera} frame {frame}")

if __name__ == '__main__':
    start = time.time()
    # Paths definitions
    save_path = '/mnt/home/oshrihalimi/capture/big_pattern_hash_detection_results' #'/mnt/home/oshrihalimi/capture/keypoints_sets/explore/cam401244_frame_0' #

    # camera = 'cam401244'  # 400143 #400894 #400936
    # frame = 0

    camera_list_path = '/mnt/home/oshrihalimi/capture/remaining_401245.txt' #'/mnt/home/oshrihalimi/capture/camera_list.txt'
    path_segmentations = '/mnt/captures/studies/pilots/sociopticon/Aug18/s--20210823--1323--0000000--pilot--patternCloth/experimental/segmentation_part/predictions/segmentation/'
    path_color_detections = '/mnt/home/oshrihalimi/capture/color_detector_results/FINAL_clean_detector_sparse_cross_entropy_400_random_keypoints_with_resize_aug_0_v1/'
    path_type_detections = '/mnt/home/oshrihalimi/capture/keypoint_location_detector_results/corners_centers_detector_unet-5_double_conv_4_conv_layers_cross_entropy_1_sparse_and_1_dense_input_gray_aug_v1'
    path_images = '/mnt/captures/studies/pilots/sociopticon/Aug18/s--20210823--1323--0000000--pilot--patternCloth/undistorted'
    board_data = np.load(
        '/mnt/home/oshrihalimi/pycharm/ColorPatternDetection/color_pattern_files/hash_table_color_big_pattern.npy',
        allow_pickle=True)
    hash_table = board_data.item()['table']
    color_board = board_data.item()['color']

    # process_heatmap(camera=camera, frame=frame,
    #                 color_path=path_color_detections, type_path=path_type_detections,
    #                 segmentation_path=path_segmentations, image_path=path_images,
    #                 hash_table=hash_table, color_board=color_board,
    #                 registration_path=save_path, registration_debug_image_path=save_path,
    #                 debug_path=save_path)

    with open(camera_list_path, 'r') as f:
        cameras = f.read().splitlines()

    frame_inds = [x for x in range(4451, 11756)] # frame range in small pattern : 0 - 4450, frame range in big pattern: 4451 - 11755

    with Pool(processes=250) as pool:
        pool.starmap(process_heatmap, ((camera, frame,
                    path_color_detections, path_type_detections,
                    path_segmentations, path_images,
                    hash_table, color_board,
                    save_path, save_path)
                    for camera in cameras for frame in frame_inds))

    end = time.time()
    print(f'Total elapsed time for multi-camera sequence processing [sec]: {end - start}')