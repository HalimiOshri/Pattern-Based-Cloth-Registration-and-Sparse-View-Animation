import cv2
import numpy as np
import pyvista as pv
from learning_algorithms.process_segmentations.filter_detector_results import Filter

def save_keypoints(keypoints, image_path, save_path):
    image = pv.read(image_path)
    corners = np.pad(keypoints['corners'], (0, 1))
    centers = np.pad(keypoints['centers'], (0, 1))

    transformation = np.array([[1, 0, 0, 0],
                               [0, -1, 0, image.bounds[3]],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
    corners = pv.PolyData(corners).transform(transformation).points
    centers = pv.PolyData(centers).transform(transformation).points

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(corners, color='red', point_size=3.,
                     render_points_as_spheres=True)
    plotter.add_mesh(centers, color='green', point_size=3.,
                     render_points_as_spheres=True)

    plotter.camera_position = 'xy'
    plotter.camera.roll += 0
    plotter.add_mesh(image, rgb=True)
    plotter.show(screenshot=save_path, window_size=[10000, 10000])
    plotter.close()
    print("Hi")

if __name__ == '__main__':
    keypoint_detections_path = f'/Users/oshrihalimi/Downloads/registration_pipeline_figure/keypoints_400872_3000.bt'
    segmentation_path = '/Users/oshrihalimi/Downloads/registration_pipeline_figure/image_400872_3000_segmentation.png'
    image_path = '/Users/oshrihalimi/Downloads/registration_pipeline_figure/original_image_400872_3000.png'
    save_path = '/Users/oshrihalimi/Downloads/registration_pipeline_figure/filtered_keypoints_400872_3000.png'

    filter = Filter(path_segmentations=segmentation_path, path_type_detections=keypoint_detections_path, conf={})
    keypoints = filter.get_keypoints()
    save_keypoints(keypoints, image_path, save_path)
