import sys
#sys.path.append('/mnt/home/oshrihalimi/KinematicAlignment/utils/')
sys.path.append('/Users/oshrihalimi/Projects/KinematicAlignment/utils/')
import tensorIO
import cv2
import numpy as np

class Filter:
    def __init__(self, path_segmentations, path_type_detections, conf):
        self.path_segmentations = path_segmentations
        self.path_type_detections = path_type_detections
        self.conf = conf

        self.size_thresh_relative_median = {'corner': 5, 'center': 5}
        self.size_thresh_num_std = {'corner': 2, 'center': 2}

        self.colormap_color = np.array([
            [0, 0, 0],  # black
            [0, 0, 255],  # red
            [0, 255, 0],  # green
            [0, 255, 255],  # yellow
            [255, 0, 0],  # blue
            [255, 0, 255],  # magenta
            [255, 255, 0],  # cyan
            [255, 255, 255]  # white
        ])

    def get_keypoints(self):
        segmentation_path = self.path_segmentations
        type_path = self.path_type_detections

        # get segmentation
        segmentation_np = cv2.imread(segmentation_path)  # 0 - background, 1 - cloth, 2 - body
        cloth_mask = 1 * (segmentation_np == 1)[:, :, 0]

        # get type
        type_result_tensor = tensorIO.ReadTensorFromBinaryFile(type_path)
        type_result_np = type_result_tensor[0].detach().permute(1, 2, 0).numpy()
        type_labels = np.argmax(type_result_np, axis=2)

        # get corners & centers
        corners_mask = 1 * (type_labels == 1) * cloth_mask
        centers_mask = 1 * (type_labels == 2) * cloth_mask

        corners_centroids = self.get_centroids(mask=corners_mask, type='corner')
        centers_centroids = self.get_centroids(mask=centers_mask, type='center')

        return {'corners': corners_centroids, 'centers': centers_centroids, 'color': None}

    def get_centroids(self, mask, type):
        mask = mask.astype(np.uint8)
        cc = cv2.connectedComponentsWithStats(
            mask, 8, cv2.CV_32S)

        # ignore background cc
        (numLabels, labels, stats, centroids) = cc
        centroids = centroids[1:, :]
        stats = stats[1:, :]
        labels = labels - 1 # background maps to -1, other cc's start from 0-index
        numLabels = numLabels - 1
        cc = (numLabels, labels, stats, centroids)

        # filter
        good_idx = self.filter_cc(cc, type)

        return centroids[good_idx, :] # igonre the background cc

    def filter_cc(self, cc, type):
        cc = self.filter_cc_size(cc, type)
        return cc

    def filter_cc_size(self, cc, type):
        (numLabels, labels, stats, centroids) = cc

        cc_size = stats[:, 4]
        median_size = np.median(cc_size)
        mean_size = np.mean(cc_size)
        std_size = np.std(cc_size)

        # meant to remove big corner clusters
        good_idx = np.logical_not(np.logical_or(cc_size > self.size_thresh_relative_median[type] * median_size,
                                         cc_size > mean_size + self.size_thresh_num_std[type] * std_size))

        return good_idx
