import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors

class RegistrationAnalyser:
    def __init__(self, registrations_dict, thresh_edges=3, thresh_cc_trustable=30, thresh_cc_distance_to_trustable=50, thresh_cc_pixel_distance_to_trustable=20,
                 radius_distance_filtering=10, min_nbr_distance_filtering=10):
        self.registrations = registrations_dict
        self.radius_distance_filtering = radius_distance_filtering # in mm
        self.min_nbr_distance_filtering = min_nbr_distance_filtering
        self.thresh_edges = thresh_edges
        self.thresh_cc_trustable = thresh_cc_trustable
        self.thresh_cc_distance_to_trustable = thresh_cc_distance_to_trustable
        self.thresh_cc_pixel_distance_to_trustable = thresh_cc_pixel_distance_to_trustable

    def filter_outliers(self):
        outlier_mask_distance_based = self.filter_outliers_distance_based()
        outlier_mask = outlier_mask_distance_based

        outlier_mask_edge_based, median = self.filter_outliers_edge_based(outliers_mask=outlier_mask, edge_num_thresh=1)
        outlier_mask = np.logical_or(outlier_mask, outlier_mask_edge_based)

        outlier_mask_cc_based = self.filter_outliers_cc_based(outliers_mask=outlier_mask)
        outlier_mask = np.logical_or(outlier_mask, outlier_mask_cc_based)

        outlier_mask_edge_based, median = self.filter_outliers_edge_based(outliers_mask=outlier_mask, edge_num_thresh=0)
        return outlier_mask, median

    def filter_outliers_distance_based(self):
        mask = self.registrations['detection_mask'] * self.registrations['cloth_mask']
        outlier_mask = np.zeros(mask.shape[:2])

        v = self.registrations['location'][mask[:, :, 0] ==1, :]
        v_tensor_ind = np.stack(np.where(mask[:, :, 0] ==1), axis=1)

        neigh = NearestNeighbors(radius=self.radius_distance_filtering)
        neigh.fit(v)
        rng = neigh.radius_neighbors()
        is_valid_num_nbr = np.array([len(x) > self.min_nbr_distance_filtering for x in rng[0]])
        outlier_inds = v_tensor_ind[np.logical_not(is_valid_num_nbr), :]
        outlier_mask[outlier_inds[:, 0], outlier_inds[:, 1]] = 1
        return outlier_mask

    def filter_outliers_cc_based(self, outliers_mask):
        mask = self.registrations['detection_mask'] * self.registrations['cloth_mask'] * (1 - outliers_mask[:, :, None])
        output = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        big_cc_ind = np.where(stats[:, 4] >= self.thresh_cc_trustable)[0]
        small_cc_ind = np.where(stats[:, 4] < self.thresh_cc_trustable)[0]

        trustable_mask = np.zeros(mask.shape[:2])
        for ind in big_cc_ind:
            trustable_mask = np.logical_or(trustable_mask, labels == ind)
        trustable_mask = np.logical_and(trustable_mask, mask[:, :, 0])

        v_trustable = self.registrations['location'][trustable_mask, :]
        trustable_tensor_ind = np.stack(np.where(trustable_mask), axis=1)
        nbrs = NearestNeighbors(n_neighbors=1).fit(trustable_tensor_ind)

        outlier_mask = np.zeros(mask.shape[:2])
        for ind in small_cc_ind:
            v_internal_to_cc = self.registrations['location'][labels == ind]
            internal_to_cc_tensor_ind = np.stack(np.where(labels == ind), axis=1)

            pixel_distances, indices = nbrs.kneighbors(internal_to_cc_tensor_ind)
            pixel_distance_to_trustable = np.min(pixel_distances)
            closest_internal = np.argmin(pixel_distances)
            closest_trustable = indices[closest_internal]
            distance_3d = np.sum((v_internal_to_cc[closest_internal, :] - v_trustable[closest_trustable, :]) ** 2) ** 0.5

            if pixel_distance_to_trustable > self.thresh_cc_pixel_distance_to_trustable: # cancel this cc: too far from trustable cc in pixel space
                outlier_mask = np.logical_or(outlier_mask, labels == ind)
            elif distance_3d > self.thresh_cc_distance_to_trustable: # cancel this cc: too far from trustable cc in 3D space
                outlier_mask = np.logical_or(outlier_mask, labels == ind)

        return outlier_mask

    def filter_outliers_edge_based(self, outliers_mask, edge_num_thresh):
        v = self.registrations['location']

        # Edges horizontal & vertical
        horizontal_edges = np.diff(v, axis=0)
        vertical_edges = np.diff(v, axis=1)
        mask = self.registrations['detection_mask'] * self.registrations['cloth_mask'] * (1 - outliers_mask[:, :, None])
        horizontal_edges_is_valid = mask[:-1, :, 0] * mask[1:, :, 0]
        vertical_edges_is_valid = mask[:, :-1, 0] * mask[:, 1:, 0]
        horizontal_edges_valid = horizontal_edges[horizontal_edges_is_valid == 1, :]
        vertical_edges_valid = vertical_edges[vertical_edges_is_valid == 1, :]

        median = np.median(np.linalg.norm(np.concatenate((horizontal_edges_valid, vertical_edges_valid), axis=0), axis=1))

        outlier_mask_h = (np.linalg.norm(horizontal_edges, axis=-1) > self.thresh_edges * median) * horizontal_edges_is_valid
        outlier_mask_v = (np.linalg.norm(vertical_edges, axis=-1) > self.thresh_edges * median) * vertical_edges_is_valid

        outlier_mask_h_right = np.concatenate((outlier_mask_h, np.zeros((1, v.shape[1]))), axis = 0)
        outlier_mask_h_left = np.concatenate((np.zeros((1, v.shape[1])), outlier_mask_h), axis = 0)
        outlier_mask_v_up = np.concatenate((outlier_mask_v, np.zeros((v.shape[0], 1))), axis = 1)
        outlier_mask_v_down = np.concatenate((np.zeros((v.shape[0], 1)), outlier_mask_v), axis = 1)

        # Edges diagonal
        diagonal_edges_0 = v[1:, 1:, :] - v[:-1, :-1, :]
        diagonal_edges_1 = v[1:, :-1, :] - v[:-1, 1:, :]
        diagonal_edges_0_is_valid = mask[1:, 1:, 0] * mask[:-1, :-1, 0]
        diagonal_edges_1_is_valid = mask[1:, :-1, 0] * mask[:-1, 1:, 0]
        outlier_mask_d_0 = (np.linalg.norm(diagonal_edges_0, axis=-1) > self.thresh_edges * np.sqrt(2) * median) * diagonal_edges_0_is_valid
        outlier_mask_d_1 = (np.linalg.norm(diagonal_edges_1, axis=-1) > self.thresh_edges * np.sqrt(2) * median) * diagonal_edges_1_is_valid
        outlier_mask_d_00 = np.pad(outlier_mask_d_0, ((0, 1), (0, 1)))
        outlier_mask_d_01 = np.pad(outlier_mask_d_0, ((1, 0), (1, 0)))
        outlier_mask_d_10 = np.pad(outlier_mask_d_1, ((0, 1), (1, 0)))
        outlier_mask_d_11 = np.pad(outlier_mask_d_1, ((1, 0), (0, 1)))

        # Final outliers mask
        outlier_mask = (outlier_mask_h_right + outlier_mask_h_left + outlier_mask_v_up + outlier_mask_v_down + \
                       outlier_mask_d_00 + outlier_mask_d_01 + outlier_mask_d_10 + outlier_mask_d_11) > edge_num_thresh # considered outlier if more than 1 edges is outlier
        return outlier_mask, median