import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pyvista as pv
import time
from collections import Counter
import cv2
import sys
sys.path.append('/mnt/home/oshrihalimi/KinematicAlignment/utils/')
sys.path.append('/Users/oshrihalimi/Projects/KinematicAlignment/utils/')
try:
    import tensorIO
except:
    pass

# TODO: consider updating detetcted keypoints by merging and addition
# TODO: sort edges cyclically

class Detections:
    def __init__(self, corners, centers, detected_color_path, debug_path=None):
        self.debug_path = debug_path

        self.detected_color_path = detected_color_path
        color_labels_tensor = tensorIO.ReadTensorFromBinaryFile(detected_color_path)
        self.color_labels_np = color_labels_tensor.detach().numpy().astype(np.uint8)
        self.corners = corners
        self.centers = centers

        self.n_corners = self.corners.shape[0]
        self.n_centers = self.centers.shape[0]

        print(f"Number of detected corners: {self.n_corners}")
        print(f"Number of detected centers: {self.n_centers}")

        # create a pool of all keypoints, shape: [#corners + #centers, 3] with labels 0-corner, 1-center
        self.all_keypoints = np.concatenate((self.corners, self.centers), axis=0)
        self.all_keypoints = np.concatenate((self.all_keypoints,
                                             np.zeros(shape=(self.n_corners + self.n_centers, 1))), axis=1)
        self.all_keypoints[self.n_corners:, -1] = 1
        self.kpt_label = self.all_keypoints[:, -1]
        self.n_kpt = self.all_keypoints.shape[0]

    def calc_opposites_criterions(self):
        # Direct Geometric Criterion
        dist_std = np.std(self.nearest_opposites['dist'], axis=1)
        dist_mean = np.mean(self.nearest_opposites['dist'], axis=1)
        dist_relative_std = dist_std / dist_mean

        dist_std_first = np.std(self.nearest_opposites['dist'][:, :2], axis=1)
        dist_mean_first = np.mean(self.nearest_opposites['dist'][:, :2], axis=1)
        dist_relative_std_first = dist_std_first / dist_mean_first

        dist_std_second = np.std(self.nearest_opposites['dist'][:, 2:], axis=1)
        dist_mean_second = np.mean(self.nearest_opposites['dist'][:, 2:], axis=1)
        dist_relative_std_second = dist_std_second / dist_mean_second

        p1 = self.all_keypoints[np.stack((np.arange(self.n_kpt),) * 4, axis=1), :-1]
        p2 = self.all_keypoints[self.nearest_opposites['ind'], :-1]
        vector_to_op = p2 - p1
        center_of_mass = np.linalg.norm(np.sum(vector_to_op, axis=1), axis=-1)

        sorted_angles = np.sort(np.arctan2(vector_to_op[..., 0], vector_to_op[..., 1]), axis=-1)
        angles_diff = sorted_angles[:, 2:] - sorted_angles[:, :2]
        cos_double_angles = np.cos(angles_diff)
        has_two_diagonals = np.all(cos_double_angles < - 0.9, axis=1)

        self.opposites_geometric_valid = np.logical_and(
            np.logical_and(has_two_diagonals, center_of_mass < 20),
            np.logical_and(dist_relative_std_first < 0.2, dist_relative_std_second < 0.2)
        )

        # Mutuality Criterion
        # Am I the among the (opposite-) nearest nbr's of my (opposite-) nearest nbr's?
        self.calc_mutual_opposite_nbr()

        # 1-Ring Geometric Criterion
        self.nearest_op_nearest_op = self.nearest_opposites['ind'][self.nearest_opposites['ind'], :]
        self.opposites_geometric_valid_1_ring = np.all(np.all(
                                                    self.opposites_geometric_valid[self.nearest_op_nearest_op],
                                                    axis=-1), axis=-1)

        # Counting Profile Criterion
        nearest_op_nearest_op_aggregate = np.reshape(self.nearest_op_nearest_op,
                                                     (self.nearest_op_nearest_op.shape[0], 16)).tolist()
        self.nearest_op_nearest_op_counter = [Counter(x) for x in nearest_op_nearest_op_aggregate]
        self.opposite_ring_counting_profile = [(sorted(list(x.values())) == [1, 1, 1, 1, 2, 2, 2, 2, 4],
                                               [y[0] for y in x.items() if y[1] == 2],
                                               [y[0] for y in x.items() if y[1] == 1])
                                for x in self.nearest_op_nearest_op_counter]
        self.ring_counting_profile_valid = np.array([x[0] for x in self.opposite_ring_counting_profile])
        self.same_nbr = np.array([x[1] if len(x[1]) == 4 else [-1, -1, -1, -1]
                                  for x in self.opposite_ring_counting_profile])
        self.same_nbr_diag = np.array([x[2] if len(x[2]) == 4 else [-1, -1, -1, -1]
                                  for x in self.opposite_ring_counting_profile])

        # Cirterion Circular
        # For opposite-same
        all_nbr_ind = np.concatenate((self.nearest_opposites['ind'], self.same_nbr), axis=1)
        p1 = self.all_keypoints[np.stack((np.arange(self.n_kpt),) * (4 + 4), axis=1), :-1]
        p2 = self.all_keypoints[all_nbr_ind, :-1]
        vector_to_nbr = p2 - p1
        angles = np.arctan2(vector_to_nbr[..., 0], vector_to_nbr[..., 1])
        ind_sort = np.argsort(angles, axis=-1)
        all_nbr_ind_s = np.take_along_axis(all_nbr_ind, ind_sort, axis=1)
        all_nbr_labels_s = self.kpt_label[all_nbr_ind_s]
        self.circular_opposite_same = np.all(np.abs(np.diff(all_nbr_labels_s, axis=1)), axis=1)

        # For axis-same-diag-same
        all_nbr_ind = np.concatenate((self.same_nbr_diag, self.same_nbr), axis=1)
        axis_diag_label = np.zeros(all_nbr_ind.shape)
        axis_diag_label[:, 4:] = 1 # label axis nbr with "1"
        p1 = self.all_keypoints[np.stack((np.arange(self.n_kpt),) * (4 + 4), axis=1), :-1]
        p2 = self.all_keypoints[all_nbr_ind, :-1]
        vector_to_nbr = p2 - p1
        angles = np.arctan2(vector_to_nbr[..., 0], vector_to_nbr[..., 1])
        ind_sort = np.argsort(angles, axis=-1)
        axis_diag_label_s = np.take_along_axis(axis_diag_label, ind_sort, axis=1)
        self.circular_axis_diag_same = np.all(np.abs(np.diff(axis_diag_label_s, axis=1)), axis=1)
        self.same_nbr_all_sorted_cyc = np.take_along_axis(all_nbr_ind, ind_sort, axis=1)

        # Optional printing
        self.corners_valid_opposites = np.logical_and(self.opposites_geometric_valid, self.kpt_label == 0)
        self.centers_valid_opposites = np.logical_and(self.opposites_geometric_valid, self.kpt_label == 1)

        self.n_corners_valid_opposites = np.sum(self.corners_valid_opposites)
        self.n_centers_valid_opposites = np.sum(self.centers_valid_opposites)

        print(f"Number of corners with valid opposite neighbors: {self.n_corners_valid_opposites}")
        print(f"Number of centers with valid opposite neighbors: {self.n_centers_valid_opposites}")

    def calc_nearest_same(self):
        corners_nbr = NearestNeighbors(n_neighbors=self.n_nn_same + 1, radius=100)
        centers_nbr = NearestNeighbors(n_neighbors=self.n_nn_same + 1, radius=100)

        corners_nbr.fit(self.corners)
        centers_nbr.fit(self.centers)

        corners_same_nbr = corners_nbr.kneighbors(self.corners, self.n_nn_same + 1, return_distance=True)
        centers_same_nbr = centers_nbr.kneighbors(self.centers, self.n_nn_same + 1, return_distance=True)

        self.nearest_same = {"dist": np.concatenate((corners_same_nbr[0][:, 1:], centers_same_nbr[0][:, 1:]), axis=0),
                             "ind": np.concatenate(
                                 (corners_same_nbr[1][:, 1:], centers_same_nbr[1][:, 1:] + self.n_corners), axis=0)
                             }

    def calc_nearest_opposites(self):
        corners_nbr = NearestNeighbors(n_neighbors=4, radius=100)
        centers_nbr = NearestNeighbors(n_neighbors=4, radius=100)

        corners_nbr.fit(self.corners)
        centers_nbr.fit(self.centers)

        corners_op_nbr = centers_nbr.kneighbors(self.corners, 4, return_distance=True)
        centers_op_nbr = corners_nbr.kneighbors(self.centers, 4, return_distance=True)

        self.nearest_opposites = {"dist": np.concatenate((corners_op_nbr[0], centers_op_nbr[0]), axis=0),
                                  "ind": np.concatenate((corners_op_nbr[1] + self.n_corners, centers_op_nbr[1]), axis=0)
                                  }

    def calc_nbr(self):
        neigh = NearestNeighbors(n_neighbors=5, radius=100)
        neigh.fit(self.all_keypoints[:, :-1])

        self.nbr = list(neigh.kneighbors(self.all_keypoints[:, :-1], 5, return_distance=True))

        # Truncate self-neighborhood: distances & indices
        self.nbr[0] = self.nbr[0][:, 1:]
        self.nbr[1] = self.nbr[1][:, 1:]

        self.nbr = {"dist": self.nbr[0],
                    "ind": self.nbr[1]
                    }

        print("Hi!")

    def calc_mutual_same_nbr(self):
        self.mutual_same_nbr = np.all(
            np.any(self.same_nbr['ind'][self.same_nbr['ind'], :] ==  # shape: (kpt's ind, nbr's ind, nbr's nbr's ind)
                   np.arange(self.n_kpt)[:, None, None],
                   axis=2), axis=1)

        # Optional printing
        self.corners_mutual_same_nbr = np.logical_and(self.mutual_same_nbr, self.kpt_label == 0)
        self.centers_mutual_same_nbr = np.logical_and(self.mutual_same_nbr, self.kpt_label == 1)

        self.n_corners_mutual_same_nbr = np.sum(self.corners_mutual_same_nbr)
        self.n_centers_mutual_same_nbr = np.sum(self.centers_mutual_same_nbr)

        print(f"Number of corners with all mutual same neighbors: {self.n_corners_mutual_same_nbr}")
        print(f"Number of centers with all mutual same neighbors: {self.n_centers_mutual_same_nbr}")

    def calc_mutual_opposite_nbr(self):
        self.mutual_opposites_nbr = np.all(
            np.any(self.nearest_opposites['ind'][self.nearest_opposites['ind'],
                   :] ==  # shape: (kpt's ind, nbr's ind, nbr's nbr's ind)
                   np.arange(self.n_kpt)[:, None, None],
                   axis=2), axis=1)

        # Optional printing
        self.corners_mutual_opposites_nbr = np.logical_and(self.mutual_opposites_nbr, self.kpt_label == 0)
        self.centers_mutual_opposites_nbr = np.logical_and(self.mutual_opposites_nbr, self.kpt_label == 1)

        self.n_corners_mutual_opposites_nbr = np.sum(self.corners_mutual_opposites_nbr)
        self.n_centers_mutual_opposites_nbr = np.sum(self.centers_mutual_opposites_nbr)

        print(f"Number of corners with all mutual opposites neighbors: {self.n_corners_mutual_opposites_nbr}")
        print(f"Number of centers with all mutual opposite neighbors: {self.n_centers_mutual_opposites_nbr}")

    def calc_mutual_nbr(self):
        self.mutual_nbr = np.all(
            np.any(self.nbr['ind'][self.nbr['ind'], :] ==  # shape: (kpt's ind, nbr's ind, nbr's nbr's ind)
                   np.arange(self.n_corners + self.n_centers)[:, None, None],
                   axis=2), axis=1)

        # Optional printing
        self.corners_mutual = np.logical_and(self.mutual_nbr, self.kpt_label == 0)
        self.centers_mutual = np.logical_and(self.mutual_nbr, self.kpt_label == 1)

        self.n_corners_mutual = np.sum(self.corners_mutual)
        self.n_centers_mutual = np.sum(self.centers_mutual)

        print(f"Number of corners with all mutual neighbors: {self.n_corners_mutual}")
        print(f"Number of centers with all mutual neighbors: {self.n_centers_mutual}")

    def _condition_hetro_edges(self):
        self.condition_hetro_edges = np.logical_and(self.mutual_opposites_nbr, self.opposites_geometric_valid)

    def _condition_homo_edges(self):
        self.condition_homo_edges = np.logical_and(
            np.logical_and(self.circular_opposite_same, self.circular_axis_diag_same),
            self.ring_counting_profile_valid
        )

    def create_graphs(self):
        self._condition_hetro_edges()
        self._connect_with_nbr_opposite()
        self._condition_homo_edges()
        self._connect_with_nbr_same()

    # TODO: try to generate hetro-quad graph in networkx and connect ego with common neighbor's of first neighbors
    def _connect_with_nbr_same(self):
        condition = np.logical_and(self.condition_homo_edges, self.kpt_label == 1)

        e1 = np.repeat(np.arange(self.all_keypoints.shape[0])[condition], 4)
        e2 = np.reshape(self.same_nbr[condition, :], (-1))
        self.edges_same_axis = np.concatenate((e1[:, None], e2[:, None]), axis=-1).astype(int)

        all_same_nbr = np.concatenate((self.same_nbr_all_sorted_cyc,
                                       self.same_nbr_all_sorted_cyc[:, 0][:, None]), axis=1)
        # TODO: consider add according to angle condition - 2 edge between diagonal nodes should be almost parallel
        self.edges_same_diag = np.reshape(np.stack((all_same_nbr[condition, :-1],
                                                    all_same_nbr[condition, 1:]), axis=2), (-1, 2))
        self.quad_edges = np.concatenate((self.edges_same_diag, self.edges_same_axis), axis=0)
        self.quad_edges = np.unique(np.concatenate((self.quad_edges,
                                              np.flip(self.quad_edges, axis=-1)), axis=0), axis=0)
        self.quad_graph_create()

    def _connect_with_nbr_opposite(self):
        condition = self.condition_hetro_edges
        nbr_list = self.nearest_opposites

        e1 = np.repeat(np.arange(self.all_keypoints.shape[0])[condition], 4)
        e2 = np.reshape(nbr_list['ind'][condition, :], (-1))
        self.edges_opposite = np.concatenate((e1[:, None], e2[:, None]), axis=-1).astype(int)

    def calc_node_color(self):
        # get opposite nbr in cyclic order
        self.quad_nodes_opposite_nbr = self.nearest_opposites['ind'][self.quad_nodes, :]
        vec2nbr = self.all_keypoints[self.quad_nodes_opposite_nbr, :-1] - self.all_keypoints[self.quad_nodes, :-1][:, None, :]
        angles = np.arctan2(vec2nbr[..., 0], vec2nbr[..., 1])
        ordered_ind = np.argsort(angles, axis=1)
        self.quad_nodes_opposite_nbr_cyclic_order = np.take_along_axis(self.quad_nodes_opposite_nbr, ordered_ind, axis=1)

        # calc color using the cyclically oriented opposite_neighbors
        self.quad_nodes_opposite_nbr_loc = self.all_keypoints[self.quad_nodes_opposite_nbr_cyclic_order, :-1]
        contours = list(self.quad_nodes_opposite_nbr_loc.astype(int))
        self.quad_nodes_pix_location = np.round(self.quad_nodes_location).astype(int)
        cnt_colours = self.get_contours_color_efficiently(self.quad_nodes_pix_location, contours, self.color_labels_np)

        # TODO: for now very simple, no averaging, nothing
        color = self.color_labels_np[self.quad_nodes_pix_location[:, 0], self.quad_nodes_pix_location[:, 1]]
        for i in range(cnt_colours.shape[0]):
            cnt_colours[i] = np.unique(np.concatenate((cnt_colours[i], np.array([color[i]]))))
        return cnt_colours
        #return np.where(color != 0, color, cnt_colours)

    def get_contours_color_efficiently(self, center, contours, color_image):
        center = np.flip(center, axis=1)
        contours = [np.flip(c, axis=1) for c in contours]

        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        patches = [color_image[b[1]:b[1] + b[3], b[0]:b[0] + b[2]] for b in bounding_boxes]
        countours_in_patches = [contours[i] - bounding_boxes[i][:2] for i in range(len(patches))]
        centers_in_patches = [center[i] - bounding_boxes[i][:2] for i in range(len(patches))]

        roi = [cv2.fillPoly(np.zeros_like(patches[i]).astype((np.uint8)), [countours_in_patches[i]], color=[1, 1, 1]) for i in range(len(patches))]
        roi_color = [roi[i] * patches[i] for i in range(len(patches))]
        votes = [labels[labels != 0] for labels in roi_color] # consider only non zero entries: inside ROI & label != 0 (corresponding to background)
        #major_vote = np.array([np.argmax(np.bincount(v)) if len(v) > 0 else 0 for v in votes])
        possible_colors = np.array([np.unique(v) if len(v) > 0 else np.array([x for x in range(8)]) for v in votes], dtype=object) #TODO: is this a bug? why we get 0-7 labels and not 1-7?

        # if self.debug_path:
        #     contours_dir = os.path.join(self.debug_path, 'contours')
        #     os.makedirs(contours_dir, exist_ok=True)
        #
        #     rgb_map = np.array([
        #         [0, 0, 0],  # black
        #         [255, 0, 0],  # red
        #         [0, 255, 0],  # green
        #         [255, 255, 0],  # yellow
        #         [0, 0, 255],  # blue
        #         [255, 0, 255],  # magenta
        #         [0, 255, 255],  # cyan
        #         [255, 255, 255]  # white
        #     ])
        #     image = rgb_map[color_image]
        #     image = cv2.polylines(image.astype(np.uint8), contours, isClosed=True,
        #                           color=[125, 125, 125], thickness=1)
        #     cv2.imwrite(os.path.join(contours_dir, f'countour_full_image.png'), image)
        #
        #     for i in range(len(patches)):
        #         image = rgb_map[patches[i]]
        #         image = cv2.polylines(image.astype(np.uint8), [countours_in_patches[i]], isClosed=True,
        #                                 color=[125, 125, 125], thickness=1)
        #         image = cv2.circle(image, (centers_in_patches[i][0], centers_in_patches[i][1]), radius=0, color=[200, 200, 200], thickness=1)
        #         roi = cv2.fillPoly(np.zeros_like(image).astype((np.uint8)), [countours_in_patches[i]], color=[1, 1, 1])
        #         image = image * roi
        #         cv2.imwrite(os.path.join(contours_dir, f'countour_{i}.png'), image)


        return possible_colors


    def plot_edges(self, save_path_graph, save_path_color_detections):
        keypoints = np.copy(self.all_keypoints)
        keypoints[:, -1] = 0

        quad_edges = np.array(self.G_quad_cyc.edges)
        g1 = keypoints[quad_edges[:, 0], :]
        g2 = keypoints[quad_edges[:, 1], :]

        corners = keypoints[self.kpt_label == 0, :]
        centers = keypoints[self.kpt_label == 1, :]

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(corners, color='pink', point_size=3.,
                         render_points_as_spheres=True)
        plotter.add_mesh(centers, color='brown', point_size=3.,
                         render_points_as_spheres=True)

        rgb_map = np.array([
            [0, 0, 0],  # black
            [255, 0, 0],  # red
            [0, 255, 0],  # green
            [255, 255, 0],  # yellow
            [0, 0, 255],  # blue
            [255, 0, 255],  # magenta
            [0, 255, 255],  # cyan
            [255, 255, 255]  # white
        ])

        color_image = rgb_map[self.color_labels_np]
        color_image =np.rot90(color_image)
        cv2.imwrite(save_path_color_detections, color_image)

        plotter.add_arrows(cent=g1, direction=g2 - g1, color='white')

        plotter.camera_position = 'yx'
        plotter.camera.roll += 180
        image = pv.read(save_path_color_detections)
        plotter.add_mesh(image, rgb=True)
        plotter.show(screenshot=save_path_graph, window_size=[10000, 10000])
        plotter.close()

    def calc_strong_edges(self):
        sorted_edges = np.sort(self.edges, axis=1)
        (unique, counts) = np.unique(sorted_edges, axis=0, return_counts=True)
        self.strong_edges = unique[counts > 1, :]
        print("Hi!")

    def quad_graph_create(self):
        import networkx as nx
        G_quad = nx.Graph()
        G_quad.add_edges_from(self.quad_edges)
        G_quad.remove_edges_from(nx.selfloop_edges(G_quad))
        self.quad_nodes = np.array(G_quad.nodes)
        self.quad_nodes_location = self.all_keypoints[self.quad_nodes, :-1]
        self.quad_nodes_color = self.calc_node_color()

        print(f"Number of quad nodes: {self.quad_nodes.shape[0]}")
        print(f"Number of quad nodes with zero color: {np.sum(self.quad_nodes_color == 0)}")


        #nx.set_node_attributes(G_quad, dict(zip(self.quad_nodes, self.quad_nodes_color)), name='color')
        nx.set_node_attributes(G_quad, dict(zip(self.quad_nodes, self.quad_nodes_location)), name='location')
        # TODO: process the graph more, identify degree, crossing nodes, add missing nodes, etc...

        # TODO: construct planar graph - assume artifacts can exist
        # TODO: make more efficient
        self.G_quad_cyc = nx.PlanarEmbedding()
        for node in G_quad.nodes:
            nbr = list(G_quad.neighbors(node))
            vec2nbr = self.all_keypoints[nbr, :-1] - self.all_keypoints[node, :-1][None, :]
            angles = np.arctan2(vec2nbr[:, 0], vec2nbr[:, 1])
            ordered_ind = np.argsort(angles)
            for i in range(len(ordered_ind)):
                self.G_quad_cyc.add_half_edge_cw(node, nbr[ordered_ind[i]], None if i==0 else nbr[ordered_ind[i-1]])
        nx.set_node_attributes(self.G_quad_cyc, dict(zip(self.quad_nodes, self.quad_nodes_pix_location)), name='location')
        nx.set_node_attributes(self.G_quad_cyc, dict(zip(self.quad_nodes, self.quad_nodes_color)), name='color')
        # Node Attributes:
        # 1) color - RGB
        # 2) location - first pixel, second pixel in image (opencv format)

        print("Hi")

    def get_quad_cyclic_graph(self):
        self.calc_nearest_opposites()
        self.calc_opposites_criterions()
        self.create_graphs()
        return self.G_quad_cyc

if __name__ == '__main__':
    start = time.time()
    PATH = '/Users/oshrihalimi/Results/full_heatmap_inference/cropped_images_trained_512_128_outx4/'
    var = np.load(os.path.join(PATH, 'clustering.npy'), allow_pickle=True).item()
    image_path = os.path.join(PATH, 'resized_4_cropped-0190 copy.png')  # var['image']
    d = Detections(corners=var['centers_0'], centers=var['centers_1'], image_path=image_path, path=PATH)

    graph = d.get_quad_cyclic_graph()

    end = time.time()
    d.plot_edges('graph.png')

    print(f'Elapsed time [sec]: {end - start}')