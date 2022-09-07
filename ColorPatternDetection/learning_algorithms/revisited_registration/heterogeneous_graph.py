from sklearn.neighbors import NearestNeighbors
import numpy as np

class HeterogeneousGraphGenerator:
    def __init__(self, keypoints):
        self.keypoints = keypoints
        self.centers = self.keypoints["centers"]
        self.corners = self.keypoints["corners"]

        self.n_corners = self.corners.shape[0]
        self.n_centers = self.centers.shape[0]

        # create a pool of all keypoints, shape: [#corners + #centers, 3] with labels 0-corner, 1-center
        self.all_keypoints = np.concatenate((self.corners, self.centers), axis=0)
        self.all_keypoints = np.concatenate((self.all_keypoints,
                                             np.zeros(shape=(self.n_corners + self.n_centers, 1))), axis=1)
        self.all_keypoints[self.n_corners:, -1] = 1
        self.kpt_label = self.all_keypoints[:, -1]
        self.n_kpt = self.all_keypoints.shape[0]

        print('\n')
        print(f"Initializing hetrogeneous graph generator with:")
        print(f"Number of detected corners: {self.n_corners}")
        print(f"Number of detected centers: {self.n_centers}")
        print('\n')

        # Generate graph
        self.calc_nearest_opposites()
        self.calc_mutual_opposite_nbr()
        self.calc_cycles()
        self.calc_final_graph()

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

        # get opposite nbr in cyclic order
        e1 = np.stack(4 * (np.arange(self.n_kpt), ), axis=1)
        e2 = self.nearest_opposites["ind"]
        vec2nbr = self.all_keypoints[e2, :-1] - self.all_keypoints[e1, :-1]
        angles = np.arctan2(vec2nbr[..., 0], vec2nbr[..., 1])
        ordered_ind = np.argsort(angles, axis=1)
        cyc_sorted_nearest_opposites_ind = np.take_along_axis(self.nearest_opposites["ind"], ordered_ind, axis=1)

        self.nearest_opposites["ind_cyc_sorted"] = cyc_sorted_nearest_opposites_ind

    def calc_mutual_opposite_nbr(self):
        self.is_mutual_opposites_nbr = np.any(self.nearest_opposites['ind_cyc_sorted'][self.nearest_opposites['ind_cyc_sorted'],
                   :] ==  # shape: (kpt's ind, nbr's ind, nbr's nbr's ind)
                   np.arange(self.n_kpt)[:, None, None],
                   axis=2)
        self.mutual_opposites_nbr = np.array([self.nearest_opposites['ind_cyc_sorted'][i, self.is_mutual_opposites_nbr[i, :]] for i in range(self.n_kpt)])

    def calc_cycles(self):
        n1 = np.array([[i] * self.mutual_opposites_nbr[i].shape[0] for i in range(self.mutual_opposites_nbr.shape[0])])
        n2 = self.mutual_opposites_nbr

        n1 = np.concatenate(tuple(n1), axis=0)
        n2 = np.concatenate(tuple(n2), axis=0)
        n_edges = n1.shape[0]

        # Positive cycles
        n3 = np.array([self.mutual_opposites_nbr[n2[i]][
                           (np.argwhere(self.mutual_opposites_nbr[n2[i]] == n1[i]).item() + 1) % self.mutual_opposites_nbr[n2[i]].shape[0]]
                       for i in range(n_edges)])
        n4 = np.array([self.mutual_opposites_nbr[n3[i]][
                           (np.argwhere(self.mutual_opposites_nbr[n3[i]] == n2[i]).item() + 1) % self.mutual_opposites_nbr[n3[i]].shape[0]]
                       for i in range(n_edges)])
        n5 = np.array([self.mutual_opposites_nbr[n4[i]][
                           (np.argwhere(self.mutual_opposites_nbr[n4[i]] == n3[i]).item() + 1) % self.mutual_opposites_nbr[n4[i]].shape[0]]
                       for i in range(n_edges)])
        n6 = np.array([self.mutual_opposites_nbr[n5[i]][
                           (np.argwhere(self.mutual_opposites_nbr[n5[i]] == n4[i]).item() + 1) % self.mutual_opposites_nbr[n5[i]].shape[0]]
                       for i in range(n_edges)])


        is_valid_cycle = np.logical_and(np.logical_and(n1 == n5, n2 == n6),
                                        np.array([np.unique(x).shape[0] for x in np.stack((n1, n2, n3, n4), axis=1)]) == 4)
        self.valid_cycles_p = np.stack((n1[is_valid_cycle], n2[is_valid_cycle], n3[is_valid_cycle], n4[is_valid_cycle]), axis=1)

        # Negative cycles
        n3 = np.array([self.mutual_opposites_nbr[n2[i]][
                           (np.argwhere(self.mutual_opposites_nbr[n2[i]] == n1[i]).item() - 1) % self.mutual_opposites_nbr[n2[i]].shape[0]]
                       for i in range(n_edges)])
        n4 = np.array([self.mutual_opposites_nbr[n3[i]][
                           (np.argwhere(self.mutual_opposites_nbr[n3[i]] == n2[i]).item() - 1) % self.mutual_opposites_nbr[n3[i]].shape[0]]
                       for i in range(n_edges)])
        n5 = np.array([self.mutual_opposites_nbr[n4[i]][
                           (np.argwhere(self.mutual_opposites_nbr[n4[i]] == n3[i]).item() - 1) % self.mutual_opposites_nbr[n4[i]].shape[0]]
                       for i in range(n_edges)])
        n6 = np.array([self.mutual_opposites_nbr[n5[i]][
                           (np.argwhere(self.mutual_opposites_nbr[n5[i]] == n4[i]).item() - 1) % self.mutual_opposites_nbr[n5[i]].shape[0]]
                       for i in range(n_edges)])


        is_valid_cycle = np.logical_and(np.logical_and(n1 == n5, n2 == n6),
                                        np.array([np.unique(x).shape[0] for x in np.stack((n1, n2, n3, n4), axis=1)]) == 4)
        self.valid_cycles_n = np.stack((n1[is_valid_cycle], n2[is_valid_cycle], n3[is_valid_cycle], n4[is_valid_cycle]), axis=1)

        # all cycles
        self.topo_valid_cycles = np.concatenate((self.valid_cycles_p, self.valid_cycles_n), axis=0)
        self.geo_topo_valid_cycles = self.geo_validate_cycle(self.topo_valid_cycles)


    def geo_validate_cycle(self, cycles):
        v1 = self.all_keypoints[cycles[:, 1].astype(int), :2] - self.all_keypoints[cycles[:, 0].astype(int), :2]
        v2 = self.all_keypoints[cycles[:, 2].astype(int), :2] - self.all_keypoints[cycles[:, 3].astype(int), :2]
        u1 = self.all_keypoints[cycles[:, 3].astype(int), :2] - self.all_keypoints[cycles[:, 0].astype(int), :2]
        u2 = self.all_keypoints[cycles[:, 2].astype(int), :2] - self.all_keypoints[cycles[:, 1].astype(int), :2]
        cos_v = np.sum(v1 * v2, axis=1) / np.linalg.norm(v1, axis=1) / np.linalg.norm(v2, axis=1)
        cos_u = np.sum(u1 * u2, axis=1) / np.linalg.norm(u1, axis=1) / np.linalg.norm(u2, axis=1)
        is_parallelogram = np.logical_and(cos_v > 0.9, cos_u > 0.9)
        return cycles[is_parallelogram, :]

    def get_intermediate_variables(self):
        '''

        :return: initial nodes (pixel coordinates & label: 0-corner, 1-center, nearest opposites indices)
        '''
        return self.all_keypoints, self.nearest_opposites["ind"], self.mutual_opposites_nbr, self.edges_in_valid_cycle

    def calc_final_graph(self):
        edges = np.concatenate((self.geo_topo_valid_cycles[:, (0, 1)],
                                              self.geo_topo_valid_cycles[:, (1, 2)],
                                              self.geo_topo_valid_cycles[:, (2, 3)],
                                              self.geo_topo_valid_cycles[:, (3, 0)]), axis=0)
        edges = np.concatenate((edges, np.flip(edges, axis=1)))
        self.final_edges = np.unique(edges, axis=0) # each edge is listed exactly twice in two directions

        self.heterogeneous_graph = {'nbr': np.array([self.final_edges[self.final_edges[:, 0]==i, 1] for i in range(self.n_kpt)]),
                                    'type': self.kpt_label,
                                    'pixel_location': self.all_keypoints[:, :2]} # label: 0-corner, 1-center
        homogeneous_edges = np.array([x[self.kpt_label[x.astype(int)].astype(bool)] for x in self.geo_topo_valid_cycles])
        return self.heterogeneous_graph, homogeneous_edges

