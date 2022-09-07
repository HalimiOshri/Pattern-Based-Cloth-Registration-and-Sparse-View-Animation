import cv2
import os
from learning_algorithms.process_segmentations.detections import Detections
import time
import numpy as np
import networkx as nx
from multiprocessing import Pool, Value, Array, Process
import itertools

# TODO: 1) register not only patch center but all the points the patch contains - Done.
#       2) Parallelize - not so simple. consider to replace networkx with a different library.
#       3) Better quad graph to start with - with existing detections it seems we are close to optimal graph.
#          Connecting with common nbr's nbr's from the hetrogeneous graph didn't improve much
#       4) Non unique hash distinguishing (very optional - the new code is goind to be unique)

class Registration:
    def __init__(self, quad_cyclic_graph, image_path, table, color_board, save_path, save_path_image):
        self.min_cc_size = 9
        self.quad = quad_cyclic_graph
        self.image = cv2.imread(image_path)
        self.table = table
        self.color_board = color_board
        self.save_path = save_path
        self.save_path_image = save_path_image

        self.cc = []
        for c in list(nx.connected_components(self.quad)):
            if len(c) >= self.min_cc_size:
                self.cc.append(c)

        self.grid_image = None
        self.registration_image = None

    def register_nodes(self, save_image=False):
        '''
        Args:
            node: node id
            grid_radius:
        Returns:
            grid of nodes if regular grid found around the node.
            grid is indexed by (x,y) offsets w.r.t central node.
            grid's x direction is oriented with node's first_nbr
        '''

        count_valid_5x5 = 0
        count_registered = 0
        count_uniquely_registered = 0
        count_hash_with_zero = 0
        for node in self.quad.nodes:
            res = self.get_5x5_grid(node)
            if res:
                grid_5x5, first_nbr = res[0], res[1]
            else:
                grid_5x5, first_nbr = None, None
            hash = self.get_hash(grid_5x5) if grid_5x5 else None
            registration = [self.table[h][0] for h in hash if h in self.table.keys()] if hash is not None else None
            board_location = [r[0] for r in registration] if registration is not None else None
            board_alignment = [r[1] for r in registration] if registration is not None else None

            nx.set_node_attributes(self.quad,
                                   {node:
                                        {"grid_5x5": grid_5x5,
                                         "grid_5x5_first_nbr": first_nbr,
                                         "hash": hash,
                                         "registration": registration,
                                         "board_location": board_location,
                                         "board_alignment": board_alignment
                                         }
                                    })

            if grid_5x5:
                count_valid_5x5 = count_valid_5x5 + 1
            if registration:
                count_registered = count_registered + 1

        print(f"Number of nodes with 5x5 valid patch: {count_valid_5x5}")
        print(f"Number of registered nodes: {count_registered}")

        self.expand_registration()
        self.vote_registration()
        print(f"Number of FINAL registered nodes: {len(self.determines_board_location)}")
        self.save_registration()

        if save_image:
            self.plot_registration_image()
            cv2.imwrite(self.save_path_image, self.registration_image)

        return self.get_registration()

    def save_registration(self):
        image_location = nx.get_node_attributes(self.quad, 'location')
        data = {"board_location": self.determines_board_location,
                "image_location": image_location}
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        np.save(self.save_path, data, allow_pickle=True)

    def get_registration(self):
        image_location = nx.get_node_attributes(self.quad, 'location')
        data = {"node_id": np.array(self.quad.nodes),
                "board_location": self.determines_board_location,
                "image_location": image_location}
        return data

    def expand_registration(self):
        for node in self.quad.nodes(data=True):
            if node[1]['registration'] is not None:
                grid_5x5 = node[1]['grid_5x5']
                alignment = node[1]['board_alignment']
                location = node[1]['board_location']

                num_registrations = len(location)
                for r in range(num_registrations):
                    grid_5x5_rot = np.rot90(grid_5x5.slice(-1, 1, -1, 1), k=-alignment[r])
                    grid_5x5_rot_flat = np.reshape(grid_5x5_rot, (-1))
                    dy, dx = np.meshgrid(range(-1, 2), range(-1, 2))
                    nbr_location = np.stack((location[r][0] + dx, location[r][1] + dy), axis=2)
                    nbr_location_flat = np.reshape(nbr_location, (-1, 2))

                    for i, node_id in enumerate(grid_5x5_rot_flat):
                        self.quad.nodes[node_id].setdefault('board_location_from_nbr', []).append(nbr_location_flat[i, :])

    def vote_registration(self):
        board_location_from_nbr = nx.get_node_attributes(self.quad, 'board_location_from_nbr')
        board_location_from_nbr = {k: np.unique(np.stack(v, axis=0), axis=0, return_counts=True)
                                           for k, v in board_location_from_nbr.items()}

        # Take the major vote, unless the vote is [1,1,...] -
        # in this case the location is dismissed
        board_location_from_nbr = {k: v[0][np.argmax(v[1]), :] for k, v in
                                           board_location_from_nbr.items()
                                           if not (np.max(v[1]) == 1 and len(v[1]) > 1)} #for small pattern we dismiss this case [1]
        self.determines_board_location = board_location_from_nbr

    def get_hash(self, grid):
        # TODO: consider code with more information
        ind = grid.slice(-1, 1, -1, 1)
        colors = np.array([self.quad.nodes[x]['color'] for x in np.reshape(ind, (-1))], dtype=object)

        all_combinations = [x for x in itertools.product(*[y for y in colors])]

        strings = [[str(integer) for integer in sequence] for sequence in all_combinations]
        a_string = ["".join(string) for string in strings]
        an_integer = [int(a) for a in a_string]
        hash = an_integer
        return hash

    def get_5x5_grid(self, node):
        grid_radius = 1
        grid = Grid(grid_radius)
        first_nbr = Grid(grid_radius)
        grid[0, 0] = node
        first_nbr[0, 0] = 0

        # place first nbr
        if len(self.quad.adj[grid[0, 0]]) != 4:
            return None

        nbr_cw = []
        for nbr in self.quad.neighbors_cw_order(node):
            nbr_cw.append(nbr)

        grid[1, 0] = nbr_cw[0]
        grid[0, -1] = nbr_cw[1]
        grid[-1, 0] = nbr_cw[2]
        grid[0, 1] = nbr_cw[3]

        # get nbr of direct nbr
        nbr_10 = list(self.quad.neighbors_cw_order(grid[1, 0]))
        nbr_0m1 = list(self.quad.neighbors_cw_order(grid[0, -1]))
        nbr_m10 = list(self.quad.neighbors_cw_order(grid[-1, 0]))
        nbr_01 = list(self.quad.neighbors_cw_order(grid[0, 1]))

        if not (len(nbr_10) >= 3 and len(nbr_0m1) >= 3 and len(nbr_m10) >= 3 and len(nbr_01) >= 3):
            return None
        # if not (4 >= len(nbr_10) >= 3 and 4 >= len(nbr_0m1) >= 3 and 4 >= len(nbr_m10) >= 3 and 4 >= len(nbr_01) >= 3):
        #     return None

        # get first corners - no need to orient them
        cnbr = list(nx.common_neighbors(self.quad, grid[1, 0], grid[0, 1]))
        cnbr.remove(node)
        if len(cnbr) == 1 and len(self.quad.adj[cnbr[0]]) >= 2:
            grid[1, 1] = cnbr[0]
        else:
            return None

        cnbr = list(nx.common_neighbors(self.quad, grid[0, 1], grid[-1, 0]))
        cnbr.remove(node)
        if len(cnbr) == 1 and len(self.quad.adj[cnbr[0]]) >= 2:
            grid[-1, 1] = cnbr[0]
        else:
            return None

        cnbr = list(nx.common_neighbors(self.quad, grid[-1, 0], grid[0, -1]))
        cnbr.remove(node)
        if len(cnbr) == 1 and len(self.quad.adj[cnbr[0]]) >= 2:
            grid[-1, -1] = cnbr[0]
        else:
            return None

        cnbr = list(nx.common_neighbors(self.quad, grid[0, -1], grid[1, 0]))
        cnbr.remove(node)
        if len(cnbr) == 1 and len(self.quad.adj[cnbr[0]]) >= 2:
            grid[1, -1] = cnbr[0]
        else:
            return None

        # If we arrive here, we should have 3x3 inner block - where the central node and its direct nbrs are oriented
        # On the grid boundary we don't check 4-degree (it might lay on the graph boundary)
        # We also, don't orient the corners nodes - no need

        # more validations: check all edges exist, all nodes are unique etc.
        for i in [-1, 0, 1]:
            for j in [-1, 0]:
                if not self.quad.has_edge(grid[i, j], grid[i, j + 1]):
                    return None

        for j in [-1, 0, 1]:
            for i in [-1, 0]:
                if not self.quad.has_edge(grid[i, j], grid[i + 1, j]):
                    return None

        return (grid, first_nbr)

    def plot_registration_image(self):
        img = self.image

        image_location = nx.get_node_attributes(self.quad, 'location')
        board_location = self.determines_board_location
        for k, board_coords in board_location.items():
            coords = tuple(image_location[k])
            radius = 1
            thickness = -1
            color = self.color_board[int(board_coords[0]), int(board_coords[1])].tolist()
            img = cv2.circle(img, (coords[1], coords[0]), radius, color, thickness)

        self.registration_image = img

    def plot_grid(self, grid):
        if self.grid_image is None:
            img = cv2.imread(self.image_path)
        else:
            img = self.grid_image
        thickness = 3
        for i in [-2, -1, 0, 1, 2]:
            for j in [-2, -1, 0, 1]:
                color = [0, 0, 0]
                c = 128 + 30 * j
                if i == -2:
                    color[0] = c
                if i == -1:
                    color[0] = c
                    color[1] = c
                if i == 0:
                    color[1] = c
                if i == 1:
                    color[1] = c
                    color[2] = c
                if i == 2:
                    color[2] = c
                # color = [255, 255, 255]
                coords = tuple(self.quad.nodes[grid[i, j]]['location'])
                coords_ = tuple(self.quad.nodes[grid[i, j + 1]]['location'])
                self.grid_image = cv2.line(img, (coords[1], coords[0]), (coords_[1], coords_[0]), color, thickness)
        for i in [-2, -1, 0, 1]:
            for j in [-2, -1, 0, 1, 2]:
                color = [0, 0, 0]
                c = 128 + 30 * j
                if i == -2:
                    color[0] = c
                if i == -1:
                    color[0] = c
                    color[1] = c
                if i == 0:
                    color[1] = c
                if i == 1:
                    color[1] = c
                    color[2] = c
                if i == 2:
                    color[2] = c
                # color = [255, 255, 255]
                coords = tuple(self.quad.nodes[grid[i, j]]['location'])
                coords_ = tuple(self.quad.nodes[grid[i + 1, j]]['location'])
                self.grid_image = cv2.line(img, (coords[1], coords[0]), (coords_[1], coords_[0]), color, thickness)

    def show_nbr_cyclic_order(self):
        img = cv2.imread(self.image_path)

        id = 41571
        node = self.quad.nodes[id]
        center_coordinates = tuple(node['location'])
        radius = 12
        color = (255, 255, 255)
        thickness = -1
        img = cv2.circle(img, (center_coordinates[1], center_coordinates[0]), radius, color, thickness)

        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)]
        radius = 12
        i = 0
        for nbr in list(self.quad.neighbors_cw_order(id)):
            center_coordinates = tuple(self.quad.nodes[nbr]['location'])
            thickness = -1
            img = cv2.circle(img, (center_coordinates[1], center_coordinates[0]), radius, colors[i], thickness)
            i = i + 1

        node['first_nbr'] = list(self.quad.neighbors_cw_order(id))[-1]
        radius = 8
        i = 0
        for nbr in list(self.quad.neighbors_cw_order(id)):
            center_coordinates = tuple(self.quad.nodes[nbr]['location'])
            thickness = -1
            img = cv2.circle(img, (center_coordinates[1], center_coordinates[0]), radius, colors[i], thickness)
            i = i + 1

        node['first_nbr'] = list(self.quad.neighbors_cw_order(id))[-1]
        radius = 6
        i = 0
        for nbr in list(self.quad.neighbors_cw_order(id)):
            center_coordinates = tuple(self.quad.nodes[nbr]['location'])
            thickness = -1
            img = cv2.circle(img, (center_coordinates[1], center_coordinates[0]), radius, colors[i], thickness)
            i = i + 1

        node['first_nbr'] = list(self.quad.neighbors_cw_order(id))[-1]
        radius = 4
        i = 0
        for nbr in list(self.quad.neighbors_cw_order(id)):
            center_coordinates = tuple(self.quad.nodes[nbr]['location'])
            thickness = -1
            img = cv2.circle(img, (center_coordinates[1], center_coordinates[0]), radius, colors[i], thickness)
            i = i + 1

        cv2.imwrite('show_nbr_cyclic_order.png', img)


class Grid:
    def __init__(self, grid_radius):
        self.grid_radius = grid_radius
        self.grid = np.zeros((2 * grid_radius + 1, 2 * grid_radius + 1))

    def __getitem__(self, index):
        return self.grid[index[0] + self.grid_radius, index[1] + self.grid_radius]

    def __setitem__(self, index, val):
        self.grid[index[0] + self.grid_radius, index[1] + self.grid_radius] = val

    def slice(self, x_0, x_1, y_0, y_1):
        return self.grid[x_0 + self.grid_radius:x_1 + self.grid_radius + 1,
               y_0 + self.grid_radius:y_1 + self.grid_radius + 1]


if __name__ == '__main__':
    clusters_path = '/Users/oshrihalimi/Results/frame_sequence_registration/inference/trained_512_128_outx4_frame_sequence_subset_cropped/clusters/'
    images_path = '/Users/oshrihalimi/Data/FrameSequenceSubsetCroppedUpsampled_4/'
    save_path = '/Users/oshrihalimi/Results/frame_sequence_registration/inference/trained_512_128_outx4_frame_sequence_subset_cropped/registrations/'
    data = np.load('hash_table.npy', allow_pickle=True)
    table = data.item()['table']
    color_board = data.item()['color']

    clusters = os.listdir(clusters_path)
    for cluster in clusters:
        start = time.time()

        var = np.load(os.path.join(clusters_path, cluster), allow_pickle=True).item()
        img_name = cluster.split('.')[0]
        image_path = os.path.join(images_path, img_name + '.png')  # var['image']
        save_path_registration = os.path.join(save_path, img_name + '.npy')
        d = Detections(corners=var['centers_0'], centers=var['centers_1'], image_path=image_path)
        graph = d.get_quad_cyclic_graph()
        r = Registration(quad_cyclic_graph=graph, image_path=None,
                         table=table, color_board=color_board, save_path=save_path_registration,
                         save_path_image=os.path.join(save_path, img_name + '.png'))
        r.register_nodes(save_image=True)

        end = time.time()
        print(f'Elapsed time [sec]: {end - start}')