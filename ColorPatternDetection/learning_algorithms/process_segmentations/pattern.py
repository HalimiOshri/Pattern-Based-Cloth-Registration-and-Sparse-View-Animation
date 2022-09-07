import numpy as np
import cv2

class Pattern:
    def __init__(self, path, patch_size):
        self.path = path
        self.pattern = cv2.imread(self.path)
        self.color = self.pattern # for visualization we save the BGR order convention in opencv
        self.pattern = np.flip(self.pattern, axis=2) # flipping to get the colors in RGB order
        self.pattern = self.convert_to_digits(self.pattern)
        self.patch_size = patch_size
        self.build_hash_table()


    def build_hash_table(self):
        '''
        Returns:
            Possible locations and alignments on the pattern board
            Hash code is distinctive if it leads to a single configuration of location & alignment
        '''
        self.hash_table = {}

        for x in range(self.pattern.shape[0] - self.patch_size):
            for y in range(self.pattern.shape[1] - self.patch_size):
                patch = self.pattern[x:x+self.patch_size, y:y+self.patch_size]
                center = (x + np.floor(self.patch_size/2), y + np.floor(self.patch_size/2))
                for r in range(4):
                    patch_rot = np.rot90(patch, k=r)
                    hash = self.hash_function(patch_rot)

                    if not hash in self.hash_table.keys():
                        self.hash_table[hash] = [(center, r)]
                    else:
                        self.hash_table[hash].append((center, r))

    def convert_to_digits(self, pattern):
        colors = np.unique(pattern.reshape(-1, 3), axis=0)

        labels_image = np.zeros(pattern.shape[:2])

        idx = np.where(np.all((pattern - np.array([255, 50, 50])[None, None, :]) == 0, axis=2)) #red
        labels_image[idx[0], idx[1]] = 1

        idx = np.where(np.all((pattern - np.array([50, 255, 50])[None, None, :]) == 0, axis=2))  # green
        labels_image[idx[0], idx[1]] = 2

        idx = np.where(np.all((pattern - np.array([255, 255, 50])[None, None, :]) == 0, axis=2))  # yellow
        labels_image[idx[0], idx[1]] = 3

        idx = np.where(np.all((pattern - np.array([50, 50, 255])[None, None, :]) == 0, axis=2))  # blue
        labels_image[idx[0], idx[1]] = 4

        idx = np.where(np.all((pattern - np.array([255, 50, 255])[None, None, :]) == 0, axis=2))  # magenta
        labels_image[idx[0], idx[1]] = 5

        idx = np.where(np.all((pattern - np.array([50, 255, 255])[None, None, :]) == 0, axis=2))  # cyan
        labels_image[idx[0], idx[1]] = 6

        idx = np.where(np.all((pattern - np.array([255, 255, 255])[None, None, :]) == 0, axis=2))  # white
        labels_image[idx[0], idx[1]] = 7

        return labels_image

    def hash_function(self, patch):
        labels = patch.astype(int)
        strings = [str(integer) for integer in labels.reshape(-1)]
        a_string = "".join(strings)
        an_integer = int(a_string)
        hash = an_integer
        return hash

if __name__ == '__main__':
    path = '/mnt/home/oshrihalimi/capture/color_pattern_files/200-600/board.png'
    pattern = Pattern(path, 3)
    table = pattern.hash_table
    data = {"table": table, "color": pattern.color}
    np.save('/mnt/home/oshrihalimi/pycharm/ColorPatternDetection/color_pattern_files/hash_table_color_big_pattern.npy', data, allow_pickle=True)
    print("Hi")