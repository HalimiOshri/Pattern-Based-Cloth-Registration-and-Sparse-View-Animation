#TODO: This file is used to train the keypoint color detector in clean_color_classifier.py
import os.path

import numpy as np
import torch
import copy
from torch.utils.data import Dataset
import cv2
from imgaug import augmenters as iaa
from learning_algorithms.data_processing.text_files.util_func import get_file_as_np_int_array

class PatchKeypointDataset(Dataset):
    def __init__(self, path_dataset_file, debug_dir, num_random_keypoints):
        super(PatchKeypointDataset, self).__init__()

        # TODO: should be configurable from the constructor's interface
        self.num_random_keypoints = num_random_keypoints
        self.std_centers = 0
        self.patch_size = 64
        self.debug_path = debug_dir

        os.makedirs(self.debug_path, exist_ok=True)
        self.path_dataset_file = path_dataset_file
        with open(path_dataset_file, 'r') as f:
            lines = f.readlines()

        self.items = []
        for line in lines:
            line = line.split()
            dline = lines[0].split()
            try:
                item = {"filename": os.path.basename(line[0]),
                     "patch": self.get_patch(line[0]),
                     "centers": self.get_centers(line[1]),
                     "corners": self.get_corners(line[2])
                     }
                if (item["centers"].shape[0] + item["corners"].shape[0]) > 0:
                    self.items.append(item)

            except OSError as exception:
                print(exception)

        self.seq = iaa.Sequential([
            iaa.Resize((0.5, 2.0)),
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            iaa.Rot90((0, 3)),
            #iaa.Resize((0.5, 1.0)),
            iaa.CropToFixedSize(height=self.patch_size, width=self.patch_size),  # crop images from each side by 0 to 16px (randomly chosen)
            iaa.CenterPadToFixedSize(height=self.patch_size, width=self.patch_size),
        ])

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.getitem(item)
        if isinstance(item, slice):
            return [self.getitem(x) for x in range(len(self.items))[item]]

    def getitem(self, item):
        item = self.items[item]
        out = {}
        out["filename"] = item["filename"]
        keypoints = np.concatenate((np.pad(item["corners"], ((0, 0), (0, 1))), item["centers"]), axis=0)
        type_label = (1 + 1 * (keypoints[:, 1:2] != 0)).astype(int) # corner=1, center = 2,
        keypoints = np.concatenate((keypoints, type_label), axis=1)

        patch_aug, locations_aug = self.seq(images=item["patch"][None, ...], keypoints=keypoints[None, :, :2])
        # locations_aug are corresponding to the patch upsampled to the output resolution
        keypoints = np.concatenate((locations_aug[0, ...], keypoints[:, 2:4]), axis=1) # concat with labels
        keypoints_inside_idx = np.all(np.logical_and(keypoints[:, :2] >= 0, keypoints[:, :2] < self.patch_size), axis=1)
        keypoints = keypoints[keypoints_inside_idx, :]

        counter = 0
        while np.any(keypoints_inside_idx) == False:
            counter = counter + 1
            patch_aug, locations_aug = self.seq(images=item["patch"][None, ...], keypoints=keypoints[None, :, :2])
            keypoints = np.concatenate((locations_aug[0, ...], keypoints[:, 2:4]), axis=1)  # concat with labels
            keypoints_inside_idx = np.all(np.logical_and(keypoints[:, :2] >= 0, keypoints[:, :2] < self.patch_size), axis=1)
            keypoints = keypoints[keypoints_inside_idx, :]
            patch_name = item["filename"]
            if counter > 3:
                print(f"Didn't find augmentation for patch {patch_name}")
                # return a random patch
                return self.getitem(np.random.choice(len(self.items)))


        # pick a random fixed-size set of annotations
        random_indices = np.random.choice(np.arange(keypoints.shape[0]), size=self.num_random_keypoints, replace=True)
        random_keypoints = keypoints[random_indices, ...]

        # add random noise to the location
        locations = random_keypoints[:, :2] + self.std_centers * np.random.standard_normal(
            size=random_keypoints[:, :2].shape)

        locations[locations > self.patch_size - 1] = self.patch_size - 1
        locations[locations < 0] = 0
        locations = locations.astype(int)
        random_keypoints = np.concatenate((locations, random_keypoints[:, 2:]), axis=-1)

        out["patch"] = patch_aug[0].transpose(2, 0, 1).astype(np.float32).copy()
        out["keypoints"] = random_keypoints
        assert random_keypoints.shape[0] == self.num_random_keypoints

        # DEBUG
        os.makedirs(self.debug_path, exist_ok=True)
        debug_image = patch_aug[0]
        debug_image[out["keypoints"][:, 1], out["keypoints"][:, 0], :] = 255
        cv2.imwrite(os.path.join(self.debug_path, out["filename"]), debug_image)
#
        return out

    def __len__(self):
        return len(self.items)

    def get_patch(self, path_patch):
        image = cv2.imread(path_patch)
        return image

    def get_centers(self, path_centers):
        centers = get_file_as_np_int_array(path_centers)
        if centers.shape[0] > 0:
            return centers
        else:
            return np.empty((0, 3))

    def get_corners(self, path_corners):
        corners = get_file_as_np_int_array(path_corners)
        if corners.shape[0] > 0:
            return corners
        else:
            return np.empty((0, 3))

class TestDataset(Dataset):
    def __init__(self, path, cameras, frames):
        super(TestDataset, self).__init__()
        self.path = path
        self.cameras = cameras
        self.frames = frames

        self.num_cameras = len(self.cameras)
        self.num_frames = len(self.frames)
        self.len = self.num_cameras * self.num_frames

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.getitem(item)
        if isinstance(item, slice):
            return [self.getitem(x) for x in range(self.len)[item]]

    def getitem(self, item):
        camera_id = item % self.num_cameras
        frame_id = int((item - camera_id) / self.num_cameras)
        filename = os.path.join(self.path, f'{self.cameras[camera_id]}', f'image{self.frames[frame_id]:04d}.png')
        image = cv2.imread(filename)

        if image is None:
            image = np.zeros((3, 2668, 4096)).astype(np.float32)
            with open(self.bad_files_path, 'a+') as f:
                print(f'BAD FILE: {filename}')
                f.write(filename + '\n')
        else:
            image = torch.Tensor(np.transpose(image, (2, 0, 1)).astype(np.float32))

        return {"image": image, "cam": self.cameras[camera_id], "frame": self.frames[frame_id]}

    def __len__(self):
        return self.len

if __name__ == '__main__':
    dataset_file_path = '/mnt/home/oshrihalimi/Data/ColorPatternAnnotations/CornersAndCenters/train_list_short.txt'
    ds = PatchKeypointDataset(dataset_file_path)