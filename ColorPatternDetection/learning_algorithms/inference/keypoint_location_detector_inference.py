from learning_algorithms.lightning_modules.color_corners_and_centers_detector import ColorClassifier
import torch
import os
from torch import jit
from learning_algorithms.data_processing.datasets.patch_keypoints_dataset import TestDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import cv2
import numpy as np
import torch

frames = [x for x in range(90000, 90003)] #TODO:

camera_list_path = '/mnt/home/oshrihalimi/capture/camera_list.txt'
bad_files_path = '/mnt/home/oshrihalimi/capture/bad_files.txt'
images_path = '/mnt/captures/studies/pilots/sociopticon/Aug18/s--20210823--1323--0000000--pilot--patternCloth/undistorted/'
save_path = '/mnt/home/oshrihalimi/capture/keypoint_location_detector_results/corners_centers_detector_unet-5_double_conv_4_conv_layers_cross_entropy_1_sparse_and_1_dense_input_gray_aug_v1'
model_path = '/mnt/home/oshrihalimi/trained_models/keypoint_location_detector/corners_centers_detector_unet-5_double_conv_4_conv_layers_cross_entropy_1_sparse_and_1_dense_input_gray_aug_v1/epoch=1998-step=1998.ckpt'

camera_inds_to_process = (0, 240)
if __name__ == '__main__':
    classifier = ColorClassifier.load_from_checkpoint(model_path)

    with open(camera_list_path, 'r') as f:
        cameras = f.read().splitlines()

    cameras = cameras[camera_inds_to_process[0] : min(camera_inds_to_process[1], len(cameras))]
    test_dataset = TestDataset(path=images_path, cameras=cameras, frames=frames, bad_files_path=bad_files_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=3, pin_memory=True)

    trainer = pl.Trainer(gpus=[0,1,2,3,4,5,6,7], default_root_dir=save_path)
    while True: #TODO:
        trainer.test(model=classifier, dataloaders=test_loader, ckpt_path=model_path, verbose=True, datamodule=None, test_dataloaders=None)
    print("Hi")