from learning_algorithms.lightning_modules.clean_color_classifier import ColorClassifier
import torch
import os
from torch import jit
from learning_algorithms.data_processing.datasets.datasets_color import TestDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import cv2
import numpy as np
import torch

frames = [x for x in range(8000, 12002)]
camera_list_path = '/mnt/home/oshrihalimi/capture/camera_list.txt'
bad_files_path = '/mnt/home/oshrihalimi/capture/bad_files.txt'
images_path = '/mnt/captures/studies/pilots/sociopticon/Aug18/s--20210823--1323--0000000--pilot--patternCloth/undistorted/'
save_path = '/mnt/home/oshrihalimi/capture/color_detector_results/FINAL_clean_detector_sparse_cross_entropy_400_random_keypoints_with_resize_aug_0_v1'
model_path = '/mnt/home/oshrihalimi/trained_models/color_detector_results/FINAL_clean_detector_sparse_cross_entropy_400_random_keypoints_with_resize_aug_0.5_2.0/epoch=1599-step=1599.ckpt'

camera_inds_to_process = (0, 240)
if __name__ == '__main__':
    classifier = ColorClassifier.load_from_checkpoint(model_path)

    with open(camera_list_path, 'r') as f:
        cameras = f.read().splitlines()

    cameras = cameras[camera_inds_to_process[0] : min(camera_inds_to_process[1], len(cameras))]
    test_dataset = TestDataset(path=images_path, cameras=cameras, frames=frames)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)

    trainer = pl.Trainer(gpus=[0,1,2,3,4,5,6,7], default_root_dir=save_path)
    trainer.test(model=classifier, dataloaders=test_loader, ckpt_path=model_path, verbose=True, datamodule=None, test_dataloaders=None)
    print("Hi")