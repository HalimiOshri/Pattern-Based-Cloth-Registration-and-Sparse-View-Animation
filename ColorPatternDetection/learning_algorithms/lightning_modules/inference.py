from learning_algorithms.lightning_modules.clean_color_classifier import ColorClassifier
import torch
import os
from torch import jit
from learning_algorithms.data_processing.datasets.patch_keypoints_dataset import TestDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import cv2
import numpy as np
import torch

frames = [503, 1138, 898, 1941, 2885, 3324, 3702, 4460, 4720, 5418, 6792, 9152, 9393, 9477, 10273, 11552]
cameras = [401479, 400874, 400876, 401534, 400897, 400160, 400143, 400221]
images_path = '/mnt/home/oshrihalimi/Data/ColorPattern/'
save_path = '/mnt/home/oshrihalimi/Results/color_pattern_detector_inference/clean_detector_cross_entropy/'

model_path = '/mnt/home/oshrihalimi/color_pattern_detection/clean_detector_cross_entropy/lightning_logs/version_2/checkpoints/epoch=99-step=99.ckpt'

if __name__ == '__main__':
    classifier = ColorClassifier.load_from_checkpoint(model_path)

    test_dataset = TestDataset(path=images_path, cameras=cameras, frames=frames)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=3, pin_memory=True)

    trainer = pl.Trainer(gpus=[0,1,2,3,4,5,6,7], default_root_dir=save_path)
    trainer.test(model=classifier, dataloaders=test_loader, ckpt_path=model_path, verbose=True, datamodule=None, test_dataloaders=None)
    print("Hi")