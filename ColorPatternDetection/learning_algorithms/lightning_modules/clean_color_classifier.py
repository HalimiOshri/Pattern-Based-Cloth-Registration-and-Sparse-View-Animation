import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from learning_algorithms.data_processing.datasets.datasets_color import PatchKeypointDataset
import os
import cv2
import torch.nn as nn
import time
import kornia
from learning_algorithms.data_processing.datasets.datasets_color import TestDataset
from torchsummary import summary
import sys
sys.path.append('/mnt/home/oshrihalimi/KinematicAlignment/utils/')
import tensorIO

class SegmnetationNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(SegmnetationNet, self).__init__()

        self.num_classes = num_classes

        self.down1 = nn.Sequential(
            *self.make_conv_bn_relu(3, 64, kernel_size=5, stride=1, padding=2),
            *self.make_conv_bn_relu(64, 64, kernel_size=5, stride=1, padding=2),
            *self.make_conv_bn_relu(64, 64, kernel_size=5, stride=1, padding=2),
            *self.make_conv_bn_relu(64, 64, kernel_size=5, stride=1, padding=2),
            *self.make_conv_bn_relu(64, 64, kernel_size=5, stride=1, padding=2),
            *self.make_conv_bn_relu(64, 64, kernel_size=5, stride=1, padding=2))

        self.up4 = nn.Sequential(
            *self.make_conv_bn_relu(128, 64, kernel_size=5, stride=1, padding=2),
            *self.make_conv_bn_relu(64, 64, kernel_size=5, stride=1, padding=2),
            *self.make_conv_bn_relu(64, 64, kernel_size=5, stride=1, padding=2),
            *self.make_conv_bn_relu(64, 64, kernel_size=5, stride=1, padding=2),
            *self.make_conv_bn_relu(64, 64, kernel_size=5, stride=1, padding=2),
            *self.make_conv_bn_relu(64, 64, kernel_size=5, stride=1, padding=2)
        )
        self.final_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1, padding=0)

    def make_conv_bn_relu(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                      padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

    def forward(self, images):
        images = images / 255
        down1 = self.down1(images)
        out1 = F.max_pool2d(down1, kernel_size=2, stride=2)
        # upsample out_last, concatenate with down1 and apply conv operations
        out = F.upsample(out1, scale_factor=float(2), mode='bilinear')
        out = torch.cat([down1, out], 1)
        out = self.up4(out)

        # final 1x1 conv for predictions
        labels_prediction = self.final_conv(out)

        return labels_prediction

 ###################################

class ColorClassifier(pl.LightningModule):
    def __init__(self):
        super(ColorClassifier, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.num_classes = 8
        self.net = SegmnetationNet(num_classes=self.num_classes)

        self.rgb_map = np.array([
            [0, 0, 0],  # black
            [255, 0, 0],  # red
            [0, 255, 0],  # green
            [255, 255, 0],  # yellow
            [0, 0, 255],  # blue
            [255, 0, 255],  # magenta
            [0, 255, 255],  # cyan
            [255, 255, 255]  # white
        ])

    def to_rgb(self, label_image):
        label_image_np = label_image.detach().cpu().numpy()
        rgb = self.rgb_map[label_image_np]
        return rgb

    def validation_step(self, batch, batch_idx):
        save_dir = os.path.join(self.trainer.log_dir, 'validation', f'epoch-{self.current_epoch}')
        os.makedirs(save_dir, exist_ok=True)

        start_time = time.time()
        image = batch["image"]
        labels_prediction = self.forward(image)
        labels_prediction = torch.argmax(labels_prediction, dim=1)
        saved_image = self.to_rgb(labels_prediction)
        print("--- %s seconds ---" % (time.time() - start_time))
        for i in range(saved_image.shape[0]):
            cv2.imwrite(os.path.join(save_dir, f'{batch["cam"][i]}_{batch["frame"][i]}.png'), np.flip(saved_image[i], axis=-1).astype(np.uint8)) # opencv assumes BGR convention

        return labels_prediction

    def test_step(self, batch, batch_idx):
        start_time = time.time()
        image = batch["image"]
        labels_prediction = self.forward(image)
        labels_prediction = torch.argmax(labels_prediction, dim=1).to(dtype=torch.int)

        for i in range(labels_prediction.shape[0]):
            save_dir = os.path.join(self.trainer.default_root_dir, f'{batch["cam"][i]}')
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f'{batch["frame"][i]}.bt')
            tensorIO.WriteTensorToBinaryFile(labels_prediction[i].cpu(), save_path)

        print("--- %s seconds ---" % (time.time() - start_time))

        return None

    def forward(self, images):
        return self.net(images)

    def training_step(self, batch, batch_idx):
        image = batch["patch"]
        locations = batch["keypoints"][..., :2] # B x #samples x 2
        print(locations.shape)
        batch_idx = torch.arange(locations.shape[0])[:, None, None].expand((locations.shape[0:2] + (1,))).to(device=locations.device) # B x #samples x 1
        sample_idx = torch.cat((batch_idx, locations), dim=2).view(-1, 3).long() # (B * #samples) x 1
        labels = batch["keypoints"][..., 2]
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes) # currently zero based - 8 classes (including corners) # the gt labels are 1 based indices # B x #samples x 7

        labels_prediction = self.forward(image)
        labels_prediction = torch.softmax(labels_prediction, 1)

        labels_prediction_sampled = labels_prediction[sample_idx[:, 0], :, sample_idx[:, 2], sample_idx[:, 1]] # the locations coordintes should be flipped!
        #labels_prediction_sampled = torch.softmax(labels_prediction_sampled, -1)
        loss = F.cross_entropy(labels_prediction_sampled,
                         labels_one_hot.view(-1, self.num_classes).to(dtype=torch.float),
                         weight=torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1]).to(device=labels_prediction_sampled.device)) # inputs & target shape: (B * #samples) x 7

        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.9,
                                                               patience=100,
                                                               min_lr=1e-8,
                                                               verbose=True), "monitor": "loss"}
        return [optimizer], [scheduler]

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["loss"])


if __name__ == '__main__':
    dataset_file_path = '/mnt/home/oshrihalimi/Data/ColorPatternAnnotations/CornersAndCenters/train_list_short.txt'
    save_path = '/mnt/home/oshrihalimi/color_pattern_detection/FINAL_clean_detector_sparse_cross_entropy_400_random_keypoints_with_resize_aug_0.5_2.0'

    frames = [503, 1138, 898, 1941, 2885, 3324, 3702, 4460, 4720, 5418, 6792, 9152, 9393, 9477, 10273, 11552]
    cameras = [401479, 400874, 400876, 401534, 400897, 400160, 400143, 400221]
    test_images_path = '/mnt/captures/studies/pilots/sociopticon/Aug18/s--20210823--1323--0000000--pilot--patternCloth/undistorted/'
    debug_dir = os.path.join(save_path, 'debug')

    classifier = ColorClassifier()
    dataset = PatchKeypointDataset(path_dataset_file=dataset_file_path, debug_dir=debug_dir, num_random_keypoints=400)
    train_loader = DataLoader(dataset, batch_size=300, shuffle=True)

    test_dataset = TestDataset(path=test_images_path, cameras=cameras, frames=frames)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    trainer = pl.Trainer(gpus=[0,1,2,3,4,5,6,7], max_epochs=1e20, check_val_every_n_epoch=100,
                         default_root_dir=save_path)

    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=test_loader)
