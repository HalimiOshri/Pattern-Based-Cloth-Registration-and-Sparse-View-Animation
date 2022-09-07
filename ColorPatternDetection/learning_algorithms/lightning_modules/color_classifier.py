import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from learning_algorithms.data_processing.datasets.patch_keypoints_dataset import PatchKeypointDataset
import os
import cv2
import torch.nn as nn
import time
import kornia

class ColorClassifier(pl.LightningModule):
    def __init__(self):
        super(ColorClassifier, self).__init__()

        self.loss = torch.nn.CrossEntropyLoss()
        self.num_classes = 8

        self.rgb_map = np.array([
            [0, 0, 0], # black
            [255, 0, 0], #red
            [0, 255, 0], #green
            [255, 255, 0], #yellow
            [0, 0, 255], #blue
            [255, 0, 255], #magenta
            [0, 255, 255], #cyan
            [255, 255, 255] #white
        ])

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
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

    def to_rgb(self, label_image):
        label_image_np = label_image.detach().cpu().numpy()
        rgb = self.rgb_map[label_image_np]
        return rgb

    def test_step(self, batch, batch_idx):
        save_dir = os.path.join(self.trainer.log_dir, 'test', f'epoch-{self.current_epoch}')
        os.makedirs(save_dir, exist_ok=True)

        start_time = time.time()
        image = batch["image"]
        labels_prediction = self.forward(image)
        labels_prediction = torch.argmax(labels_prediction, dim=1)
        saved_image = self.to_rgb(labels_prediction)
        print("--- %s seconds ---" % (time.time() - start_time))
        for i in range(saved_image.shape[0]):
            cv2.imwrite(os.path.join(save_dir, f'{batch["cam"][i]}_{batch["frame"][i]}.png'), np.flip(saved_image[i], axis=-1).astype(np.uint8)) # opencv assumes BGR convention


    def validation_step(self, batch, batch_idx):
        save_dir = '/mnt/home/oshrihalimi/Results/color_pattern_detector_inference/cross_entropy_experiment_1.0_weight_black_scheduler_patience_100/'
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

    def forward(self, image):
        image = image / 255
        down1 = self.down1(image)
        out1 = F.max_pool2d(down1, kernel_size=2, stride=2)
        # upsample out_last, concatenate with down1 and apply conv operations
        out = F.upsample(out1, scale_factor=float(2), mode='bilinear')
        out = torch.cat([down1, out], 1)
        out = self.up4(out)

        # final 1x1 conv for predictions
        final_out = self.final_conv(out)
        labels_prediction = final_out
        return labels_prediction

    def training_step(self, batch, batch_idx):
        image = batch["patch"]
        locations = batch["centers"][..., :2] # B x #samples x 2
        print(locations.shape)
        batch_idx = torch.arange(locations.shape[0])[:, None, None].expand((locations.shape[0:2] + (1,))).to(device=locations.device) # B x #samples x 1
        sample_idx = torch.cat((batch_idx, locations), dim=2).view(-1, 3).long() # (B * #samples) x 1
        labels = batch["centers"][..., 2]
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes) # currently zero based - 8 classes (including corners) # the gt labels are 1 based indices # B x #samples x 7

        labels_prediction = self.forward(image)

        labels_prediction_sampled = labels_prediction[sample_idx[:, 0], :, sample_idx[:, 2], sample_idx[:, 1]] # the locations coordintes should be flipped!
        labels_prediction_sampled = torch.softmax(labels_prediction_sampled, -1)
        loss1 = F.cross_entropy(labels_prediction_sampled,
                         labels_one_hot.view(-1, self.num_classes).to(dtype=torch.float),
                         weight=torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1]).to(device=labels_prediction_sampled.device)) # inputs & target shape: (B * #samples) x 7

        #loss2 = kornia.losses.total_variation(image / torch.max(labels_prediction, axis=1).values[:, None, :, :]).mean()
        # image_lapacian = kornia.filters.blur_pool2d(kornia.color.gray.RgbToGrayscale()(image) / 255, kernel_size=3)
        # labels_prediction_lapacian = kornia.filters.blur_pool2d(torch.max(labels_prediction, dim=1, keepdim=True)[0], kernel_size=3)
        # loss2 = ((image_lapacian - labels_prediction_lapacian) ** 2).mean()
        # loss2 = kornia.losses.dice_loss(input=labels_prediction_sampled[:, :, None, None], target=labels.view(-1)[:, None, None])

        x = labels_prediction_sampled.to(dtype=torch.double)
        x = x[:, 1:] #only colors - not black background

        # vx = x - torch.mean(x, axis=0)[None, :]
        # sigma_x = torch.mean(vx ** 2, axis=0) ** 0.5
        # sx = vx / sigma_x
        # corr = torch.matmul(sx.transpose(1, 0), sx) / vx.shape[0]
        # loss2 = ((corr - torch.eye(self.num_classes).to(device=corr.device)) ** 2).sum()

        # concurrence loss for colors
        concurrence = torch.tril(torch.matmul(x.transpose(1, 0), x) / x.shape[0], diagonal=-1).sum()
        loss2 = concurrence
        loss = loss1 + loss2
        self.log("loss_CE", loss1)
        self.log("loss_color_concurrence", loss2)
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
    save_path = '/mnt/home/oshrihalimi/color_pattern_detection/uniform_cross_entropy_1_and color_concurrence loss_1_scheduler_patience_100'

    classifier = ColorClassifier(save_path=save_path)
    dataset = PatchKeypointDataset(path_dataset_file=dataset_file_path)
    train_loader = DataLoader(dataset, batch_size=300, shuffle=True)
    validation_loader = DataLoader(dataset, batch_size=30, shuffle=True)
    trainer = pl.Trainer(gpus=[0,1,2,3,4,5,6,7], max_epochs=1e20, val_check_interval=1,
                         default_root_dir=save_path)

    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=validation_loader)
