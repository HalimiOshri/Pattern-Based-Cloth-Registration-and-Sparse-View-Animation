import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from learning_algorithms.data_processing.datasets.patch_keypoints_dataset import PatchKeypointDataset
import os
import cv2
import time
from learning_algorithms.torch_modules.unet import UNet

class ColorClassifier(pl.LightningModule):
    def __init__(self, save_path):
        super(ColorClassifier, self).__init__()

        self.loss = torch.nn.CrossEntropyLoss()
        self.num_classes = 8
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

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

        self.unet = UNet(n_channels=3, n_classes=8)

    def validation_step(self, batch, batch_idx):
        if (self.current_epoch % 100) != 0:
            return
        save_dir = os.path.join(self.save_path, f'{self.current_epoch}')
        os.makedirs(save_dir, exist_ok=True)

        image = batch["patch"]
        filename = batch["filename"]

        labels_prediction = self.forward(image)

        labels_prediction = torch.argmax(labels_prediction, dim=1)
        saved_image = self.to_rgb(labels_prediction)
        for i in range(saved_image.shape[0]):
            cv2.imwrite(os.path.join(save_dir, filename[i]), np.flip(saved_image[i], axis=-1).astype(np.uint8)) # opencv assumes BGR convention

        start_time = time.time()
        example_path = '/mnt/home/oshrihalimi/Data/ColorPattern/cam401534/image0005.png'
        example_image = torch.Tensor(cv2.imread(example_path).transpose(2, 0, 1)[None, ...].astype(float)).to(device=image.device)
        example_prediction = self.forward(example_image)
        example_prediction = torch.argmax(example_prediction, dim=1)
        print("--- %s seconds ---" % (time.time() - start_time))
        saved_example_image = self.to_rgb(example_prediction)
        cv2.imwrite(os.path.join(save_dir, 'example_cam401534_frame_0005.png'),
                    np.flip(saved_example_image[0], axis=-1).astype(np.uint8))  # opencv assumes BGR convention
        print(example_path)

        start_time = time.time()
        example_path = '/mnt/home/oshrihalimi/Data/ColorPattern/cam401534/image5000.png'
        example_image = torch.Tensor(cv2.imread(example_path).transpose(2, 0, 1)[None, ...].astype(float)).to(device=image.device)
        example_prediction = self.forward(example_image)
        example_prediction = torch.argmax(example_prediction, dim=1)
        print("--- %s seconds ---" % (time.time() - start_time))
        saved_example_image = self.to_rgb(example_prediction)
        cv2.imwrite(os.path.join(save_dir, 'example_cam401534_frame_5000.png'),
                    np.flip(saved_example_image[0], axis=-1).astype(np.uint8))  # opencv assumes BGR convention
        print(example_path)

        start_time = time.time()
        example_path = '/mnt/home/oshrihalimi/Data/ColorPattern/cam400897/image0005.png'
        example_image = torch.Tensor(cv2.imread(example_path).transpose(2, 0, 1)[None, ...].astype(float)).to(device=image.device)
        example_prediction = self.forward(example_image)
        example_prediction = torch.argmax(example_prediction, dim=1)
        print("--- %s seconds ---" % (time.time() - start_time))
        saved_example_image = self.to_rgb(example_prediction)
        cv2.imwrite(os.path.join(save_dir, 'example_cam400897_frame_0005.png'),
                    np.flip(saved_example_image[0], axis=-1).astype(np.uint8))  # opencv assumes BGR convention
        print(example_path)

        start_time = time.time()
        example_path = '/mnt/home/oshrihalimi/Data/ColorPattern/cam400897/image5000.png'
        example_image = torch.Tensor(cv2.imread(example_path).transpose(2, 0, 1)[None, ...].astype(float)).to(device=image.device)
        example_prediction = self.forward(example_image)
        example_prediction = torch.argmax(example_prediction, dim=1)
        print("--- %s seconds ---" % (time.time() - start_time))
        saved_example_image = self.to_rgb(example_prediction)
        cv2.imwrite(os.path.join(save_dir, 'example_cam400897_frame_5000.png'),
                    np.flip(saved_example_image[0], axis=-1).astype(np.uint8))  # opencv assumes BGR convention
        print(example_path)

        start_time = time.time()
        example_path = '/mnt/home/oshrihalimi/Data/ColorPattern/cam401535/image0005.png'
        example_image = torch.Tensor(cv2.imread(example_path).transpose(2, 0, 1)[None, ...].astype(float)).to(device=image.device)
        example_prediction = self.forward(example_image)
        example_prediction = torch.argmax(example_prediction, dim=1)
        print("--- %s seconds ---" % (time.time() - start_time))
        saved_example_image = self.to_rgb(example_prediction)
        cv2.imwrite(os.path.join(save_dir, 'example_401535_frame_0005.png'),
                    np.flip(saved_example_image[0], axis=-1).astype(np.uint8))  # opencv assumes BGR convention
        print(example_path)

        start_time = time.time()
        example_path = '/mnt/home/oshrihalimi/Data/ColorPattern/cam401535/image5000.png'
        example_image = torch.Tensor(cv2.imread(example_path).transpose(2, 0, 1)[None, ...].astype(float)).to(device=image.device)
        example_prediction = self.forward(example_image)
        example_prediction = torch.argmax(example_prediction, dim=1)
        print("--- %s seconds ---" % (time.time() - start_time))
        saved_example_image = self.to_rgb(example_prediction)
        cv2.imwrite(os.path.join(save_dir, 'example_401535_frame_5000.png'),
                    np.flip(saved_example_image[0], axis=-1).astype(np.uint8))  # opencv assumes BGR convention
        print(example_path)

    def to_rgb(self, label_image):
        label_image_np = label_image.detach().cpu().numpy()
        rgb = self.rgb_map[label_image_np]
        return rgb


    def forward(self, image):
        image = image / 255
        final_out = self.unet(image)

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
        #loss2 = concurrence
        loss = loss1# + loss2
        self.log("loss_CE", loss1)
        #self.log("loss_color_concurrence", loss2)
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
    save_path = '/mnt/home/oshrihalimi/color_pattern_detection/uniform_cross_entropy_scheduler_patience_100_unet_5_upsample_2x2'

    classifier = ColorClassifier(save_path=save_path)
    dataset = PatchKeypointDataset(path_dataset_file=dataset_file_path)
    train_loader = DataLoader(dataset, batch_size=100, shuffle=True)
    validation_loader = DataLoader(dataset, batch_size=30, shuffle=True)
    trainer = pl.Trainer(gpus=[0,1,2,3,4,5,6,7], max_epochs=1e20, val_check_interval=1,
                         default_root_dir=save_path)

    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=validation_loader)
