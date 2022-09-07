"""
A version of convolutional VAE which has separate encoders.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.aux.ae_geometry import GeometryBase

from src.losses.losses import kl_loss, laplacian_loss


from models.aux.io import config_to_model

from models.aux.blocks import (
    Conv2dBias,
    ConvDownBlock,
    ConvBlock,
    UpConvBlockDeep,
    WeightNorm,
    weights_initializer,
    WarpConv,
    tile2d,
)


from src.rendering.render import sample_uv, values_to_uv

import cv2

import logging

logger = logging.getLogger(__name__)


class ConvEncoder(nn.Module):
    """A joint encoder for tex and geometry."""

    def __init__(
        self,
        dataset,
        body_lbs_fn,
        clothes_lbs_fn,
        num_blocks,
        num_channels,
        num_embs_channels,
        body_verts_unposed_mean,
        clothes_verts_unposed_mean,
        noise_std,
        uv_size,
        global_scaling,
        body_tex_mean=None,
        body_tex_std=None,
        clothes_tex_mean=None,
        clothes_tex_std=None,
        lrelu_slope=0.2,
        encode_texture=True,
        # use_masks=True,  # not used?
        **kwargs,
    ):
        """Fixed-width conv encoder."""
        super().__init__()

        self.uv_size = uv_size

        self.body_geometry_base = GeometryBase(
            uv_size=self.uv_size,
            lbs_scale=dataset.body_lbs_scale,
            lbs_template_verts=dataset.body_lbs_template_verts,
            uv_coords=dataset.body_uv_coords,
            uv_mapping=dataset.body_uv_mapping,
            uv_faces=dataset.body_uv_faces,
            nbs_idxs=dataset.body_nbs_idxs,
            nbs_weights=dataset.body_nbs_weights,
            global_scaling=global_scaling,

        )
        self.clothes_geometry_base = GeometryBase(
            uv_size=self.uv_size,
            lbs_scale=dataset.clothes_lbs_scale,
            lbs_template_verts=dataset.clothes_lbs_template_verts,
            uv_coords=dataset.clothes_uv_coords,
            uv_mapping=dataset.clothes_uv_mapping,
            uv_faces=dataset.clothes_uv_faces,
            nbs_idxs=dataset.clothes_nbs_idxs,
            nbs_weights=dataset.clothes_nbs_weights,
            global_scaling=global_scaling,
        )

        self.body_geometry_base.lbs_fn = body_lbs_fn
        self.clothes_geometry_base.lbs_fn = clothes_lbs_fn
        self.num_blocks = num_blocks
        self.sizes = [self.uv_size // 2 ** b for b in range(num_blocks)]
        self.noise_std = noise_std

        # self.use_masks = use_masks

        # setting the sizes of conv layers
        self.num_embs_channels = num_embs_channels

        self.resize_tex = nn.UpsamplingBilinear2d((self.uv_size, self.uv_size))

        if isinstance(num_channels, int):
            self.num_channels = [3] + self.num_blocks * [num_channels]
        else:
            self.num_channels = list(num_channels)

        logger.info(f"ConvEncoder: num_channels = {self.num_channels}")

        # TODO: maybe here we can assume these will be specified directly?
        self.register_buffer("body_verts_unposed_mean", torch.tensor(body_verts_unposed_mean, dtype=torch.float32))
        self.register_buffer("clothes_verts_unposed_mean", torch.tensor(clothes_verts_unposed_mean, dtype=torch.float32))

        body_tex_mean = torch.tensor(body_tex_mean, dtype=torch.float32)
        body_tex_mean = self.resize_tex(body_tex_mean[np.newaxis])[0]
        self.register_buffer("body_tex_mean", body_tex_mean)
        self.register_buffer("body_tex_std", torch.tensor(body_tex_std))
        clothes_tex_mean = torch.tensor(clothes_tex_mean, dtype=torch.float32)
        clothes_tex_mean = self.resize_tex(clothes_tex_mean[np.newaxis])[0]
        self.register_buffer("clothes_tex_mean", clothes_tex_mean)
        self.register_buffer("clothes_tex_std", torch.tensor(clothes_tex_std))

        # # Donglai: already implemented in geometry_base
        # body_index_image, body_bary_image = render_vertex_index_image(
        #     self.body_geometry_base.uv_coords.cuda(),
        #     self.body_geometry_base.uv_faces.cuda(),
        #     self.body_geometry_base.uv_mapping.cuda(),
        #     self.body_geometry_base.uv_size,
        # )
        # self.body_geometry_base.register_buffer("index_image", body_index_image)
        # self.body_geometry_base.register_buffer("bary_image", body_bary_image)

        # clothes_index_image, clothes_bary_image = render_vertex_index_image(
        #     self.clothes_geometry_base.uv_coords.cuda(),
        #     self.clothes_geometry_base.uv_faces.cuda(),
        #     self.clothes_geometry_base.uv_mapping.cuda(),
        #     self.clothes_geometry_base.uv_size,
        # )
        # self.clothes_geometry_base.register_buffer("index_image", clothes_index_image)
        # self.clothes_geometry_base.register_buffer("bary_image", clothes_bary_image)

        self.conv_blocks = nn.ModuleList([])

        # NOTE: there are four groups, verts and texture for body and clothes
        self.encode_texture = encode_texture
        if encode_texture:
            num_groups = 4
            logger.info('encoding mean texture, num of groups = 4')
        else:
            num_groups = 2
            logger.info('NOT encoding mean texture, num of groups = 2')

        for b in range(self.num_blocks):
            self.conv_blocks.append(
                ConvDownBlock(
                    self.num_channels[b] * num_groups,
                    self.num_channels[b + 1] * num_groups,
                    self.sizes[b],
                    groups=num_groups,
                )
            )

        # TODO: should we put initializer
        self.mu_conv = WeightNorm(
            nn.Conv2d(
                self.num_channels[-1] * num_groups,
                self.num_embs_channels * num_groups,
                1,
                groups=num_groups,
            )
        )
        self.logvar_conv = WeightNorm(
            nn.Conv2d(
                self.num_channels[-1] * num_groups,
                self.num_embs_channels * num_groups,
                1,
                groups=num_groups,
            )
        )

        self.apply(weights_initializer(lrelu_slope))

        logger.warning("NOTE: the initialization for mu / logvar has changed")
        self.mu_conv.apply(weights_initializer(1.0))
        self.logvar_conv.apply(weights_initializer(1.0))

    def forward(self, motion, body_verts, clothes_verts, body_tex, clothes_tex, **kwargs):

        # batch_size = motion.shape[0]

        preds = dict()

        # converting motion to the unposed
        scale = self.body_geometry_base.lbs_scale.expand(motion.shape[0], -1)
        body_verts_unposed = (
            self.body_geometry_base.lbs_fn.unpose(motion, scale, body_verts / self.body_geometry_base.global_scaling)
            - self.body_verts_unposed_mean
        )
        clothes_verts_unposed = (
            self.clothes_geometry_base.lbs_fn.unpose(motion, scale, clothes_verts / self.clothes_geometry_base.global_scaling)
            - self.clothes_verts_unposed_mean
        )

        """
        from codec_hand.pyutils import io
        v, f = io.load_ply('/mnt/home/donglaix/data/bodyPoseTrackingUpper/int_0_cpl_10000.0_lpl_0/LowResMesh/000500.ply')
        unposed = self.body_geometry_base.lbs_fn.unpose(motion, scale, body_verts / self.body_geometry_base.global_scaling)
        import pdb; pdb.set_trace()
        io.save_ply('/mnt/home/donglaix/body_unposed.ply', unposed[0].detach().cpu(), f)

        v, f = io.load_ply('/mnt/home/donglaix/data/codecResClothes/000500.ply')
        unposed = self.clothes_geometry_base.lbs_fn.unpose(motion, scale, clothes_verts / self.clothes_geometry_base.global_scaling)
        io.save_ply('/mnt/home/donglaix/clothes_unposed.ply', unposed[0].detach().cpu(), f)
        """

        # TODO: maybe this has to be done once in the upper class?
        body_verts_unposed_uv = values_to_uv(body_verts_unposed, self.body_geometry_base.index_image, self.body_geometry_base.bary_image)
        clothes_verts_unposed_uv = values_to_uv(clothes_verts_unposed, self.clothes_geometry_base.index_image, self.clothes_geometry_base.bary_image)

        if self.encode_texture:
            body_tex_delta = (body_tex - self.body_tex_mean) / self.body_tex_std
            clothes_tex_delta = (clothes_tex - self.clothes_tex_mean) / self.clothes_tex_std
            joint = torch.cat([body_verts_unposed_uv, body_tex_delta, clothes_verts_unposed_uv, clothes_tex_delta], 1)
        else:
            joint = torch.cat([body_verts_unposed_uv, clothes_verts_unposed_uv], 1)

        x = joint

        for b in range(self.num_blocks):
            x = self.conv_blocks[b](x)

        # these are body-only embeddings
        embs_mu = self.mu_conv(x)
        embs_logvar = self.logvar_conv(x)

        # NOTE: the noise is only applied to the input-conditioned values
        if self.training:
            noise = torch.randn_like(embs_mu)
            embs = embs_mu + torch.exp(0.5 * embs_logvar) * noise * self.noise_std
        else:
            embs = embs_mu.clone()

        preds.update(
            embs=embs,
            embs_mu=embs_mu,
            embs_logvar=embs_logvar,
        )

        return preds


class ConvDecoder(nn.Module):
    """Multi-region view-independent decoder."""

    def __init__(
        self,
        dataset,
        init_uv_size,
        num_pose_dims,
        num_pose_enc_dims,
        num_embs_channels,
        init_channels,
        min_channels=8,
        lrelu_slope=0.2,
        drop_mask=False,
        body_tex_mean=None,
        body_tex_std=None,
        clothes_tex_mean=None,
        clothes_tex_std=None,
        warping=True,
        body_conditioned_on_pose=True,
        body_conditioned_on_face_kpts=False,
        body_conditioned_on_latent=False,
        clothes_conditioned_on_pose=False,
        clothes_conditioned_on_latent=True,
        face_region_path=None,
        face_region_erode=False,
        face_kpts_ref_path=None,
        n_face_kpts=0,
        n_face_embs_enc_channels=0,
        pose_regions_path=None,
        **kwargs,
    ):
        """Constructor.

        Args:
            init_uv_size: the size of the initial uv size. Embeddings are resizes to this size
            num_embs_channels: the number of input embeddings. NOTE: this is per tex / verts
        """
        super().__init__()

        self.uv_size = kwargs['uv_size']

        self.body_geometry_base = GeometryBase(
            uv_size=self.uv_size,
            lbs_scale=dataset.body_lbs_scale,
            lbs_template_verts=dataset.body_lbs_template_verts,
            uv_coords=dataset.body_uv_coords,
            uv_mapping=dataset.body_uv_mapping,
            uv_faces=dataset.body_uv_faces,
            nbs_idxs=dataset.body_nbs_idxs,
            nbs_weights=dataset.body_nbs_weights,
            lbs=kwargs['body_lbs'],
            global_scaling=kwargs['global_scaling'],
        )
        self.clothes_geometry_base = GeometryBase(
            uv_size=self.uv_size,
            lbs_scale=dataset.clothes_lbs_scale,
            lbs_template_verts=dataset.clothes_lbs_template_verts,
            uv_coords=dataset.clothes_uv_coords,
            uv_mapping=dataset.clothes_uv_mapping,
            uv_faces=dataset.clothes_uv_faces,
            nbs_idxs=dataset.clothes_nbs_idxs,
            nbs_weights=dataset.clothes_nbs_weights,
            lbs=kwargs['clothes_lbs'],
            global_scaling=kwargs['global_scaling'],
        )

        self.init_uv_size = init_uv_size
        self.num_embs_channels = num_embs_channels

        self.num_blocks = int(np.log2(self.uv_size // self.init_uv_size))
        self.sizes = [init_uv_size * 2 ** s for s in range(self.num_blocks + 1)]

        self.drop_mask = drop_mask

        self.num_channels = [
            max(init_channels // 2 ** b, min_channels)
            for b in range(self.num_blocks + 1)
        ]

        self.resize_tex = nn.UpsamplingBilinear2d((self.uv_size, self.uv_size))

        body_tex_mean = torch.tensor(body_tex_mean, dtype=torch.float32)
        body_tex_mean = self.resize_tex(body_tex_mean[np.newaxis])[0]
        self.register_buffer("body_tex_mean", body_tex_mean)
        self.register_buffer("body_tex_std", torch.tensor(body_tex_std))
        clothes_tex_mean = torch.tensor(clothes_tex_mean, dtype=torch.float32)
        clothes_tex_mean = self.resize_tex(clothes_tex_mean[np.newaxis])[0]
        self.register_buffer("clothes_tex_mean", clothes_tex_mean)
        self.register_buffer("clothes_tex_std", torch.tensor(clothes_tex_std))

        # processing the pose
        # encoding only truly local part of the pose (3)
        self.local_pose_conv_block = ConvBlock(
            num_pose_dims,
            num_pose_enc_dims,
            self.init_uv_size,
            kernel_size=1,
            padding=0,
        )

        num_groups = 4

        self.body_conditioned_on_pose = body_conditioned_on_pose
        self.body_conditioned_on_face_kpts = body_conditioned_on_face_kpts
        self.body_conditioned_on_latent = body_conditioned_on_latent
        self.clothes_conditioned_on_pose = clothes_conditioned_on_pose
        self.clothes_conditioned_on_latent = clothes_conditioned_on_latent
        body_condition_dim = 0
        if self.body_conditioned_on_pose:
            body_condition_dim += num_pose_enc_dims
            logger.info('decoder: body conditioned on pose')
        if self.body_conditioned_on_face_kpts:
            body_condition_dim += n_face_embs_enc_channels
            logger.info('decoder: body conditioned on face kpts')
        if self.body_conditioned_on_latent:
            body_condition_dim += num_embs_channels
            logger.info('decoder: body conditioned on latent code')
        clothes_condition_dim = 0
        if self.clothes_conditioned_on_pose:
            clothes_condition_dim += num_pose_enc_dims
            logger.info('decoder: clothes conditioned on pose')
        if self.clothes_conditioned_on_latent:
            clothes_condition_dim += num_embs_channels
            logger.info('decoder: clothes conditioned on latent code')

        self.conv_block_body_verts = ConvBlock(
            body_condition_dim, init_channels, self.init_uv_size,
        )

        self.conv_block_body_tex = ConvBlock(
            body_condition_dim, init_channels, self.init_uv_size,
        )

        self.conv_block_clothes_verts = ConvBlock(
            clothes_condition_dim, init_channels, self.init_uv_size,
        )

        self.conv_block_clothes_tex = ConvBlock(
            clothes_condition_dim, init_channels, self.init_uv_size,
        )

        self.embs_resize = nn.UpsamplingBilinear2d(
            size=(self.init_uv_size, self.init_uv_size)
        )

        self.conv_blocks = nn.ModuleList([])
        for b in range(self.num_blocks):
            self.conv_blocks.append(
                UpConvBlockDeep(
                    self.num_channels[b] * num_groups,
                    self.num_channels[b + 1] * num_groups,
                    self.sizes[b + 1],
                    groups=num_groups,
                ),
            )

        self.body_verts_conv = WeightNorm(
            Conv2dBias(self.num_channels[-1], 3, 3, self.uv_size)
        )

        self.body_tex_conv = WeightNorm(
            Conv2dBias(self.num_channels[-1], 3, 3, self.uv_size)
        )

        self.clothes_verts_conv = WeightNorm(
            Conv2dBias(self.num_channels[-1], 3, 3, self.uv_size)
        )

        self.clothes_tex_conv = WeightNorm(
            Conv2dBias(self.num_channels[-1], 3, 3, self.uv_size)
        )

        self.warping = warping
        if self.warping:
            self.body_warp_block = WarpConv(self.num_channels[-1], 3, self.uv_size)
            self.clothes_warp_block = WarpConv(self.num_channels[-1], 3, self.uv_size)
        else:
            logger.info(f"skipping the warping layer")

        if pose_regions_path is not None:
            logger.info('Decoder: use pose region path')

            pose_regions = torch.as_tensor(
                np.load(pose_regions_path)[6:], dtype=torch.float32
            ).cuda()
            pose_map_hd = values_to_uv(pose_regions[..., np.newaxis], self.body_geometry_base.index_image, self.body_geometry_base.bary_image)
            pose_map_ksize = 51
            pose_map_hd = F.max_pool2d(
                pose_map_hd, pose_map_ksize, 1, padding=pose_map_ksize // 2
            )
            pose_cond_mask = (
                F.interpolate(pose_map_hd, size=(init_uv_size, init_uv_size))
                .to(torch.int32)
                .permute(1, 0, 2, 3)
            )
            self.register_buffer("pose_cond_mask", pose_cond_mask)  # the mask is only for the body
        else:
            self.pose_cond_mask = 1.0

        self.n_face_kpts = n_face_kpts
        self.n_face_embs_enc_channels = n_face_embs_enc_channels

        if self.body_conditioned_on_face_kpts:
            self.face_embs_conv_block = ConvBlock(
                n_face_kpts * 3,
                n_face_embs_enc_channels,
                init_uv_size,
                kernel_size=1,
                padding=0,
            )

            if face_region_path is not None and not face_region_erode:
                logger.info(f"using face region from {face_region_path}")
                face_mask_hd = torch.as_tensor(np.load(face_region_path), dtype=torch.float32)
                face_cond_mask = F.interpolate(
                    face_mask_hd[np.newaxis, np.newaxis],
                    (self.init_uv_size, self.init_uv_size),
                ).to(torch.int32)
            elif face_region_path is not None and face_region_erode:
                logger.info(f"using face region from {face_region_path}")
                logger.info("doing erosion to shrink it")

                np_face_region = np.load(face_region_path)
                np_face_region = cv2.erode(
                    cv2.resize(
                        np_face_region,
                        (self.init_uv_size, self.init_uv_size),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.uint8),
                    np.ones((3, 3), dtype=np.uint8),
                )
                face_cond_mask = torch.as_tensor(np_face_region, dtype=torch.int32)
            else:
                logger.info("using a hardcoded face region")
                face_cond_mask = torch.zeros(
                    (1, 1, self.init_uv_size, self.init_uv_size), dtype=torch.int32
                )
                face_cond_mask[..., self.init_uv_size // 2 :, : self.init_uv_size // 2] = 1
            self.register_buffer("face_cond_mask", face_cond_mask)

            if face_kpts_ref_path:
                # a subset for kpts
                self.register_buffer(
                    "face_kpts_ref",
                    torch.as_tensor(np.load(face_kpts_ref_path), dtype=torch.float32),
                )

                face_kpts_idxs = torch.where(torch.all(self.face_kpts_ref != 0.0, axis=-1))[0].to(
                    torch.int64
                )
                self.register_buffer("face_kpts_idxs", face_kpts_idxs)

                assert self.n_face_kpts == len(face_kpts_idxs), "these should match"
            else:
                logger.warning("`face_kpts_ref_path` is not provided")

        self.apply(weights_initializer(lrelu_slope))

        self.body_verts_conv.apply(weights_initializer(1.0))
        self.body_tex_conv.apply(weights_initializer(1.0))
        self.clothes_verts_conv.apply(weights_initializer(1.0))
        self.clothes_tex_conv.apply(weights_initializer(1.0))

    def forward(
        self, motion, embs, face_kpts_3d=None, face_kpts_3d_valid=None, **kwargs,
    ):
        batch_size = motion.shape[0]
        preds = dict()

        # processing pose
        local_pose = motion[:, 6:]

        local_pose_conv = self.local_pose_conv_block(
            tile2d(local_pose, self.init_uv_size) * self.pose_cond_mask
        )
        preds.update(local_pose_conv=local_pose_conv)

        embs_conv = self.embs_resize(embs)

        # zeroing the latent dimensions at training time
        if self.training and self.drop_mask:
            drop_mask = torch.randint(
                0, 2, (batch_size, 1, 1, 1), device=embs_conv.device
            )
            embs_conv = embs_conv * drop_mask

        embs_conv_body, embs_conv_clothes = torch.split(
            embs_conv, self.num_embs_channels, 1
        )
        assert embs_conv_body.shape[1] == embs_conv_clothes.shape[1]  # should have equal channel numbers

        if self.body_conditioned_on_face_kpts:
            face_embs = (
                (face_kpts_3d - self.face_kpts_ref[np.newaxis])
                * face_kpts_3d_valid[..., np.newaxis]
            )[:, self.face_kpts_idxs]
            face_embs = face_embs.reshape(batch_size, self.n_face_kpts * 3)
            face_embs_masked = tile2d(face_embs, self.init_uv_size) * self.face_cond_mask
            face_embs_conv = self.face_embs_conv_block(face_embs_masked)
            preds.update(face_embs_conv=face_embs_conv)

        body_condition = []
        if self.body_conditioned_on_pose:
            body_condition.append(local_pose_conv)
        if self.body_conditioned_on_face_kpts:
            body_condition.append(face_embs_conv)
        if self.body_conditioned_on_latent:
            body_condition.append(embs_conv_body)
        body_condition = torch.cat(body_condition, dim=1)
        clothes_condition = []
        if self.clothes_conditioned_on_pose:
            clothes_condition.append(local_pose_conv)
        if self.clothes_conditioned_on_latent:
            clothes_condition.append(embs_conv_clothes)
        clothes_condition = torch.cat(clothes_condition, dim=1)

        body_verts_conv = self.conv_block_body_verts(body_condition)
        body_tex_conv = self.conv_block_body_tex(body_condition)

        clothes_verts_conv = self.conv_block_clothes_verts(clothes_condition)
        clothes_tex_conv = self.conv_block_clothes_tex(clothes_condition)

        # NOTE: the are joint embeddings
        joint_embs = torch.cat([body_verts_conv, body_tex_conv, clothes_verts_conv, clothes_tex_conv], axis=1)

        x = joint_embs
        for b in range(self.num_blocks):
            x = self.conv_blocks[b](x)
        body_verts_features, body_tex_features, clothes_verts_features, clothes_tex_features = torch.split(x, self.num_channels[-1], 1)

        body_verts_uv_delta_rec = self.body_verts_conv(body_verts_features)
        body_verts_delta_rec = sample_uv(body_verts_uv_delta_rec, self.body_geometry_base.uv_coords, self.body_geometry_base.uv_mapping)
        body_verts_template_pose = self.body_geometry_base.lbs_template_verts + body_verts_delta_rec
        scale = self.body_geometry_base.lbs_scale.expand(motion.shape[0], -1)
        body_verts_rec = (
            self.body_geometry_base.lbs_fn(motion, scale, body_verts_template_pose)
            * self.body_geometry_base.global_scaling[np.newaxis]
        )
        preds.update(body_verts=body_verts_rec, body_verts_uv_delta_rec=body_verts_uv_delta_rec, body_verts_template_pose=body_verts_template_pose)

        clothes_verts_uv_delta_rec = self.clothes_verts_conv(clothes_verts_features)
        clothes_verts_delta_rec = sample_uv(clothes_verts_uv_delta_rec, self.clothes_geometry_base.uv_coords, self.clothes_geometry_base.uv_mapping)
        clothes_verts_template_pose = self.clothes_geometry_base.lbs_template_verts + clothes_verts_delta_rec
        scale = self.clothes_geometry_base.lbs_scale.expand(motion.shape[0], -1)
        clothes_verts_rec = (
            self.clothes_geometry_base.lbs_fn(motion, scale, clothes_verts_template_pose)
            * self.clothes_geometry_base.global_scaling[np.newaxis]
        )
        preds.update(clothes_verts=clothes_verts_rec, clothes_verts_uv_delta_rec=clothes_verts_uv_delta_rec, clothes_verts_template_pose=clothes_verts_template_pose)

        body_tex_rec_norm = self.body_tex_conv(body_tex_features)
        clothes_tex_rec_norm = self.clothes_tex_conv(clothes_tex_features)

        if self.warping:
            preds.update(  # for visualization purpose
                unwarped_body_tex=body_tex_rec_norm * self.body_tex_std + self.body_tex_mean,
                unwarped_clothes_tex=clothes_tex_rec_norm * self.clothes_tex_std + self.clothes_tex_mean
            )
            body_tex_rec_norm = self.body_warp_block(body_tex_rec_norm, body_tex_features)
            clothes_tex_rec_norm = self.clothes_warp_block(clothes_tex_rec_norm, clothes_tex_features)

        body_tex_rec = body_tex_rec_norm * self.body_tex_std + self.body_tex_mean
        clothes_tex_rec = clothes_tex_rec_norm * self.clothes_tex_std + self.clothes_tex_mean
        # NOTE: we are computing the basic texture loss here to avoid transfer on gpu
        preds.update(body_tex=body_tex_rec, body_tex_norm=body_tex_rec_norm, clothes_tex=clothes_tex_rec, clothes_tex_norm=clothes_tex_rec_norm)

        return preds


class RegionVAE(nn.Module):
    def __init__(self, encoder, decoder, dataset, invisible_mean=False):
        super().__init__()

        self.invisible_mean = invisible_mean

        if self.invisible_mean:
            logger.info(f"Enforce the texture to be mean of interpolation when invisible")

        # TODO: should we have a shared LBS here?
        self.decoder = config_to_model(
            decoder,
            dataset=dataset,
            body_tex_mean=dataset.body_tex_mean,
            body_tex_std=dataset.body_tex_std,
            clothes_tex_mean=dataset.clothes_tex_mean,
            clothes_tex_std=dataset.clothes_tex_std,
        )

        self.encoder = config_to_model(
            encoder,
            dataset=dataset,
            body_lbs_fn=self.decoder.body_geometry_base.lbs_fn,
            clothes_lbs_fn=self.decoder.clothes_geometry_base.lbs_fn,
            body_verts_unposed_mean=dataset.body_verts_unposed_mean,
            clothes_verts_unposed_mean=dataset.clothes_verts_unposed_mean,
            body_tex_mean=dataset.body_tex_mean,
            body_tex_std=dataset.body_tex_std,
            clothes_tex_mean=dataset.clothes_tex_mean,
            clothes_tex_std=dataset.clothes_tex_std,
        )


    def forward(self, motion, body_verts, clothes_verts, body_tex, clothes_tex, mode="ae", **kwargs):
        preds = dict()

        if mode != "ae":
            raise ValueError(f"unknown mode {mode}")

        enc_preds = self.encoder(motion, body_verts, clothes_verts, body_tex, clothes_tex)
        dec_preds = self.decoder(motion, **enc_preds, **kwargs)
        preds.update(**dec_preds, **enc_preds)

        body_tex = self.decoder.resize_tex(body_tex)
        body_tex_norm = (body_tex - self.decoder.body_tex_mean) / self.decoder.body_tex_std
        clothes_tex = self.decoder.resize_tex(clothes_tex)
        clothes_tex_norm = (clothes_tex - self.decoder.clothes_tex_mean) / self.decoder.clothes_tex_std

        # TODO: sqrt?
        loss_body_verts_rec = (preds["body_verts"] - body_verts).pow(2).mean(dim=(1, 2))
        resized_body_tex_mask = self.decoder.resize_tex(kwargs['body_tex_mask'])
        loss_body_tex_rec = (
            resized_body_tex_mask
            * (preds["body_tex_norm"] - body_tex_norm) +
            ((1.0 - resized_body_tex_mask) * preds["body_tex_norm"] if self.invisible_mean else 0.0)
        ).abs().mean(dim=(1, 2, 3))
        loss_clothes_verts_rec = (preds["clothes_verts"] - clothes_verts).pow(2).mean(dim=(1, 2))
        resized_clothes_tex_mask = self.decoder.resize_tex(kwargs['clothes_tex_mask'])
        loss_clothes_tex_rec = (
            resized_clothes_tex_mask
            * (preds["clothes_tex_norm"] - clothes_tex_norm) +
            ((1.0 - resized_clothes_tex_mask) * preds["clothes_tex_norm"] if self.invisible_mean else 0.0)
        ).abs().mean(dim=(1, 2, 3))

        # TODO: add laplacian
        loss_body_verts_laplacian = laplacian_loss(
            preds["body_verts"], body_verts, self.encoder.body_geometry_base.nbs_idxs, self.encoder.body_geometry_base.nbs_weights,
        )
        loss_clothes_verts_laplacian = laplacian_loss(
            preds["clothes_verts"], clothes_verts, self.encoder.clothes_geometry_base.nbs_idxs, self.encoder.clothes_geometry_base.nbs_weights,
        )

        # computing normal average losses
        preds.update(
            loss_body_verts_rec=loss_body_verts_rec,
            loss_body_verts_laplacian=loss_body_verts_laplacian,
            loss_body_tex_rec=loss_body_tex_rec,
            loss_clothes_verts_rec=loss_clothes_verts_rec,
            loss_clothes_verts_laplacian=loss_clothes_verts_laplacian,
            loss_clothes_tex_rec=loss_clothes_tex_rec,
        )

        return preds


class TotalVAELoss(nn.Module):
    def __init__(self, weights, kl_type="default", **kwargs):
        super().__init__()
        self.weights = weights
        self.kl_type = kl_type
        assert self.kl_type in ["default", "anneal"]

    def forward(self, preds, targets, inputs=None, iteration=None):

        loss_dict = dict()

        loss_dict.update(
            loss_body_verts_rec=preds["loss_body_verts_rec"].mean(),
            loss_body_tex_rec=preds["loss_body_tex_rec"].mean(),
            loss_body_verts_laplacian=preds["loss_body_verts_laplacian"].mean(),
            loss_clothes_verts_rec=preds["loss_clothes_verts_rec"].mean(),
            loss_clothes_tex_rec=preds["loss_clothes_tex_rec"].mean(),
            loss_clothes_verts_laplacian=preds["loss_clothes_verts_laplacian"].mean(),
            loss_kl_embs=kl_loss(preds["embs_mu"], preds["embs_logvar"]),
        )

        loss = (
            loss_dict["loss_body_verts_rec"] * self.weights.body_geometry_rec
            + loss_dict["loss_body_tex_rec"] * self.weights.body_tex_rec
            + loss_dict["loss_body_verts_laplacian"] * self.weights.body_geometry_laplacian
            + loss_dict["loss_clothes_verts_rec"] * self.weights.clothes_geometry_rec
            + loss_dict["loss_clothes_tex_rec"] * self.weights.clothes_tex_rec
            + loss_dict["loss_clothes_verts_laplacian"] * self.weights.clothes_geometry_laplacian
        )
        if self.kl_type == "default":
            # standard KL
            loss += loss_dict["loss_kl_embs"] * self.weights.kl
        elif self.kl_type == "anneal" and iteration is not None:
            c = self.weights.kl_anneal
            kl_weight = (1.0 - min(iteration, c.end_at) / c.end_at) * (
                c.initial_value - c.min_value
            ) + c.min_value
            loss += loss_dict["loss_kl_embs"] * kl_weight
            loss_dict["loss_kl_weight"] = torch.tensor(kl_weight)

        loss_dict.update(loss_total=loss)

        return loss, loss_dict



