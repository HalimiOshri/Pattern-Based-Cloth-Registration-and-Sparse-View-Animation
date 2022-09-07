import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
# from external.fbca.models.fake_dof import DOF

from external.fbca.models.linear_blend_skinning import LBSModule, LBSModuleOld

# from external.fbca.models.bodies.seams import SeamSampler

from external.fbca.pyutils.render import GeometryModule

from drtk.renderlayer import RenderLayer, edge_grad_estimator
from external.fbca.pyutils.geomutils import depth_discontuity_mask, vert_normals_v2

from external.fbca.models.blocks import (
    UpConvBlockDeep,
    weights_initializer,
)

from external.fbca.models.camera import CameraCalibrationNetV2
import external.fbca.models.layers as la


from external.fbca.models.shadow import ShadowUNet

from external.fbca.pyutils.geomutils import compute_tbn_uv

from external.fbca.models.image.unet import UNetWB


import logging

logger = logging.getLogger(__name__)


class UpscaleNet(nn.Module):
    def __init__(
        self, in_channels, out_channels=3, n_ftrs=16, size=1024, upscale_factor=2
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            la.Conv2dWNUB(in_channels, n_ftrs, size, size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            la.Conv2dWNUB(n_ftrs, n_ftrs, size, size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out_block = la.Conv2dWNUB(
            n_ftrs,
            out_channels * upscale_factor ** 2,
            size,
            size,
            kernel_size=1,
            padding=0,
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.apply(weights_initializer(0.2))
        self.out_block.apply(weights_initializer(1.0))

    def forward(self, x):
        x = self.conv_block(x)
        x = self.out_block(x)
        return self.pixel_shuffle(x)

class ClothGeoTexUNet(nn.Module):
    def __init__(
        self,
        uv_size=1024,
        n_ftrs=4,
        num_input_channels=3,
        **kwargs,
    ):
        super().__init__()
        # TODO: can we just condition on the output of the face decoder?

        self.num_input_channels = num_input_channels
        self.uv_size = uv_size

        self.normals_conv = nn.Sequential(
            la.Conv2dWNUB(
                in_channels=num_input_channels,
                out_channels=n_ftrs,
                kernel_size=3,
                height=self.uv_size,
                width=self.uv_size,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.verts_unet = nn.Sequential(
            UNetWB(in_channels=n_ftrs, out_channels=n_ftrs, n_init_ftrs=n_ftrs * 2, size=uv_size, out_scale=0.1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.verts_conv = la.Conv2dWNUB(
            in_channels=n_ftrs,
            out_channels=3,
            kernel_size=3,
            height=self.uv_size,
            width=self.uv_size,
            padding=1,
        )

        self.tex_conv = nn.Sequential(
            # NOTE: if we are using 4x4 convs, it makes more sense to use glorot (!)
            UNetWB(in_channels=n_ftrs, out_channels=8, n_init_ftrs=16, size=uv_size, out_scale=0.1),
            nn.LeakyReLU(0.2, inplace=True),
            la.Conv2dWNUB(
                in_channels=8,
                out_channels=3,
                kernel_size=3,
                height=self.uv_size,
                width=self.uv_size,
                padding=1,
            ),
        )

        # self.apply(lambda x: la.glorot(x, 0.2))
        self.apply(weights_initializer(0.2))
        self.tex_conv[-1].apply(weights_initializer(1.0))
        self.verts_conv.apply(weights_initializer(1.0))

    def forward(self, normalized_input,):
        '''

        :param normals: normalized coordinates in uv space
        :return:
        '''

        normals = normalized_input["normalized_uv"]

        B = normals.shape[0]

        normals_ftrs = self.normals_conv(normals)
        verts_ftrs = self.verts_unet(normals_ftrs)
        verts_uv_delta = self.verts_conv(verts_ftrs)

        tex_mean_rec = self.tex_conv(normals_ftrs)

        output = {
            "verts_uv_delta": verts_uv_delta,
            "tex_mean_rec": tex_mean_rec,
            "normals_ftrs": normals_ftrs,
            "verts_ftrs": verts_ftrs,
        }
        output.update(normalized_input)
        return output

class GeoTexUNet(nn.Module):
    def __init__(
        self,
        geo_fn, # TODO: ?
        face_cond_mask, # TODO: ?
        pose_cond_mask, # TODO: ?
        uv_size=1024,
        n_face_embs=256,
        n_ftrs=4,
    ):
        super().__init__()
        # TODO: can we just condition on the output of the face decoder?

        self.geo_fn = geo_fn

        self.uv_size = uv_size

        # converting face_embs to
        self.face_embs_to_conv = nn.Sequential(
            la.LinearWN(n_face_embs, 4 * 4 * 64), nn.LeakyReLU(0.2, inplace=True)
        )
        self.face_conv = nn.Sequential(
            UpConvBlockDeep(64, 64, 8),
            UpConvBlockDeep(64, 64, 16),
            UpConvBlockDeep(64, 64, 32),
            UpConvBlockDeep(64, 32, 64),
            UpConvBlockDeep(32, 16, 128),
            UpConvBlockDeep(16, 8, 256),
            UpConvBlockDeep(8, 8, 512),
            la.Conv2dWNUB(
                in_channels=8,
                out_channels=n_ftrs,
                kernel_size=1,
                height=512,
                width=512,
                padding=0,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.normals_conv = nn.Sequential(
            la.Conv2dWNUB(
                in_channels=3,
                out_channels=n_ftrs,
                kernel_size=3,
                height=self.uv_size,
                width=self.uv_size,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.pose_conv = nn.Sequential(
            la.Conv2dWNUB(
                in_channels=3,
                out_channels=n_ftrs,
                kernel_size=3,
                height=self.uv_size,
                width=self.uv_size,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.verts_unet = nn.Sequential(
            UNetWB(n_ftrs + n_ftrs, 4, n_init_ftrs=8, size=1024, out_scale=0.1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.verts_conv = la.Conv2dWNUB(
            in_channels=4,
            out_channels=3,
            kernel_size=3,
            height=self.uv_size,
            width=self.uv_size,
            padding=1,
        )

        self.tex_conv = nn.Sequential(
            # NOTE: if we are using 4x4 convs, it makes more sense to use glorot (!)
            UNetWB(n_ftrs + n_ftrs, 8, n_init_ftrs=16, size=1024, out_scale=0.1),
            nn.LeakyReLU(0.2, inplace=True),
            la.Conv2dWNUB(
                in_channels=8,
                out_channels=3,
                kernel_size=3,
                height=self.uv_size,
                width=self.uv_size,
                padding=1,
            ),
        )

        self.register_buffer("face_cond_mask", face_cond_mask)
        self.register_buffer("pose_cond_mask", pose_cond_mask)

        # self.apply(lambda x: la.glorot(x, 0.2))
        self.apply(weights_initializer(0.2))
        self.tex_conv[-1].apply(weights_initializer(1.0))
        self.verts_conv.apply(weights_initializer(1.0))

    def forward(self, normals, pose, face_embs):

        B = normals.shape[0]

        pose_ftrs = self.pose_conv(pose)

        face_embs_conv = self.face_embs_to_conv(face_embs).reshape(-1, 64, 4, 4)
        face_embs_conv = self.face_conv(face_embs_conv)

        normals_ftrs = self.normals_conv(normals)

        face_ftrs = th.zeros(
            (B, 4, self.uv_size, self.uv_size),
            dtype=face_embs_conv.dtype,
            device=face_embs_conv.device,
        )
        face_ftrs[:, :, self.uv_size // 2 :, : self.uv_size // 2] = face_embs_conv

        joint_ftrs = (
            self.face_cond_mask * face_ftrs + (1.0 - self.face_cond_mask) * normals_ftrs
        )
        joint_ftrs = th.cat([joint_ftrs, pose_ftrs], dim=1)

        verts_ftrs = self.verts_unet(joint_ftrs)
        verts_uv_delta = self.verts_conv(verts_ftrs)
        tex_mean_rec = self.tex_conv(joint_ftrs)

        verts_delta_rec = self.geo_fn.from_uv(verts_uv_delta)

        return {
            "verts_delta_rec": verts_delta_rec,
            "verts_uv_delta": verts_uv_delta,
            "tex_mean_rec": tex_mean_rec,
            "face_ftrs": face_ftrs,
            "normals_ftrs": normals_ftrs,
            "verts_ftrs": verts_ftrs,
        }


class ViewTexUNet(nn.Module):
    def __init__(self, geo_fn, uv_size, seam_sampler):
        super().__init__()

        self.geo_fn = geo_fn

        self.seam_sampler = seam_sampler

        self.unet = UNetWB(3 + 3, 3, n_init_ftrs=8, size=uv_size)

    def forward(self, verts_rec, tex_mean_rec, camera_pos):

        # computing view cosine
        with th.no_grad():
            # NOTE: this we can do once?
            mask = (self.geo_fn.index_image != -1).any(dim=-1)
            # TODO: if we want to subsample?
            idxs = self.geo_fn.index_image[mask]
            tri_uv = self.geo_fn.uv_coords[self.geo_fn.uv_mapping[idxs, 0]]
            tri_xyz = verts_rec[:, idxs]
            # TODO: use this as a rotation?
            t, b, n = compute_tbn_uv(tri_xyz, tri_uv)
            tbn_rot = th.stack((t, -b, n), dim=-2)
            B = verts_rec.shape[0]
            tbn_rot_uv = th.zeros(
                (B, self.geo_fn.uv_size, self.geo_fn.uv_size, 3, 3),
                dtype=th.float32,
                device=verts_rec.device,
            )
            tbn_rot_uv[:, mask] = tbn_rot
            view = F.normalize(camera_pos[:, np.newaxis] - verts_rec, dim=-1)
            view_uv = self.geo_fn.to_uv(view)
            view_cond = th.einsum("bhwij,bjhw->bihw", tbn_rot_uv, view_uv)

            view_cond = self.seam_sampler.impaint(view_cond)

        tex_cond = tex_mean_rec
        cond = th.cat([view_cond, tex_cond], dim=1)
        tex_view = self.unet(cond)
        return {"tex_view_rec": tex_view}


class ViewVAE(nn.Module):
    def __init__(
        self,
        dataset,
        lbs_fn,
        face_region_path,
        pose_regions_path,
        seam_data_paths,
        base_uv_size=1024,
        rendering_enabled=True,
        fake_dof_enabled=True,
    ):
        super().__init__()

        self.geo_fn = GeometryModule(
            dataset.uv_coords,
            dataset.uv_faces,
            dataset.uv_mapping,
            dataset.faces,
            dataset.uv_size,
        )

        self.seam_sampler = SeamSampler(seam_data_paths[1024])
        self.seam_sampler_2k = SeamSampler(seam_data_paths[2048])

        pose_regions = th.as_tensor(np.load(pose_regions_path)[6:], dtype=th.float32)
        self.register_buffer("pose_regions", pose_regions)

        pose_regions_uv = self.geo_fn.to_uv(
            pose_regions[..., np.newaxis].permute(2, 1, 0).cuda()
        ).cpu()
        pose_regions_uv = self.seam_sampler.impaint(pose_regions_uv)
        pose_cond_mask = (
            F.interpolate(pose_regions_uv, size=(256, 256), mode="nearest") > 0.1
        ).to(th.float32)

        assert base_uv_size == 1024

        self.base_uv_size = base_uv_size

        # loading deps
        logger.info(f"using face region from {face_region_path}")
        face_cond_mask = F.interpolate(
            th.as_tensor(np.load(face_region_path), dtype=th.float32)[
                np.newaxis, np.newaxis
            ],
            (self.base_uv_size, self.base_uv_size),
        ).to(th.int32)

        tex_mean = F.interpolate(
            th.as_tensor(dataset.tex_mean)[np.newaxis], scale_factor=2, mode="bilinear"
        )[0]
        self.register_buffer("tex_mean", tex_mean)
        self.register_buffer("tex_std", th.as_tensor(dataset.tex_std))

        self.geo_fn = GeometryModule(
            dataset.uv_coords,
            dataset.uv_faces,
            dataset.uv_mapping,
            dataset.faces,
            dataset.uv_size,
        )

        self.lbs_fn = LBSModule(
            **lbs_fn,
            lbs_scale=dataset.lbs_scale,
            lbs_template_verts=dataset.lbs_template_verts,
            global_scaling=dataset.global_scaling,
            # TODO: this should be the same with the template
            verts_unposed_mean=dataset.verts_unposed_mean,
        )

        self.lbs_fn_pt = LBSModuleOld(
            **lbs_fn,
            lbs_scale=dataset.lbs_scale,
            lbs_template_verts=dataset.lbs_template_verts,
            global_scaling=dataset.global_scaling,
            # TODO: this should be the same with the template
            verts_unposed_mean=dataset.verts_unposed_mean,
        )

        # produces geometry displacement and view-independent texture
        self.decoder = GeoTexUNet(
            self.geo_fn, face_cond_mask, pose_cond_mask, self.base_uv_size
        )

        self.decoder_view = ViewTexUNet(
            self.geo_fn, self.base_uv_size, self.seam_sampler
        )

        self.upscale_net = UpscaleNet(
            in_channels=6, size=1024, upscale_factor=2, n_ftrs=8
        )

        self.shadow_net = ShadowUNet(
            uv_size=2048,
            shadow_size=256,
            ao_mean=dataset.ao_mean,
            n_dims=4,
            interp_mode="bilinear",
            biases=False,
        )

        self.camera_calibration_net = CameraCalibrationNetV2(dataset.cameras_ids)

        self.rendering_enabled = rendering_enabled

        if self.rendering_enabled:
            self.image_height = dataset.image_height
            self.image_width = dataset.image_width

            self.renderer = RenderLayer(
                h=self.image_height,
                w=self.image_width,
                vt=self.geo_fn.uv_coords,
                vi=self.geo_fn.faces,
                vti=self.geo_fn.uv_faces,
                flip_uvs=False,
            )

        self.fake_dof_enabled = fake_dof_enabled
        if self.fake_dof_enabled:
            self.fake_dof_body = DOF(list(dataset.cameras.keys()))

    def _render(
        self,
        verts,
        tex,
        intrinsics3x3,
        extrinsics3x4,
        capture_bg_image,
        camera_id,
        depth_inf=1e7,
    ):
        B = verts.shape[0]

        # TODO: we should fill the background (!)
        # camera = convert_camera_parameters(extrinsics3x4, intrinsics3x3)

        if capture_bg_image is None:
            capture_bg_image = th.zeros(
                (B, 4, self.image_height, self.image_width),
                dtype=th.float32,
                device=verts.device,
            )
        else:
            bg = th.zeros_like(capture_bg_image[:, :1])
            capture_bg_image = th.cat([capture_bg_image, bg], dim=1)

        tex_seg = th.ones_like(tex[:, :1])
        tex_rgb_seg = th.cat([tex, tex_seg], dim=1)

        with th.no_grad():
            faces = self.geo_fn.faces.to(th.int64)
            vn = vert_normals_v2(verts, faces)
            vn = (-vn + 1.0) / 2

        rendered_rgb = self.renderer(
            verts,
            tex_rgb_seg,
            Rt=extrinsics3x4,
            K=intrinsics3x3,
            vn=vn,
            background=capture_bg_image,
            output_filters=[
                "render",
                "depth_img",
                "mask",
                "alpha",
                "vn_img",
                "index_img",
                "bary_img",
                "v_pix",
            ],
        )

        rgb_seg = rendered_rgb["render"][:, :4].contiguous()

        if self.training:
            rgb_seg = edge_grad_estimator(
                v_pix=rendered_rgb["v_pix"],
                vi=self.renderer.vi,
                bary_img=rendered_rgb["bary_img"].detach(),
                img=rgb_seg,
                index_img=rendered_rgb["index_img"],
            )

        rgb = rgb_seg[:, :3]
        seg = rgb_seg[:, 3:4]

        depth = rendered_rgb["depth_img"].detach()[:, np.newaxis]
        # NOTE: ignore most of the border pixels (mostly background-related)
        disc_mask = depth_discontuity_mask(depth, threshold=500, pool_ksize=5)

        depth = (
            rendered_rgb["depth_img"]
            + (1.0 - rendered_rgb["mask"].to(th.float32)) * depth_inf
        )

        preds = dict(
            rendered_rgb=rgb,
            rendered_mask=seg,
            rendered_normals=rendered_rgb["vn_img"],
            depth=depth,
            depth_disc_mask=disc_mask,
        )

        if self.fake_dof_enabled:
            # compute dof stuff here
            preds["rendered_rgb"] = self.fake_dof_body(
                img=preds["rendered_rgb"],
                depth=rendered_rgb["depth_img"].detach(),
                bkg=capture_bg_image[:, :3],
                mask=rendered_rgb["mask"][:, np.newaxis].to(th.float32).detach(),
                cam_idxs=camera_id,
            )

            preds["dof_reg_loss"] = self.fake_dof_body.get_regularization_loss(
                camera_id
            )

        return preds

    def cond_normals(self, verts, R_root_inv):
        # for lbs: `lbs_verts = self.lbs_fn.template_pose(motion)`
        with th.no_grad():
            verts_norm = (verts / self.lbs_fn.global_scaling).bmm(R_root_inv)
            vn_norm = vert_normals_v2(verts_norm, self.geo_fn.faces.to(th.long))
            vn_norm = (vn_norm + 1.0) / 2.0
            vn_norm_uv = self.geo_fn.to_uv(vn_norm)
            vn_norm_uv = self.seam_sampler.impaint(vn_norm_uv)
        return vn_norm_uv

    def cond_pose(self, motion):

        B = motion.shape[0]

        with th.no_grad():

            lbs_verts, _, _, state_t, _, _ = self.lbs_fn.lbs_fn(
                motion,
                scales=self.lbs_fn.lbs_scale.expand(B, -1),
                # TODO: this should probably be more smooth?
                vertices=self.lbs_fn.lbs_template_verts,
            )

            root_pos = state_t[:, 1]

            # NOTE: this we can do once?
            mask = (self.geo_fn.index_image != -1).any(dim=-1)
            # TODO: if we want to subsample?
            idxs = self.geo_fn.index_image[mask]
            tri_uv = self.geo_fn.uv_coords[self.geo_fn.uv_mapping[idxs, 0]]
            tri_xyz = lbs_verts[:, idxs]
            # TODO: use this as a rotation?
            t, b, n = compute_tbn_uv(tri_xyz, tri_uv)
            tbn_rot = th.stack((t, -b, n), dim=-2)
            tbn_rot_uv = th.zeros(
                (B, self.geo_fn.uv_size, self.geo_fn.uv_size, 3, 3),
                dtype=th.float32,
                device=lbs_verts.device,
            )
            tbn_rot_uv[:, mask] = tbn_rot
            root_view = F.normalize(root_pos[:, np.newaxis] - lbs_verts, dim=-1)
            root_view_uv = self.geo_fn.to_uv(root_view)
            # TODO: we should probably try adding some sort of distance?
            # TODO: or, maybe orientation to two closest neigbhours + one-hot encoding?
            root_view_cond = th.einsum("bhwij,bjhw->bihw", tbn_rot_uv, root_view_uv)
            root_view_cond = self.seam_sampler.impaint(root_view_cond)

        return root_view_cond

    def forward_tex(
        self,
        tex_mean_rec,
        tex_view_rec,
        shadow_map,
        camera_id,
    ):
        # TODO: should this be a UNet instead?
        x = th.cat([tex_mean_rec, tex_view_rec], dim=1)
        # TODO: maybe we should try directly predicting the value instead of a delta?
        tex_rec = tex_mean_rec + tex_view_rec

        tex_rec = self.seam_sampler.impaint(tex_rec)
        tex_rec = self.seam_sampler.resample(tex_rec)

        tex_rec = F.interpolate(
            tex_rec, size=(2048, 2048), mode="bilinear", align_corners=False
        )
        tex_rec = tex_rec + self.upscale_net(x)

        tex_rec = tex_rec * self.tex_std + self.tex_mean
        tex_rec = tex_rec * shadow_map

        tex_rec = self.seam_sampler_2k.impaint(tex_rec)
        tex_rec = self.seam_sampler_2k.resample(tex_rec)
        tex_rec = self.seam_sampler_2k.resample(tex_rec)

        if camera_id is not None:
            tex_rec = self.camera_calibration_net(tex_rec, camera_id)

        tex_rec = tex_rec.clamp_(min=0, max=255)
        return tex_rec

    def forward(
        self,
        motion,
        ao,
        face_embs,
        camera_pos,
        verts=None,
        camera_id=None,
        intrinsics3x3=None,
        extrinsics3x4=None,
        capture_bg_image=None,
        **kwargs,
    ):
        preds = {}

        # TODO: see if we need any additional
        with th.no_grad():
            R_root_inv = self.lbs_fn.compute_root_rotation(motion).permute(0, 2, 1)
            normals_uv = self.cond_normals(verts, R_root_inv)
            pose_uv = self.cond_pose(motion)
            # verts_lbs = self.lbs_fn.template_pose(motion)
            # TODO: should we also provide lbs normals?
            # normals_uv_lbs = self.cond_normals(verts_lbs, R_root_inv).detach()

        dec_preds = self.decoder(normals_uv, pose_uv, face_embs)

        verts_rec = self.lbs_fn_pt.pose(dec_preds["verts_delta_rec"], motion)

        shadow_preds = self.shadow_net(ao_map=ao)

        dec_view_preds = self.decoder_view(
            verts_rec=verts_rec,
            tex_mean_rec=dec_preds["tex_mean_rec"],
            camera_pos=camera_pos,
        )

        tex_rec = self.forward_tex(
            dec_preds["tex_mean_rec"],
            dec_view_preds["tex_view_rec"],
            shadow_map=shadow_preds["shadow_map"],
            camera_id=camera_id,
        )

        preds.update(
            verts=verts_rec,
            tex_rec=tex_rec,
            **dec_preds,
            **shadow_preds,
            **dec_view_preds,
        )

        if self.rendering_enabled:
            render_preds = self._render(
                verts_rec,
                tex_rec,
                camera_id=camera_id,
                intrinsics3x3=intrinsics3x3,
                extrinsics3x4=extrinsics3x4,
                capture_bg_image=capture_bg_image,
            )
            preds.update(**render_preds)

        return preds