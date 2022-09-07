import numpy as np
import os
import pickle

import torch
import torch.utils.data

from src import pyutils
from src.pyutils import camera_layers

import cv2

from src.kinematic_modeling.lbs.locator_keypoint import Locators

import logging

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    """Custom dataset implementation."""

    def __init__(
            self,
            uv_size,
            frame_list_path,
            motion_path,
            clothes_reference_mesh_path,
            clothes_reference_uv_mapping_path,
            clothes_nbs_idxs_path,
            clothes_nbs_weights_path,
            clothes_geometry_mesh_path,

            frame_drop_ok=False,
            motion_velocity_path=None,
            clothes_lbs_template_path=None,
            clothes_lbs_scale_path=None,
            frame_lookup_path=None,
            motion_length=None,
            global_scaling=None,
            transform_path=None,
            kpts_3d_path=None,
            locator_path=None,
            body_verts_unposed_mean_path=None,
            clothes_verts_unposed_mean_path=None,
            # image related
            inverse_rendering=False,
            krt_path=None,
            cameras=None,
            valid_cameras=None,
            sample_cameras=False,
            camera_id=None,
            image_path=None,
            image_mask_path=None,
            image_part_mask_path=None,
            bad_image_part_mask_view_list=None,
            ignore_masks_path=None,
            background_path=None,
            image_height=None,
            image_width=None,
            image_ds_rate=None,
            # texture
            tex_mean_path=None,
            tex_std_path=None,
            tex_path=None,
            tex_mask_path=None,
            # ao texture
            clothes_ao_mean_path=None,
            precomputed_body_ao_path=None,
            precomputed_clothes_ao_path=None,
            pose_sequence_length=None,
            frame_interval=2,
            latent_code_path=None,
            latent_code_mean_path=None,
            pose_encoding_normal_path=None,
            camera_render_mode=None,
            camera_render_zoom_speed=0.003,
            camera_render_rotate_speed=None,
            **kwargs
    ):
        self.uv_size = uv_size

        self.motion_length = motion_length
        self.global_scaling = global_scaling

        self.kpts_3d_path = kpts_3d_path
        self.face_kpts_3d_path = face_kpts_3d_path
        self.face_kpts_3d_vidxs_path = face_kpts_3d_vidxs_path
        if self.face_kpts_3d_vidxs_path:
            # using these to define the mask (!)
            self.face_kpts_3d_vidxs = np.loadtxt(
                self.face_kpts_3d_vidxs_path, dtype=np.int32
            )
        self.locator_path = locator_path

        self.transform_path = transform_path
        self.motion_path = motion_path
        self.motion_velocity_path = motion_velocity_path

        # TODO: mb we can make this automatic?
        self.frame_list_path = frame_list_path
        self.body_reference_mesh_path = body_reference_mesh_path
        self.body_reference_uv_mapping_path = body_reference_uv_mapping_path
        self.body_nbs_idxs_path = body_nbs_idxs_path
        self.body_nbs_weights_path = body_nbs_weights_path
        self.body_lbs_template_path = body_lbs_template_path
        self.body_lbs_scale_path = body_lbs_scale_path
        self.body_geometry_mesh_path = body_geometry_mesh_path
        self.body_verts_unposed_mean_path = body_verts_unposed_mean_path

        self.clothes_reference_mesh_path = clothes_reference_mesh_path
        self.clothes_reference_uv_mapping_path = clothes_reference_uv_mapping_path
        self.clothes_nbs_idxs_path = clothes_nbs_idxs_path
        self.clothes_nbs_weights_path = clothes_nbs_weights_path
        self.clothes_lbs_template_path = clothes_lbs_template_path
        self.clothes_lbs_scale_path = clothes_lbs_scale_path
        self.clothes_geometry_mesh_path = clothes_geometry_mesh_path
        self.clothes_verts_unposed_mean_path = clothes_verts_unposed_mean_path

        if camera_render_mode:
            self.camera_render_mode = camera_render_mode
            self.camera_render_zoom_speed = camera_render_zoom_speed
            self.camera_render_rotate_speed = camera_render_rotate_speed
        else:
            self.camera_render_mode = 0

        # image things
        # assert image_ds_rate == 1, "downsampling rate != 1 is not supported yet"
        # images
        self.image_path = image_path
        self.image_mask_path = image_mask_path
        self.image_part_mask_path = image_part_mask_path
        self.bad_image_part_mask_view_list = bad_image_part_mask_view_list
        self.ignore_masks_path = ignore_masks_path
        self.background_path = background_path

        self.image_ds_rate = image_ds_rate
        self.image_height = image_height // image_ds_rate
        self.image_width = image_width // image_ds_rate

        self.inverse_rendering = inverse_rendering
        self.krt_path = krt_path
        self.valid_cameras = valid_cameras

        self.camera_id = camera_id
        self.sample_cameras = sample_cameras
        self.pose_sequence_length = pose_sequence_length
        self.frame_interval = frame_interval

        self.latent_code_path = latent_code_path
        self.latent_code_mean_path = latent_code_mean_path

        self.pose_encoding_normal_path = pose_encoding_normal_path
        self.frame_drop_ok = frame_drop_ok

        if cameras is None:
            self.cameras = pyutils.io.load_krt(self.krt_path)
            # filtering the cameras
            if self.valid_cameras is not None:
                with open(self.valid_cameras) as f:
                    lines = f.readlines()
                    lines = [line.rstrip() for line in lines]
                    self.valid_camera_prefix = lines
                logger.info(
                    f"only loading cameras with prefixes: {self.valid_camera_prefix}"
                )
                self.cameras = {
                    cid: camera
                    for cid, camera in self.cameras.items()
                    if any(cid.startswith(p) for p in self.valid_camera_prefix)
                }
                logger.info(f"loaded {len(self.cameras)} cameras")
            # post-prorcessing cameras
            for index, (cid, camera) in enumerate(self.cameras.items()):
                camera["index"] = index
                camera["position"] = -np.dot(
                    camera["extrin"][:3, :3].T, camera["extrin"][:3, 3]
                )
                camera["intrin"] /= image_ds_rate
                camera["intrin"][2, 2] = 1.0

            if not camera_id and not sample_cameras:
                raise ValueError(
                    "you should either sample cameras or specify a single one"
                )

            if self.ignore_masks_path:
                logger.info("reading ignore masks")

                for cid, camera in self.cameras.items():
                    path = self.ignore_masks_path.format(camera=cid)
                    if os.path.exists(path):
                        camera["ignore_mask"] = (
                                pyutils.io.imread(
                                    path, (self.image_height, self.image_width)
                                )[0]
                                / 255.0
                        )
                    else:
                        camera["ignore_mask"] = np.zeros(
                            (self.image_height, self.image_width), dtype=np.float32
                        )
                logger.info("done!")

            if self.background_path:
                logger.info("reading background images")
                for cid, camera in self.cameras.items():
                    path = self.background_path.format(camera=cid)
                    camera["background_image"] = pyutils.io.imread(
                        path, (self.image_height, self.image_width),
                    )
                logger.info("done!")
        else:
            self.cameras = cameras
        self.cameras_ids = list(self.cameras.keys())

        self.tex_mean_path = tex_mean_path
        self.tex_std_path = tex_std_path
        self.tex_path = tex_path
        self.tex_mask_path = tex_mask_path

        # loading the cameras if necessary
        # TODO: we need a unified way to do these things?
        # if isinstance(cameras, str):
        #     # TODO: check if the order is consistent here
        #     cameras = list(pyutils.io.load_krt(cameras).keys())
        if self.tex_mean_path:
            tex_mean = cv2.cvtColor(cv2.imread(tex_mean_path), cv2.COLOR_BGR2RGB).astype(np.float32)
            self.body_tex_mean = cv2.resize(tex_mean[:1024, :1024, :], (self.uv_size, self.uv_size)).transpose(
                (2, 0, 1))
            self.clothes_tex_mean = cv2.resize(tex_mean[1024:, 1024:, :], (self.uv_size, self.uv_size)).transpose(
                (2, 0, 1))

        self.body_tex_std = (
            float(np.genfromtxt(self.tex_std_path) ** 0.5)
            if self.tex_std_path
            else 64.0
        )
        self.clothes_tex_std = self.body_tex_std

        self.body_ao_mean_path = body_ao_mean_path
        self.clothes_ao_mean_path = clothes_ao_mean_path
        if self.body_ao_mean_path:
            self.body_ao_mean = (
                    pyutils.io.imread(
                        self.body_ao_mean_path,
                        dst_size=(self.uv_size, self.uv_size),
                        dtype=np.float32,
                    )[:1]
                    / 255.0
            )
        else:
            self.body_ao_mean = np.zeros([1, self.uv_size, self.uv_size], dtype=np.float32)

        if self.clothes_ao_mean_path:
            self.clothes_ao_mean = (
                    pyutils.io.imread(
                        self.clothes_ao_mean_path,
                        dst_size=(self.uv_size, self.uv_size),
                        dtype=np.float32,
                    )[:1]
                    / 255.0
            )
        else:
            self.clothes_ao_mean = np.zeros([1, self.uv_size, self.uv_size], dtype=np.float32)

        self.precomputed_body_ao_path = precomputed_body_ao_path
        self.precomputed_clothes_ao_path = precomputed_clothes_ao_path

        # embeddings
        self.face_embs_path = face_embs_path
        self.frame_lookup_path = frame_lookup_path

        # loading the frames
        # NOTE: these
        self.frame_list = np.genfromtxt(self.frame_list_path, dtype=np.str)
        # TODO: generate the ids in advance?
        self.num_frames = len(self.frame_list)

        if len(self.frame_list.shape) == 1:
            logger.warning("frame list does not contain sequence numbers, adding fake")
            self.frame_list = np.stack(
                [np.array(["unknown"] * self.num_frames), self.frame_list], axis=1
            )

        if self.face_embs_path is not None:
            self.face_embs = np.load(face_embs_path)
            assert (
                    frame_lookup_path is not None
            ), "you should also specify a frame lookup table"
        if self.frame_lookup_path is not None:
            with open(self.frame_lookup_path, "rb") as fh:
                self.frame_lookup = pickle.load(fh)

        # body
        if self.body_lbs_template_path is not None:
            self.body_lbs_template_verts, _ = pyutils.io.load_ply(self.body_lbs_template_path)
            self.body_lbs_template_verts = np.asarray(
                self.body_lbs_template_verts, dtype=np.float32
            )

        if self.body_lbs_scale_path is not None:
            self.body_lbs_scale = np.loadtxt(self.body_lbs_scale_path).astype(np.float32)
            self.body_lbs_scale = self.body_lbs_scale[np.newaxis]
            if len(self.body_lbs_scale.shape) == 3:
                self.body_lbs_scale = self.body_lbs_scale[:, 0, :]

        self.body_verts_unposed_mean = None
        if self.body_verts_unposed_mean_path:
            body_verts_unposed_mean, _ = pyutils.io.load_ply(self.body_verts_unposed_mean_path)
            self.body_verts_unposed_mean = np.asarray(body_verts_unposed_mean, dtype=np.float32)

        # NOTE: are these used in the model or the dataset?
        # reading uv reference
        # body
        # logger.info(f"loading pre-computed neighbors")
        # self.body_nbs_idxs = np.loadtxt(self.body_nbs_idxs_path).astype(np.int64)
        # self.body_nbs_weights = np.loadtxt(self.body_nbs_weights_path).astype(np.float32)
        # logger.info("done!")
        #
        # logger.info(f"loading the reference mesh")
        #
        # _, uv_coords, faces, uv_faces = map(
        #     np.array, pyutils.io.load_obj(self.body_reference_mesh_path)
        # )
        # self.body_faces = faces.astype(np.int32)
        # self.body_uv_coords = uv_coords.astype(np.float32)
        # self.body_uv_faces = uv_faces.astype(np.int32)
        # self.body_uv_mapping = np.loadtxt(self.body_reference_uv_mapping_path).astype(np.int64)

        # clothes
        if self.clothes_lbs_template_path is not None:
            self.clothes_lbs_template_verts, _ = pyutils.io.load_ply(self.clothes_lbs_template_path)
            self.clothes_lbs_template_verts = np.asarray(
                self.clothes_lbs_template_verts, dtype=np.float32
            )

        if self.clothes_lbs_scale_path is not None:
            self.clothes_lbs_scale = np.loadtxt(self.clothes_lbs_scale_path).astype(np.float32)
            self.clothes_lbs_scale = self.clothes_lbs_scale[np.newaxis]
            if len(self.clothes_lbs_scale.shape) == 3:
                self.clothes_lbs_scale = self.clothes_lbs_scale[:, 0, :]

        self.clothes_verts_unposed_mean = None
        if self.clothes_verts_unposed_mean_path:
            clothes_verts_unposed_mean, _ = pyutils.io.load_ply(self.clothes_verts_unposed_mean_path)
            self.clothes_verts_unposed_mean = np.asarray(clothes_verts_unposed_mean, dtype=np.float32)

        self.clothes_nbs_idxs = np.loadtxt(self.clothes_nbs_idxs_path).astype(np.int64)
        self.clothes_nbs_weights = np.loadtxt(self.clothes_nbs_weights_path).astype(np.float32)

        logger.info(f"loading the reference mesh")

        _, uv_coords, faces, uv_faces = map(
            np.array, pyutils.io.load_obj(self.clothes_reference_mesh_path)
        )
        self.clothes_faces = faces.astype(np.int32)
        self.clothes_uv_coords = uv_coords.astype(np.float32)
        self.clothes_uv_faces = uv_faces.astype(np.int32)
        self.clothes_uv_mapping = np.loadtxt(self.clothes_reference_uv_mapping_path).astype(np.int64)

        logger.info(f"done!")

        # for 3D rot-center keypoints
        if self.kpts_3d_path and self.locator_path:
            locator = Locators()
            logger.info(f"loading all rot-center keypoints {self.num_frames} frames")
            self.kpts_3d = locator.loadBatchKeypointsAndLocatorJson(
                self.locator_path, self.kpts_3d_path, self.frame_list[:, 1]
            )
            logger.info("done!")

        self.rotate_segmentation = 'rotate_segmentation' in kwargs and kwargs['rotate_segmentation']
        if self.rotate_segmentation:
            logger.warning("Temporarily rotating images according to He Wen's segmentation image")
            self.segmentation_rotate_dict = {}
            with open('/mnt/home/donglaix/data/segmentationRotation.txt') as f:
                for line in f:
                    s = line.split()
                    assert s[1] in ('True', 'False')
                    self.segmentation_rotate_dict[s[0]] = s[1] == 'True'

        if self.latent_code_mean_path is not None:
            self.latent_code_mean = pyutils.tensorIO.ReadArrayFromBinaryFile(self.latent_code_mean_path).astype(
                np.float32)

        if self.camera_render_mode > 0:
            # HACK hardcoded here only for Sociopticon
            self.renderscl = 1.0
            # self.rotaxle = torch.from_numpy(np.array([0.007813,-0.975712, -0.218917], dtype=np.float32))
            self.rotaxle = torch.from_numpy(
                np.array([0.001021, -0.981891, -0.189446], dtype=np.float32)
            )
            # self.centroid = torch.from_numpy(np.array([-102.182884, 208.875565, 2909.277832],dtype=np.float32))
            self.centroid = torch.from_numpy(
                np.array([-38.1118, -327.7896, 2761.0889], dtype=np.float32)
            )

            if self.camera_render_mode == 1 and self.camera_render_rotate_speed is None:
                self.camera_render_rotate_speed = 1.0
            elif self.camera_render_rotate_speed is None:
                self.camera_render_rotate_speed = 0.005

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        """Get the next sample.

        Args:
            idx: integer index of the sample

        Returns:
            a dict of numpy tensors which are then passed to the model

        """

        seq, frame = self.frame_list[idx]

        # by default we just have a fixed camera id
        camera_id = self.camera_id
        if self.sample_cameras:
            camera_id = np.random.choice(self.cameras_ids)

        fmts = dict(frame=int(frame), camera=camera_id, seq=seq, dataset_idx=idx)

        # motion = np.loadtxt(self.motion_path.format(**fmts)).astype(np.float32)
        if self.pose_sequence_length is None:
            # return pose of that specific frame
            if self.motion_path.endswith('.bt'):
                motion = pyutils.tensorIO.ReadArrayFromBinaryFile(self.motion_path.format(**fmts)).astype(np.float32)
            elif self.motion_path.endswith('.txt'):
                motion = np.loadtxt(self.motion_path.format(**fmts)).astype(np.float32)
        else:
            assert type(self.pose_sequence_length) == int
            motion_list = []
            for i in range(self.pose_sequence_length):
                tmp_fmts = dict(frame=int(frame) - i * self.frame_interval, camera=camera_id, seq=seq)
                motion_path = self.motion_path.format(**tmp_fmts)
                if os.path.isfile(motion_path):
                    if motion_path.endswith('.bt'):
                        motion_list.append(pyutils.tensorIO.ReadArrayFromBinaryFile(motion_path).astype(np.float32))
                    elif motion_path.endswith('.txt'):
                        motion_list.append(np.loadtxt(motion_path).astype(np.float32))
                else:
                    assert i != 0, "The pose file for the target frame must exist!"
                    motion_list.append(motion_list[-1])  # repeat the same
            motion_list.reverse()  # reverse the order
            # [(batch size), pose dimension, num frames]
            motion = np.stack(motion_list, axis=0).transpose(1, 0)

        inputs = dict(motion=motion, frame=frame, seq=seq, dataset_idx=idx)
        targets = dict()

        if self.motion_velocity_path is not None:
            if self.pose_sequence_length is None:
                # return pose of that specific frame
                if self.motion_velocity_path.endswith('.bt'):
                    motion_velocity = pyutils.tensorIO.ReadArrayFromBinaryFile(
                        self.motion_velocity_path.format(**fmts)).astype(np.float32)
                elif self.motion_velocity_path.endswith('.txt'):
                    motion_velocity = np.loadtxt(self.motion_velocity_path.format(**fmts)).astype(np.float32)
            else:
                assert type(self.pose_sequence_length) == int
                motion_velocity_list = []
                for i in range(self.pose_sequence_length):
                    tmp_fmts = dict(frame=int(frame) - i * self.frame_interval, camera=camera_id, seq=seq)
                    motion_velocity_path = self.motion_velocity_path.format(**tmp_fmts)
                    if os.path.isfile(motion_velocity_path):
                        if motion_velocity_path.endswith('.bt'):
                            motion_velocity_list.append(
                                pyutils.tensorIO.ReadArrayFromBinaryFile(motion_velocity_path).astype(np.float32))
                        elif motion_path.endswith('.txt'):
                            motion_velocity_list.append(np.loadtxt(motion_velocity_path).astype(np.float32))
                    else:
                        assert i != 0, "The pose file for the target frame must exist!"
                        motion_velocity_list.append(motion_list[-1])  # repeat the same
                motion_velocity_list.reverse()  # reverse the order
                # [(batch size), pose dimension, num frames]
                motion_velocity = np.stack(motion_velocity_list, axis=0).transpose(1, 0)
            inputs.update(motion_velocity=motion_velocity)

        if self.body_geometry_mesh_path:
            body_verts, _ = pyutils.io.load_ply(self.body_geometry_mesh_path.format(**fmts))
            body_verts = np.asarray(body_verts, dtype=np.float32)
            inputs.update(body_verts=body_verts)
            targets.update(body_verts=body_verts)

        if self.clothes_geometry_mesh_path:
            clothes_verts, _ = pyutils.io.load_ply(self.clothes_geometry_mesh_path.format(**fmts))
            clothes_verts = np.asarray(clothes_verts, dtype=np.float32)
            inputs.update(clothes_verts=clothes_verts)
            targets.update(clothes_verts=clothes_verts)

        # if self.transform_path:
        #     transf = np.genfromtxt(self.transform_path.format(**fmts)).astype(
        #         np.float32
        #     )
        #     transfmat = transf.astype(np.float32)
        #     transfmat[:3, :3] = transf[:3, :3].T
        #     transfmat[:3, 3] = -np.dot(
        #         transf[:3, :3].T, transf[:3, 3]
        #     )  # doing inverse now
        #     inputs.update(transfmat=transfmat)

        if self.transform_path:
            transf = np.genfromtxt(self.transform_path.format(**fmts)).astype(
                np.float32
            )
        else:
            transf = np.identity(4).astype(np.float32)

        transfmat = transf.astype(np.float32)
        transfmat[:3, :3] = transf[:3, :3].T
        transfmat[:3, 3] = -np.dot(
            transf[:3, :3].T, transf[:3, 3]
        )  # doing inverse now
        inputs.update(transfmat=transfmat)

        # loading calibration
        # TODO: should we take into account downsampling here?
        # if self.cameras is not None:
        #     inputs.update(
        #         camera_id=camera_id,
        #         intrinsics3x3=self.cameras[camera_id]["intrin"].astype(np.float32),
        #         extrinsics3x4=self.cameras[camera_id]["extrin"].astype(np.float32),
        #         # TODO: maybe this can be done in the model itself?
        #         camera_pos=self.cameras[camera_id]["position"].astype(np.float32),
        #     )

        # loading calibration
        # TODO: should we take into account downsampling here?
        if self.cameras is not None and camera_id is not None:

            newintmat = torch.from_numpy(
                self.cameras[camera_id]["intrin"].astype(np.float32)
            )
            newextmat = torch.from_numpy(
                self.cameras[camera_id]["extrin"].astype(np.float32)
            )
            newviewpt = torch.from_numpy(
                self.cameras[camera_id]["position"].astype(np.float32)
            )

            if self.camera_render_mode > 0:
                ##########################################
                tensor_transf = torch.from_numpy(transf)
                camrot = camera_layers.RotateCameraLayerModule(newintmat, newextmat)
                rotaxle = self.rotaxle  # -self.rotaxle*200.0
                # rotctr = - tensor_transf[:3, :3].t().mv(tensor_transf[:3, 3]) -self.rotaxle*200.0
                rotctr = self.centroid

                if self.camera_render_mode == 5 or self.camera_render_mode == 7:
                    # renderidx = np.cos(idx * 0.003) * 300.
                    renderidx = np.sin(idx * 0.003) * 150.0
                    # renderidx = np.sin(idx * 0.005) * 50.
                else:
                    renderidx = idx * self.camera_render_rotate_speed

                if (
                        self.camera_render_mode % 2 == 0
                ):  # 0 is not rotating, and 1 is rotating
                    renderidx = 0

                if self.camera_render_mode >= 2:  # 2 and 3 is zooming in
                    # renderscale = 0.3 + 0.7 * (1.0+np.cos(idx * 0.003))/2
                    if self.camera_render_mode == 7:
                        renderscale = (
                                0.25
                                + 0.15
                                * (1.0 + np.cos(idx * self.camera_render_zoom_speed))
                                / 2
                        )
                        # renderscale = 0.2 + 0.05 * (1.0+np.cos(idx * 0.003))/2
                    else:
                        renderscale = (
                                0.7
                                + 0.3
                                * (1.0 + np.cos(idx * self.camera_render_zoom_speed))
                                / 2
                        )
                    # renderscale = 0.6 + 0.4 * (1.0+np.cos(renderidx * 0.001))/2
                else:
                    renderscale = 1.0
                # renderscale = 0.6 + 0.4 * (1.0-idx * 0.1)/2
                # if renderscale< 0.3:
                #   renderscale = 0.3

                newintmat, newextmat = camrot(
                    rotaxle, rotctr, renderidx, renderscale
                )
                newcamctr = -newextmat[:3, :3].t().mv(newextmat[:3, 3])
                newviewpt = tensor_transf[:3, :3].mv(
                    newcamctr + tensor_transf[:3, :3].t().mv(tensor_transf[:3, 3])
                )

            inputs.update(
                camera_id=camera_id,
                intrinsics3x3=newintmat,
                extrinsics3x4=newextmat,
                # TODO: maybe this can be done in the model itself?
                camera_pos=newviewpt,
            )

            # inputs.update(
            #     camera_id=camera_id,
            #     intrinsics3x3=self.cameras[camera_id]["intrin"].astype(np.float32),
            #     extrinsics3x4=self.cameras[camera_id]["extrin"].astype(np.float32),
            #     # TODO: maybe this can be done in the model itself?
            #     camera_pos=self.cameras[camera_id]["position"].astype(np.float32),
            # )

        if self.image_path and (self.image_mask_path or self.image_part_mask_path):
            try:
                capture_image = pyutils.io.imread(
                    self.image_path.format(**fmts),
                    (self.image_height, self.image_width),
                )
                if np.sum(capture_image) < 1.0:
                    logger.warning(
                        f"capture image at {self.image_path.format(**fmts)} is empty"
                    )
                    if not self.frame_drop_ok:
                        raise ValueError("capture image is empty")
                    # TODO: should we return None here?

                # cameras with these prefixes have wrong part segmentation
                bad_image_part_mask_view = False
                if self.bad_image_part_mask_view_list is not None:
                    for bad_image_part_mask_view_prefix in self.bad_image_part_mask_view_list:
                        if camera_id.startswith(bad_image_part_mask_view_prefix):
                            bad_image_part_mask_view = True
                            break

                if not bad_image_part_mask_view and self.image_part_mask_path and os.path.exists(
                        self.image_part_mask_path.format(**fmts)
                ):
                    capture_image_part_mask = pyutils.io.imread(
                        self.image_part_mask_path.format(**fmts),
                        (self.image_height, self.image_width),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    if self.rotate_segmentation:
                        if self.segmentation_rotate_dict[camera_id]:
                            capture_image_part_mask = capture_image_part_mask[:, ::-1, ::-1]

                    capture_image_mask = np.any(capture_image_part_mask != 0, axis=0)[
                        np.newaxis
                    ].astype(np.float32)

                    clothes_mask = np.all(capture_image_part_mask == 1, axis=0)
                    body_mask = np.any(capture_image_part_mask != 0, axis=0) * (~clothes_mask)

                    targets.update(
                        body_mask=body_mask,
                        clothes_mask=clothes_mask,
                        part_mask_valid=np.array(True),
                    )

                elif self.image_mask_path:
                    capture_image_mask = pyutils.io.imread(
                        self.image_mask_path.format(**fmts),
                        (self.image_height, self.image_width),
                    )
                    # TODO: the real bg?
                    capture_image_mask = capture_image_mask[:1] / 255.0

                    targets.update(
                        body_mask=np.ones([self.image_height, self.image_width], dtype=bool),
                        clothes_mask=np.ones([self.image_height, self.image_width], dtype=bool),
                        part_mask_valid=np.array(False),
                    )

                if np.sum(capture_image_mask) < 1.0:
                    logger.warning(f"capture image mask is empty")
                    if not self.frame_drop_ok:
                        raise ValueError("capture image mask is empty")

                capture_bg_image = (
                        capture_image * (1.0 - capture_image_mask)
                        + capture_image_mask
                        * self.cameras[camera_id]["background_image"]
                )

                inputs.update(
                    capture_bg_image=capture_bg_image,
                    ignore_mask=self.cameras[camera_id]["ignore_mask"],
                    view_bg_image=self.cameras[camera_id]["background_image"],
                )

                targets.update(
                    capture_image=capture_image,
                    capture_image_mask=capture_image_mask,
                )
            except (RuntimeError, ValueError):
                logger.info(f"error when reading {self.image_path.format(**fmts)}")
                return None

        if self.tex_path:
            tex = cv2.cvtColor(cv2.imread(self.tex_path.format(**fmts)), cv2.COLOR_BGR2RGB).astype(np.float32)
            body_tex = tex[:1024, :1024, :].transpose((2, 0, 1))
            clothes_tex = tex[1024:, 1024:, :].transpose((2, 0, 1))
            inputs.update(body_tex=body_tex, clothes_tex=clothes_tex)

        if self.tex_mask_path:
            tex_mask = cv2.cvtColor(cv2.imread(self.tex_mask_path.format(**fmts)), cv2.COLOR_BGR2RGB).astype(bool)
            body_tex_mask = (~tex_mask[:1024, :1024, :]).transpose((2, 0, 1)).astype(
                np.float32)  # the default mask has black for valid region
            clothes_tex_mask = (~tex_mask[1024:, 1024:, :]).transpose((2, 0, 1)).astype(
                np.float32)  # the default mask has black for valid region
        else:
            body_tex_mask = np.ones([3, 1024, 1024], dtype=np.float32)
            clothes_tex_mask = np.ones([3, 1024, 1024], dtype=np.float32)
        inputs.update(body_tex_mask=body_tex_mask, clothes_tex_mask=clothes_tex_mask)

        if self.precomputed_body_ao_path:
            precomputed_body_ao = pyutils.io.imread(self.precomputed_body_ao_path.format(**fmts))[:1] / 255.0
            inputs.update(precomputed_body_ao=precomputed_body_ao)

        if self.precomputed_clothes_ao_path:
            precomputed_clothes_ao = pyutils.io.imread(self.precomputed_clothes_ao_path.format(**fmts))[:1] / 255.0
            inputs.update(precomputed_clothes_ao=precomputed_clothes_ao)

        if self.kpts_3d_path:
            targets.update(kpts_3d=self.kpts_3d[int(frame)])

        if self.face_kpts_3d_path:
            face_kpts_3d = np.load(self.face_kpts_3d_path.format(**fmts)).astype(
                np.float32
            )
            face_kpts_3d_valid = (face_kpts_3d != 0.0).all(axis=-1)

            inputs.update(
                face_kpts_3d=face_kpts_3d,
                face_kpts_3d_valid=face_kpts_3d_valid,
            )

        if self.frame_lookup_path:
            # this is a global frame index
            frame_idx = self.frame_lookup[frame]
            inputs.update(face_embs=self.face_embs[frame_idx])

        if self.latent_code_path is not None:
            if self.pose_sequence_length is None:
                latent_code = pyutils.tensorIO.ReadArrayFromBinaryFile(self.latent_code_path.format(**fmts)).astype(
                    np.float32)
            else:
                assert type(self.pose_sequence_length) == int
                latent_code_list = []
                for i in range(self.pose_sequence_length):
                    tmp_fmts = dict(frame=int(frame) - i * self.frame_interval, camera=camera_id, seq=seq)
                    latent_code_path = self.latent_code_path.format(**tmp_fmts)
                    if os.path.isfile(latent_code_path):
                        latent_code_list.append(
                            pyutils.tensorIO.ReadArrayFromBinaryFile(latent_code_path).astype(np.float32))
                    else:
                        if i == 0:
                            logger.warning(
                                "The latent code file for the target frame must exist: {}".format(latent_code_path))
                            return None
                        latent_code_list.append(latent_code_list[-1])  # repeat the same
                latent_code_list.reverse()  # reverse the order
                # [(batch size), pose dimension, num frames, height, width]
                latent_code = np.stack(latent_code_list, axis=0).transpose(1, 0, 2, 3)
            inputs.update(latent_code=latent_code)
            targets.update(latent_code=latent_code)

        if self.latent_code_mean_path is not None:
            inputs.update(latent_code_mean=self.latent_code_mean)

        if self.pose_encoding_normal_path is not None:
            if self.pose_sequence_length is None:
                pose_encoding_normal = pyutils.tensorIO.ReadArrayFromBinaryFile(
                    self.pose_encoding_normal_path.format(**fmts)).astype(np.float32)
            else:
                assert type(self.pose_sequence_length) == int
                pose_encoding_normal_list = []
                for i in range(self.pose_sequence_length):
                    tmp_fmts = dict(frame=int(frame) - i * self.frame_interval, camera=camera_id, seq=seq)
                    pose_encoding_normal_path = self.pose_encoding_normal_path.format(**tmp_fmts)
                    if os.path.isfile(pose_encoding_normal_path):
                        pose_encoding_normal_list.append(
                            pyutils.tensorIO.ReadArrayFromBinaryFile(pose_encoding_normal_path).astype(np.float32))
                    else:
                        if i == 0:
                            logger.warning("The pose encoding file for the target frame must exist: {}".format(
                                pose_encoding_normal_path))
                            return None
                        pose_encoding_normal_list.append(pose_encoding_normal_list[-1])  # repeat the same
                pose_encoding_normal_list.reverse()  # reverse the order
                # [(batch size), num frames, num vertices, 3]
                pose_encoding_normal = np.stack(pose_encoding_normal_list, axis=0)
            inputs.update(pose_encoding_normal=pose_encoding_normal)

        return inputs, targets
