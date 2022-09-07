import numpy as np
import os

import torch as th

import pyutils

# camera related layers for rendering only
from pyutils import camera_layers
from pyutils.io import load_krt

import cv2

from kinematic_modeling.lbs.locator_keypoint import Locators
from pyutils.lmdb_dict import LMDBDict

import logging

logger = logging.getLogger(__name__)


class Dataset(th.utils.data.Dataset):
    """Custom dataset implementation."""

    def __init__(
            self,
            uv_size,
            frame_list_path,
            reference_mesh_path,
            reference_uv_mapping_path,
            nbs_idxs_path,
            nbs_weights_path,
            motion_path,
            geometry_mesh_path,
            lbs_template_path=None,
            lbs_scale_path=None,
            lbs_pos_map_path=None,
            lbs_normal_map_path=None,
            face_embs_path=None,
            frame_lookup_path=None,
            motion_length=None,
            global_scaling=None,
            lbs=None,
            transform_path=None,
            lbs_weights_path=None,
            geometry_uv_path=None,
            # NOTE: it is much easier to just specify formats directly
            lbs_lookatpt_path=None,
            kpts_3d_path=None,
            face_kpts_3d_path=None,
            face_kpts_3d_vidxs_path=None,
            locator_path=None,
            verts_unposed_mean_path=None,
            # image related
            inverse_rendering=False,
            krt_path=None,
            cameras=None,
            valid_camera_prefix=None,
            sample_cameras=False,
            camera_id=None,
            image_path=None,
            image_db_path=None,
            # TODO: add the same to the masks?
            image_mask_path=None,
            image_part_mask_path=None,
            image_mask_db_path=None,
            ignore_masks_path=None,
            ignore_masks_db_path=None,
            background_path=None,
            background_db_path=None,
            embs_db_path=None,
            image_height=None,
            image_width=None,
            image_ds_rate=None,
            # texture
            tex_mean_path=None,
            tex_std_path=None,
            tex_path=None,
            tex_ip_path=None,
            # ao texture
            ao_mean_path=None,
            ao_path=None,
            ao_verts_path=None,
            root_rot_db_path=None,
            embs_seq_path=None,
            motion_seq_path=None,
            face_embs_seq_path=None,
            embs_seq_length=20,
            augmentation_config_path=None,
            mugsy_embs_path=None,
            floor_ao_path=None,
            floor_mesh_path=None,
            floor_uv_mapping_path=None,
            #
            geometry_nocorr_path=None,
            #
            camera_render_mode=None,
            camera_render_zoom_speed=0.003,
            camera_render_rotate_speed=None,
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

        # TODO: mb we can make this automatic?
        self.frame_list_path = frame_list_path
        self.reference_mesh_path = reference_mesh_path
        self.reference_uv_mapping_path = reference_uv_mapping_path
        self.nbs_idxs_path = nbs_idxs_path
        self.nbs_weights_path = nbs_weights_path
        self.transform_path = transform_path
        self.motion_path = motion_path
        self.lbs_weights_path = lbs_weights_path
        self.lbs_template_path = lbs_template_path
        self.lbs_lookatpt_path = lbs_lookatpt_path
        self.lbs_scale_path = lbs_scale_path
        self.geometry_uv_path = geometry_uv_path
        self.geometry_mesh_path = geometry_mesh_path
        self.geometry_nocorr_path = geometry_nocorr_path
        self.verts_unposed_mean_path = verts_unposed_mean_path

        self.floor_mesh_path = floor_mesh_path
        self.floor_uv_mapping_path = floor_uv_mapping_path

        self.lbs = lbs

        # image things
        # assert image_ds_rate == 1, "downsampling rate != 1 is not supported yet"
        # images
        self.image_path = image_path
        self.image_db_path = image_db_path
        self.image_mask_path = image_mask_path
        self.image_mask_db_path = image_mask_db_path
        self.image_part_mask_path = image_part_mask_path
        self.ignore_masks_path = ignore_masks_path
        self.ignore_masks_db_path = ignore_masks_db_path
        self.background_path = background_path
        self.background_db_path = background_db_path

        self.embs_db_path = embs_db_path

        self.image_ds_rate = image_ds_rate
        self.image_height = image_height // image_ds_rate
        self.image_width = image_width // image_ds_rate

        self.inverse_rendering = inverse_rendering
        self.krt_path = krt_path
        self.valid_camera_prefix = valid_camera_prefix

        self.camera_id = camera_id
        self.sample_cameras = sample_cameras

        if camera_render_mode:
            self.camera_render_mode = camera_render_mode
            self.camera_render_zoom_speed = camera_render_zoom_speed
            self.camera_render_rotate_speed = camera_render_rotate_speed
        else:
            self.camera_render_mode = 0

        self.augmentation_config_path = augmentation_config_path
        if self.augmentation_config_path is not None:
            self.augmentation_config = th.load(self.augmentation_config_path)

        # TODO: for simplicity we might just want to specify all via DB or all via
        assert (
                self.image_path is None or self.image_db_path is None
        ), "you can only specify path or DB path"

        assert (
                self.image_mask_path is None or self.image_mask_db_path is None
        ), "you can only specify path or DB path"

        assert (
                self.ignore_masks_path is None or self.ignore_masks_db_path is None
        ), "only path or DB path can be specified"

        assert (
                self.background_path is None or self.background_db_path is None
        ), "only path or DB path can be specified"

        if self.image_db_path:
            self.image_db = LMDBDict(self.image_db_path, readonly=True)

        if self.image_mask_db_path:
            self.image_mask_db = LMDBDict(self.image_mask_db_path, readonly=True)

        if self.background_db_path:
            self.bg_db = LMDBDict(self.background_db_path, readonly=True)

        if self.ignore_masks_db_path:
            self.ignore_masks_db = LMDBDict(self.ignore_masks_db_path, readonly=True)

        if self.embs_db_path:
            self.embs_db = LMDBDict(self.embs_db_path, readonly=True)

        if cameras is None:
            self.cameras = load_krt(self.krt_path)
            # filtering the cameras
            if self.valid_camera_prefix:
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
            elif self.background_db_path:
                logger.info("reading ignore masks")
                for cid, camera in self.cameras.items():
                    camera["ignore_mask"] = self.ignore_masks_db[cid]
                logger.info("done!")

            if self.background_path:
                logger.info("reading background images")
                for camera_id, camera in self.cameras.items():
                    path = self.background_path.format(camera=camera_id)
                    camera["background_image"] = pyutils.io.imread(
                        path,
                        (self.image_height, self.image_width),
                    )
                logger.info("done!")
            elif self.background_db_path:
                # TODO: check the size?
                logger.info("reading background images")
                for camera_id, camera in self.cameras.items():
                    camera["background_image"] = self.bg_db[camera_id]
                logger.info("done!")
        else:
            self.cameras = cameras
        self.cameras_ids = list(self.cameras.keys())

        self.root_rot_db_path = root_rot_db_path
        if self.root_rot_db_path:
            self.root_rot_db = LMDBDict(root_rot_db_path)

        self.tex_mean_path = tex_mean_path
        self.tex_std_path = tex_std_path
        self.tex_path = tex_path
        self.tex_ip_path = tex_ip_path

        # loading the cameras if necessary
        # TODO: we need a unified way to do these things?
        # if isinstance(cameras, str):
        #     # TODO: check if the order is consistent here
        #     cameras = list(pyutils.io.load_krt(cameras).keys())
        if self.tex_mean_path:
            self.tex_mean = pyutils.io.imread(
                tex_mean_path, dst_size=(self.uv_size, self.uv_size), dtype=np.float32
            )

        self.tex_std = (
            float(np.genfromtxt(self.tex_std_path) ** 0.5)
            if self.tex_std_path
            else 64.0
        )

        self.ao_mean_path = ao_mean_path
        self.ao_path = ao_path
        self.ao_verts_path = ao_verts_path
        if self.ao_mean_path:
            self.ao_mean = (
                    pyutils.io.imread(
                        self.ao_mean_path,
                        # dst_size=(self.uv_size, self.uv_size),
                        dtype=np.float32,
                    )[:1]
                    / 255.0
            )

        self.floor_ao_path = floor_ao_path

        self.lbs_pos_map_path = lbs_pos_map_path
        self.lbs_normal_map_path = lbs_normal_map_path

        # face embeddings
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

        # NOTE: this is estimated once for the entire thing
        if self.lbs_template_path is not None:
            self.lbs_template_verts, _ = pyutils.io.load_ply(self.lbs_template_path)
            self.lbs_template_verts = np.asarray(
                self.lbs_template_verts, dtype=np.float32
            )

        if self.lbs_scale_path is not None:
            self.lbs_scale = np.loadtxt(self.lbs_scale_path).astype(np.float32)
            self.lbs_scale = self.lbs_scale[np.newaxis]
            if len(self.lbs_scale.shape) == 3:
                self.lbs_scale = self.lbs_scale[:, 0, :]

        self.verts_unposed_mean = None
        if self.verts_unposed_mean_path:

            if self.verts_unposed_mean_path.endswith(".npy"):
                self.verts_unposed_mean = np.load(self.verts_unposed_mean_path).astype(
                    np.float32
                )
            elif self.verts_unposed_mean_path.endswith(".ply"):
                self.verts_unposed_mean = pyutils.io.load_ply(
                    self.verts_unposed_mean_path
                )[0]
            else:
                raise ValueError()

        # NOTE: are these used in the model or the dataset?
        # reading uv reference

        if self.nbs_idxs_path and self.nbs_weights_path:
            logger.info(f"loading pre-computed neighbors")
            self.nbs_idxs = np.loadtxt(self.nbs_idxs_path).astype(np.int64)
            self.nbs_weights = np.loadtxt(self.nbs_weights_path).astype(np.float32)
            logger.info("done!")

        logger.info(f"loading the reference mesh")

        _, uv_coords, faces, uv_faces = map(
            np.array, pyutils.io.load_obj(self.reference_mesh_path)
        )
        self.faces = faces.astype(np.int32)
        self.uv_coords = uv_coords.astype(np.float32)
        self.uv_faces = uv_faces.astype(np.int32)
        self.uv_mapping = np.loadtxt(self.reference_uv_mapping_path).astype(np.int64)

        logger.info(f"done!")

        if self.floor_mesh_path:
            logger.info(f"loading the floor mesh: {self.floor_mesh_path}")
            floor_verts, floor_uv_coords, floor_faces, floor_uv_faces = map(
                np.array, pyutils.io.load_obj(self.floor_mesh_path)
            )
            self.floor_verts = floor_verts.astype(np.float32)
            self.floor_faces = floor_faces.astype(np.int32)
            self.floor_uv_coords = floor_uv_coords.astype(np.float32)
            self.floor_uv_faces = floor_uv_faces.astype(np.int32)
            logger.info(f"done!")

        if self.floor_uv_mapping_path:
            self.floor_uv_mapping = np.loadtxt(self.floor_uv_mapping_path).astype(
                np.int64
            )

        # for 3D rot-center keypoints
        if self.kpts_3d_path and self.locator_path:
            # TODO: load
            locator = Locators()
            logger.info(f"loading all rot-center keypoints {self.num_frames} frames")
            self.kpts_3d = locator.loadBatchKeypointsAndLocatorJson(
                self.locator_path, self.kpts_3d_path, self.frame_list[:, 1]
            )
            logger.info("done!")

        self.embs_seq_path = embs_seq_path
        self.motion_seq_path = motion_seq_path
        self.embs_seq_length = embs_seq_length
        self.face_embs_seq_path = face_embs_seq_path

        if self.embs_seq_path is not None and self.motion_seq_path is not None:
            self.embs_seq = np.load(self.embs_seq_path)
            self.motion_seq = np.load(self.motion_seq_path)

        if self.face_embs_seq_path is not None:
            self.face_embs_seq = np.load(self.face_embs_seq_path)

        self.mugsy_embs_path = mugsy_embs_path
        if self.mugsy_embs_path:
            # TODO: mb we can just join these with
            self.mugsy_embs = np.load(self.mugsy_embs_path)

        if self.camera_render_mode > 0:
            # HACK hardcoded here only for Sociopticon
            self.renderscl = 1.0
            # self.rotaxle = th.from_numpy(np.array([0.007813,-0.975712, -0.218917], dtype=np.float32))
            self.rotaxle = th.from_numpy(
                np.array([0.001021, -0.981891, -0.189446], dtype=np.float32)
            )
            # self.centroid = th.from_numpy(np.array([-102.182884, 208.875565, 2909.277832],dtype=np.float32))
            self.centroid = th.from_numpy(
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
        try:
            seq, frame = self.frame_list[idx]

            fmts = dict(frame=int(frame), seq=seq, dataset_idx=idx)

            # by default we just have a fixed camera id
            camera_id = self.camera_id
            if self.sample_cameras:
                camera_id = np.random.choice(self.cameras_ids)

            if camera_id is not None:
                fmts.update(camera=camera_id)

            inputs = dict(frame=frame, seq=seq, index=idx, dataset_idx=idx)
            targets = dict()

            if self.motion_path:
                motion = np.loadtxt(self.motion_path.format(**fmts)).astype(np.float32)
                inputs.update(motion=motion)

            if self.geometry_mesh_path:
                verts, _ = pyutils.io.load_ply(self.geometry_mesh_path.format(**fmts))
                verts = np.asarray(verts, dtype=np.float32)
                inputs.update(verts=verts)
                targets.update(verts=verts)

            if self.geometry_nocorr_path:
                verts_nocorr, _ = pyutils.io.load_ply(
                    self.geometry_nocorr_path.format(**fmts)
                )
                verts_nocorr = np.asarray(verts_nocorr)
                targets.update(verts_nocorr=verts_nocorr)

            if self.root_rot_db_path:
                inputs.update(R_root_inv=self.root_rot_db[frame])

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
            if self.cameras is not None and camera_id is not None:

                newintmat = th.from_numpy(
                    self.cameras[camera_id]["intrin"].astype(np.float32)
                )
                newextmat = th.from_numpy(
                    self.cameras[camera_id]["extrin"].astype(np.float32)
                )
                newviewpt = th.from_numpy(
                    self.cameras[camera_id]["position"].astype(np.float32)
                )

                if self.camera_render_mode > 0:
                    ##########################################
                    tensor_transf = th.from_numpy(transf)
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

            if self.geometry_uv_path:
                # geometry: TODO: maybe we can just add mask externally?
                verts_uv_raw = pyutils.io.imread_exr(
                    self.geometry_uv_path.format(**fmts)
                )
                verts_uv_mask = (verts_uv_raw != 0).astype(np.float32)

                # TODO: should this not be further?
                verts_uv = (
                    (
                            np.dot(
                                verts_uv_raw.reshape(self.uv_size * self.uv_size, 3),
                                transf[:3, :3].T,
                            )
                            + transf[:3, 3].reshape(1, 3)
                    )
                        .reshape(self.uv_size, self.uv_size, 3)
                        .transpose((2, 0, 1))
                        .astype(np.float32)
                )

                # TODO: see if this is actually necessary?
                inputs.update(
                    verts_uv=verts_uv,
                    verts_uv_mask=verts_uv_mask,
                )

            if self.lbs_pos_map_path:
                # TODO: resize here?
                lbs_pos_map = (
                        pyutils.io.imread_exr(
                            self.lbs_pos_map_path.format(**fmts)
                        ).transpose((2, 0, 1))
                        / 1000.0
                )
                inputs.update(lbs_pos_map=lbs_pos_map)

            if self.lbs_normal_map_path:
                lbs_normal_map = (
                        pyutils.io.imread(self.lbs_normal_map_path.format(**fmts)) / 255.0
                )
                inputs.update(lbs_normal_map=lbs_normal_map)

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
                        raise ValueError("capture image is empty")
                        # TODO: should we return None here?
                    if self.image_part_mask_path and os.path.exists(
                            self.image_part_mask_path.format(**fmts)
                    ):
                        capture_image_mask = pyutils.io.imread(
                            self.image_part_mask_path.format(**fmts),
                            (self.image_height, self.image_width),
                        )
                        capture_image_mask = np.any(capture_image_mask != 0, axis=0)[
                            np.newaxis
                        ].astype(np.float32)

                    elif self.image_mask_path:
                        capture_image_mask = pyutils.io.imread(
                            self.image_mask_path.format(**fmts),
                            (self.image_height, self.image_width),
                        )
                        # TODO: the real bg?
                        capture_image_mask = capture_image_mask[:1] / 255.0
                    else:
                        raise ValueError(f"no capture mask provided for frame {frame}")

                    if np.sum(capture_image_mask) < 1.0:
                        logger.warning(f"capture image mask is empty")
                        raise ValueError("capture image mask is empty")

                    # if self.floor_ao_path is None or self.floor_mesh_path is None:
                    #     capture_bg_image = (
                    #         capture_image * (1.0 - capture_image_mask)
                    #         + capture_image_mask
                    #         * self.cameras[camera_id]["background_image"]
                    #     )
                    if "background_image" in self.cameras[camera_id]:
                        if self.floor_ao_path is None or self.floor_mesh_path is None:
                            capture_bg_image = (
                                    capture_image * (1.0 - capture_image_mask)
                                    + capture_image_mask
                                    * self.cameras[camera_id]["background_image"]
                            )
                        else:
                            capture_bg_image = self.cameras[camera_id][
                                "background_image"
                            ]
                    else:
                        capture_bg_image = capture_image * (1.0 - capture_image_mask)

                    if "ignore_mask" in self.cameras[camera_id]:
                        inputs.update(
                            ignore_mask=self.cameras[camera_id]["ignore_mask"]
                        )

                    inputs.update(
                        capture_bg_image=capture_bg_image,
                    )

                    targets.update(
                        capture_image=capture_image,
                        capture_image_mask=capture_image_mask,
                    )
                except (RuntimeError, ValueError) as err:
                    logger.info(
                        f"error when reading {self.image_path.format(**fmts)}, {err}"
                    )
                    return None
            elif self.image_db_path and self.image_mask_db_path:

                capture_image = np.transpose(
                    self.image_db[(camera_id, frame)], (2, 0, 1)
                )
                inputs.update(capture_image=capture_image)

                capture_image_mask = self.image_mask_db[(camera_id, frame)] / 255.0

                if np.sum(capture_image_mask) < 1.0:
                    logger.warning(f"capture image mask is empty")
                    raise ValueError("capture image mask is empty")

                capture_bg_image = (
                        capture_image * (1.0 - capture_image_mask)
                        + capture_image_mask * self.cameras[camera_id]["background_image"]
                )

                inputs.update(
                    capture_bg_image=capture_bg_image,
                    ignore_mask=self.cameras[camera_id]["ignore_mask"],
                    capture_image_mask=capture_image_mask,
                )

                targets.update(capture_image_mask=capture_image_mask)

            if self.tex_path:
                if self.tex_ip_path:
                    tex = cv2.imread(self.tex_path.format(**fmts))
                    tex_ip_mask = cv2.imread(
                        self.tex_ip_path.format(**fmts), cv2.IMREAD_UNCHANGED
                    )
                    tex = cv2.inpaint(tex, tex_ip_mask, 3, cv2.INPAINT_TELEA)
                    tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
                    tex = tex.transpose((2, 0, 1)).astype(np.float32)
                else:
                    tex = pyutils.io.imread(self.tex_path.format(**fmts))
                inputs.update(tex=tex)

            if self.embs_db_path:
                inputs.update(embs=self.embs_db[frame])

            if self.ao_path:
                ao = pyutils.io.imread(self.ao_path.format(**fmts))[:1] / 255.0
                inputs.update(ao=ao)

            if self.ao_verts_path:
                ao_verts = np.load(self.ao_verts_path.format(**fmts))
                inputs.update(ao_verts=ao_verts)

            if self.floor_ao_path:
                floorao = (
                        pyutils.io.imread(self.floor_ao_path.format(**fmts))[:1] / 255.0
                )
                inputs.update(floorao=floorao)

            if self.kpts_3d_path:
                inputs.update(kpts_3d=self.kpts_3d[int(frame)])
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

            if self.face_embs_path:
                # this is a global frame index
                if self.face_embs_path.endswith(".npz"):
                    with np.load(self.face_embs_path.format(**fmts)) as npz:
                        face_embs = npz["enc_z"][:].astype(np.float32)
                        face_Rt = np.vstack(
                            (np.hstack((npz["R"], npz["t"][:, None])), [0, 0, 0, 1])
                        ).astype(np.float32)
                        inputs.update(face_Rt=face_Rt)

                elif self.face_embs_path.endswith(".txt"):
                    face_embs = np.loadtxt(self.face_embs_path.format(**fmts)).astype(
                        np.float32
                    )
                else:
                    raise ValueError("unsupported `face_embs` format")
                inputs.update(face_embs=face_embs)

            if self.embs_seq_path:
                start, end = max(0, idx - self.embs_seq_length + 1), idx + 1
                delta_index = self.embs_seq_length - (end - start)

                embs_seq = self.embs_seq[start:end]
                motion_seq = self.motion_seq[start:end]

                # padding with the first value
                embs_pad = np.tile(embs_seq[:1], (delta_index, 1, 1, 1))
                motion_pad = np.tile(motion_seq[:1], (delta_index, 1))

                motion_seq = np.concatenate([motion_pad, motion_seq], axis=0).transpose(
                    1, 0
                )
                embs_seq = np.concatenate([embs_pad, embs_seq], axis=0).transpose(
                    1, 0, 2, 3
                )
                inputs.update(motion_seq=motion_seq)
                targets.update(embs_seq=embs_seq)

            if self.face_embs_seq_path:
                start, end = max(0, idx - self.embs_seq_length + 1), idx + 1
                face_embs_seq = self.face_embs_seq[start:end]
                face_embs_pad = np.tile(face_embs_seq[:1], (delta_index, 1))
                face_embs_seq = np.concatenate(
                    [face_embs_pad, face_embs_seq], axis=0
                ).transpose(1, 0)
                inputs.update(face_embs_seq=face_embs_seq)

            if self.mugsy_embs_path:
                # just randomly sampling a face code
                mugsy_idx = np.random.choice(self.mugsy_embs.shape[0])
                inputs.update(mugsy_embs=self.mugsy_embs[mugsy_idx])

            if self.augmentation_config_path:
                # getting an index
                src_frame = np.random.choice(self.augmentation_config["frames"])
                clothes_verts = pyutils.io.load_ply(
                    self.augmentation_config["path_fmt"].format(frame=src_frame)
                )[0]
                inputs.update(
                    clothes_verts=clothes_verts,
                    clothes_idxs=self.augmentation_config["clothes_idxs"],
                    clothes_frame=src_frame,
                )

        except (RuntimeError, ValueError, FileNotFoundError) as e:
            logger.warning(f"Unhandled error: {e}")
            return None

        return inputs, targets


class _RepeatSampler:
    """Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(th.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)