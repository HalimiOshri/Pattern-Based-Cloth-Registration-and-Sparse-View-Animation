import numpy as np
import os
import pickle

import torch
import torch.utils.data

from src import pyutils
from src.pyutils import camera_layers

import cv2

from src.kinematic_modeling.lbs.locator_keypoint import Locators
from external.KinematicAlignment.utils.tensorIO import ReadTensorFromBinaryFile

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
        partial_mask_path=None,
        motion_velocity_path=None,
        clothes_lbs_template_path=None,
        clothes_lbs_scale_path=None,
        motion_length=None,
        global_scaling=None,
        transform_path=None,
        kpts_3d_path=None,
        face_kpts_3d_path=None,
        face_kpts_3d_vidxs_path=None,
        locator_path=None,
        clothes_verts_unposed_mean_path=None,
        krt_path=None,
        path_camera_indices=None,
        image_height=None,
        image_width=None,
        image_ds_rate=None,
        pose_sequence_length=None,
        frame_interval=2,
        per_camera_delta_pixel_uv_path=None,
        posed_LBS_mesh_path=None,
        posed_LBS_uv_path=None,
        driving_camera_ids=None,
        template_obj_path=None,
        template_ply_path=None,
        barycentric_path=None,
        tIndex_path=None,
        dummy_texture_path=None,
        temporal_buffer_size=None,
        deformer=None,
        #LBO_path=None, LBO_num=None, LBO_cutoff=None, LBO_exp_deacy=None, LBO_scale=None,
        **kwargs
    ):
        # self.LBO_path = LBO_path
        # self.LBO_num = LBO_num
        # self.LBO_cutoff = LBO_cutoff
        # self.LBO_exp_deacy = LBO_exp_deacy
        # self.LBO_scale = LBO_scale
        self.deformer = deformer
        self.uv_size = uv_size
        self.temporal_buffer_size = temporal_buffer_size
        self.template_obj_path = template_obj_path
        self.template_ply_path = template_ply_path
        self.barycentric_path = barycentric_path
        self.tIndex_path = tIndex_path
        self.dummy_texture_path = dummy_texture_path

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

        self.clothes_reference_mesh_path = clothes_reference_mesh_path
        self.clothes_reference_uv_mapping_path = clothes_reference_uv_mapping_path
        self.clothes_nbs_idxs_path = clothes_nbs_idxs_path
        self.clothes_nbs_weights_path = clothes_nbs_weights_path
        self.clothes_lbs_template_path = clothes_lbs_template_path
        self.clothes_lbs_scale_path = clothes_lbs_scale_path
        self.clothes_geometry_mesh_path = clothes_geometry_mesh_path
        self.clothes_verts_unposed_mean_path = clothes_verts_unposed_mean_path
        self.partial_mask_path = partial_mask_path
        self.per_camera_delta_pixel_uv_path = per_camera_delta_pixel_uv_path
        self.posed_LBS_mesh_path = posed_LBS_mesh_path
        self.posed_LBS_uv_path = posed_LBS_uv_path
        self.driving_camera_ids = driving_camera_ids

        self.image_height = image_height // image_ds_rate
        self.image_width = image_width // image_ds_rate

        self.krt_path = krt_path
        self.path_camera_indices = path_camera_indices

        self.pose_sequence_length = pose_sequence_length
        self.frame_interval = frame_interval

        # loading the frames
        # NOTE: these
        self.frame_list = np.genfromtxt(self.frame_list_path, dtype=np.str)
        self.all_frames = np.array([int(frame) for frame in self.frame_list[:, 1]])
        self.min_frame = np.min(self.all_frames)
        self.max_frame = np.max(self.all_frames)

        # TODO: generate the ids in advance?
        self.num_frames = len(self.frame_list)

        if len(self.frame_list.shape) == 1:
            logger.warning("frame list does not contain sequence numbers, adding fake")
            self.frame_list = np.stack(
                [np.array(["unknown"] * self.num_frames), self.frame_list], axis=1
            )

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

        fmts = dict(frame=int(frame), seq=seq, dataset_idx=idx)

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
                tmp_fmts = dict(frame=int(frame) - i * self.frame_interval, seq=seq)
                motion_path = self.motion_path.format(**tmp_fmts)
                if os.path.isfile(motion_path):
                    if motion_path.endswith('.bt'):
                        motion_list.append(pyutils.tensorIO.ReadArrayFromBinaryFile(motion_path).astype(np.float32))
                    elif motion_path.endswith('.txt'):
                        motion_list.append(np.loadtxt(motion_path).astype(np.float32))
                else:
                    assert i != 0, "The pose file for the target frame must exist!"
                    motion_list.append(motion_list[-1])   # repeat the same
            motion_list.reverse()   # reverse the order
            # [(batch size), pose dimension, num frames]
            motion = np.stack(motion_list, axis=0).transpose(1, 0)

        inputs = dict(motion=motion, frame=frame, seq=seq, dataset_idx=idx)
        targets = dict()

        if self.clothes_geometry_mesh_path:
            clothes_verts, _ = pyutils.io.load_ply(self.clothes_geometry_mesh_path.format(**fmts))
            clothes_verts = np.asarray(clothes_verts, dtype=np.float32)
            inputs.update(clothes_verts=clothes_verts)
            targets.update(clothes_verts=clothes_verts)

        if self.posed_LBS_mesh_path:
            frame_window = np.maximum(np.array([fmts["frame"] - i for i in range(0, self.temporal_buffer_size)]),
                                      self.min_frame)
            fmts_window = fmts
            posed_LBS_verts = []
            for frame in frame_window:
                fmts_window["frame"] = frame
                posed_LBS_verts.append(pyutils.io.load_ply(self.posed_LBS_mesh_path.format(**fmts))[0])

            posed_LBS_verts = torch.stack(posed_LBS_verts, dim=0)
            inputs.update(posed_LBS_verts=posed_LBS_verts)

        if self.per_camera_delta_pixel_uv_path:
            try:
                frame_window = np.maximum(np.array([fmts["frame"] - i for i in range(0, self.temporal_buffer_size)]), self.min_frame)
                fmts_window = fmts
                per_camera_delta_pixel_uv = []
                for frame in frame_window:
                    fmts_window["frame"] = frame
                    per_camera_delta_pixel_uv.append(torch.load(self.per_camera_delta_pixel_uv_path.format(**fmts_window)))
                per_camera_delta_pixel_uv = torch.stack(per_camera_delta_pixel_uv, dim=0)
                #per_camera_delta_pixel_uv = ReadTensorFromBinaryFile(self.per_camera_delta_pixel_uv_path.format(**fmts))
                if self.driving_camera_ids:
                    ids = np.array(self.driving_camera_ids)
                    per_camera_delta_pixel_uv = per_camera_delta_pixel_uv[:, ids, ...]
                # per_camera_delta_pixel_uv = per_camera_delta_pixel_uv.view(-1, self.uv_size, self.uv_size, 2)
                inputs.update(
                    per_camera_delta_pixel_uv=per_camera_delta_pixel_uv
                )
            except (RuntimeError, ValueError):
                logger.info(f"error when reading {self.per_camera_delta_pixel_uv_path}")
                return None

        if self.kpts_3d_path:
            targets.update(kpts_3d=self.kpts_3d[int(frame)])

        return inputs, targets
