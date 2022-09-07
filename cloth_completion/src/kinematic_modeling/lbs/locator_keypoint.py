# from body_vae.config import *
# from body_vae.momentum_python_utils import Quaternion
import os

import json
from struct import unpack

import numpy as np

import torch
import torch.nn as nn


import logging

log = logging.getLogger(__name__)


keypoint2locator_map = {
    0: "nose",
    # 2: "right_shoulder",
    3: "right_elbow",
    # 4: "right_wrist",
    # 5: "left_shoulder",
    6: "left_elbow",
    # 7: "left_wrist",
    # 8: "right_hip",
    9: "right_knee",
    10: "right_ankle",
    # 11: "left_hip",
    12: "left_knee",
    13: "left_ankle",
    14: "right_eye",
    15: "left_eye",
    16: "right_ear",
    17: "left_ear",
    18: "LEFT_WRIST",
    19: "LEFT_THUMB_PROXIMAL",
    20: "LEFT_THUMB_INTERMEDIATE",
    21: "LEFT_THUMB_DISTAL",
    22: "l_thumb_null",
    23: "LEFT_INDEX_PROXIMAL",
    24: "LEFT_INDEX_INTERMEDIATE",
    25: "LEFT_INDEX_DISTAL",
    26: "l_index_null",
    27: "LEFT_MIDDLE_PROXIMAL",
    28: "LEFT_MIDDLE_INTERMEDIATE",
    29: "LEFT_MIDDLE_DISTAL",
    30: "l_middle_null",
    31: "LEFT_RING_PROXIMAL",
    32: "LEFT_RING_INTERMEDIATE",
    33: "LEFT_RING_DISTAL",
    34: "l_ring_null",
    35: "LEFT_PINKY_PROXIMAL",
    36: "LEFT_PINKY_INTERMEDIATE",
    37: "LEFT_PINKY_DISTAL",
    38: "l_pinky_null",
    39: "RIGHT_WRIST",
    40: "RIGHT_THUMB_PROXIMAL",
    41: "RIGHT_THUMB_INTERMEDIATE",
    42: "RIGHT_THUMB_DISTAL",
    43: "r_thumb_null",
    44: "RIGHT_INDEX_PROXIMAL",
    45: "RIGHT_INDEX_INTERMEDIATE",
    46: "RIGHT_INDEX_DISTAL",
    47: "r_index_null",
    48: "RIGHT_MIDDLE_PROXIMAL",
    49: "RIGHT_MIDDLE_INTERMEDIATE",
    50: "RIGHT_MIDDLE_DISTAL",
    51: "r_middle_null",
    52: "RIGHT_RING_PROXIMAL",
    53: "RIGHT_RING_INTERMEDIATE",
    54: "RIGHT_RING_DISTAL",
    55: "r_ring_null",
    56: "RIGHT_PINKY_PROXIMAL",
    57: "RIGHT_PINKY_INTERMEDIATE",
    58: "RIGHT_PINKY_DISTAL",
    59: "r_pinky_null",
    130: "r_bigtoe",
    # 131: "r_smalltoe",
    132: "r_heel",
    133: "l_bigtoe",
    # 134: "l_smalltoe",
    135: "l_heel",
}


def batchRTStoMat(batch_rts):
    """
    :param batch_rts: batch * nr_vertices * 8
    :return: batch * nr_vertices * 3 * 4
    """
    tr = batch_rts[:, :, 3:7]
    ts = batch_rts[:, :, 7].unsqueeze(2)
    tt = batch_rts[:, :, 0:3]

    # convert to matrices
    twx = 2.0 * tr[:, :, 0] * tr[:, :, 3]
    twy = 2.0 * tr[:, :, 1] * tr[:, :, 3]
    twz = 2.0 * tr[:, :, 2] * tr[:, :, 3]
    txx = 2.0 * tr[:, :, 0] * tr[:, :, 0]
    txy = 2.0 * tr[:, :, 1] * tr[:, :, 0]
    txz = 2.0 * tr[:, :, 2] * tr[:, :, 0]
    tyy = 2.0 * tr[:, :, 1] * tr[:, :, 1]
    tyz = 2.0 * tr[:, :, 2] * tr[:, :, 1]
    tzz = 2.0 * tr[:, :, 2] * tr[:, :, 2]
    mat = torch.stack(
        (
            torch.stack((1.0 - (tyy + tzz), txy + twz, txz - twy), dim=2) * ts,
            torch.stack((txy - twz, 1.0 - (txx + tzz), tyz + twx), dim=2) * ts,
            torch.stack((txz + twy, tyz - twx, 1.0 - (txx + tyy)), dim=2) * ts,
            tt,
        ),
        dim=3,
    )
    return mat


def loadFabianVectorFile(fn):
    with open(fn, "rb") as f:
        content = f.read()
    nr = unpack("i", content[:4])[0]
    data = unpack("f" * nr * 3, content[4:])
    return np.array(data, dtype=np.float32).reshape(nr, 3)


class Locators(nn.Module):
    """
    Torch Tensor based Locators class
    Takes Locators and transform them according to skin weights

    -- A very dirty layer copied from the main VAE pipeline
    """

    def __init__(self, names=None):
        super(Locators, self).__init__()
        """Create a skin from given index/weight map"""
        # self.weights = weights
        # self.indices = indices
        self.names = [] if names is None else names
        self.key2locind_map = {}

    def forward(self, states):

        """
        :param states: lbs skinning state
         batch_size x nr_skeleton_joints x 8 Skeleton States
            8 stands form 3 translation + 4 rotation (quaternion) + 1 scale
        """
        mat = batchRTStoMat(states)
        vs = (
            torch.einsum(
                "ijkl,jl->ijk", mat[:, self.indices, :, :3], self.locator_offsets
            )
            + mat[:, self.indices, :, 3]
        )
        return vs

    def loadLocators(self, js, joint_names, bindState, dtype=torch.float, fabian_fn=""):
        """Parse a skin from a json description"""
        mat = batchRTStoMat(bindState.to("cpu"))
        vertex_counter = 0
        locator_offsets = []
        indices = []

        # TODO: remove the hacked Fabian offset
        if os.path.isfile(fabian_fn):
            log.info(f"loading Fabian's locator offset from {fabian_fn}")
            fabian_offset = torch.FloatTensor(loadFabianVectorFile(fabian_fn))
        else:
            fabian_offset = None

        for iloc, loc in enumerate(js["locators"]):
            name = loc["name"]
            parentJoint = loc["parentName"]
            parentJointIndex = None
            if parentJoint in joint_names:
                parentJointIndex = joint_names.index(parentJoint)
            if parentJointIndex is None:
                log.warning(
                    "Joint {} for locator {} not found".format(parentJoint, name)
                )
                continue
            # TODO: Locks are not saved or used currently
            if "lockX" in loc:
                locks = [loc["lockX"], loc["lockY"], loc["lockZ"]]

            if "globalX" in loc:
                # Global coordinates, transform to bone-local
                vg = torch.tensor(
                    [loc["globalX"], loc["globalY"], loc["globalZ"], 1.0], dtype=dtype
                )

                M = torch.cat(
                    (
                        mat[0, parentJointIndex, :, :],
                        torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=mat.dtype),
                    ),
                    dim=0,
                )
                Mi = torch.inverse(M)
                v = Mi.mm(vg.reshape((4, 1)))[0:3, 0]
            else:
                # Local coordinates
                if fabian_offset is None:
                    v = torch.tensor(
                        [loc["offsetX"], loc["offsetY"], loc["offsetZ"]], dtype=dtype
                    )
                else:
                    v = fabian_offset[iloc]

            self.names.append(name)
            locator_offsets.append(v)
            indices.append(parentJointIndex)

        self.register_buffer("indices", torch.LongTensor(indices))
        self.register_buffer("locator_offsets", torch.stack(locator_offsets, 0))
        log.info(f"locator_offsets.shape: {self.locator_offsets.shape}")

        locator2keypoint_map = {}

        for k, v in keypoint2locator_map.items():
            locator2keypoint_map[v] = k

        for il, l in enumerate(self.names):
            if l in locator2keypoint_map:
                keypoint_index = locator2keypoint_map[l]
                self.key2locind_map[keypoint_index] = il
        return self

    def loadKeypoint(self, keypoint3d_path, frame_idx):

        #kptsfile = f"{keypoint3d_path}/image{int(frame_idx):06d}_kp3d.json"
        kptsfile = keypoint3d_path.format(frame=int(frame_idx))

        if not os.path.exists(kptsfile):
            raise FileNotFoundError(kptsfile)

        with open(kptsfile, "r") as fid:
            pose = json.load(fid)
        best_inliers = None
        best_pts = None
        best_inliers_sum = 0
        for person in pose["people"]:
            inliers = np.array(person["inliers"]).reshape((-1, 1)).astype(np.float32)
            pts = np.array(person["pts"]).reshape((-1, 3)).astype(np.float32)
            if np.sum(inliers) > best_inliers_sum:
                best_inliers = inliers
                best_pts = pts
                best_inliers_sum = np.sum(inliers)
        if best_inliers is None:
            assert False
            best_inliers = np.zeros((18 + 21 * 2 + 6 * 2 + 70, 1), dtype=np.float32)
            best_pts = np.zeros((18 + 21 * 2 + 6 * 2 + 70, 3), dtype=np.float32)

        locator_targets = np.zeros((len(self.names), 5), dtype=np.float32)
        for ki, li in self.key2locind_map.items():
            if True:
                locator_targets[li, :3] = best_pts[ki, :3]
                locator_targets[li, 3] = 1
                # locator_targets[li,4] = best_inliers[ki,3]
        return locator_targets

    def loadBatchKeypoints(self, path, frame_idxs):
        kpts = [
            (int(idx), self.loadKeypoint(path, idx)) for idx in frame_idxs
        ]
        return dict(kpts)

    # this is mainly for dataset.py
    def loadBatchKeypointsAndLocatorJson(
        self, loc_js_fpath, kps_path, frame_idxs
    ):
        with open(loc_js_fpath, "r") as f:
            js = json.load(f)

        for iloc, loc in enumerate(js["locators"]):
            name = loc["name"]
            self.names.append(name)

        locator2keypoint_map = {}

        for k, v in keypoint2locator_map.items():
            locator2keypoint_map[v] = k

        for il, l in enumerate(self.names):
            if l in locator2keypoint_map:
                keypoint_index = locator2keypoint_map[l]
                self.key2locind_map[keypoint_index] = il

        kpts = [
            (int(idx), self.loadKeypoint(kps_path, idx)) for idx in frame_idxs
        ]
        return dict(kpts)


class Limitations(nn.Module):
    def __init__(self, limits):
        super(Limitations, self).__init__()
        limit_param_index = []
        limit_param_upper = []
        limit_param_lower = []
        limit_param_weight = []
        limit_joint_index = []
        limit_joint_upper = []
        limit_joint_lower = []
        limit_joint_weight = []
        for limit in limits:
            if limit["type"] == "LimitMinMaxJointValue":
                limit_joint_index.append(limit["valueIndex"])
                limit_joint_upper.append(limit["limits"][1])
                limit_joint_lower.append(limit["limits"][0])
                limit_joint_weight.append(limit["weight"])
            elif limit["type"] == "LimitMinMaxParameter":
                limit_param_index.append(limit["parameterIndex"])
                limit_param_upper.append(limit["limits"][1])
                limit_param_lower.append(limit["limits"][0])
                limit_param_weight.append(limit["weight"])
            else:
                log.warning(f"cannot recognize limit type {limit['type']}")
                continue

        self.register_buffer("limit_param_index", torch.LongTensor(limit_param_index))
        self.register_buffer(
            "limit_param_upper", torch.FloatTensor(limit_param_upper)[None, :]
        )
        self.register_buffer(
            "limit_param_lower", torch.FloatTensor(limit_param_lower)[None, :]
        )
        self.register_buffer(
            "limit_param_weight", torch.FloatTensor(limit_param_weight)[None, :]
        )

        self.register_buffer("limit_joint_index", torch.LongTensor(limit_joint_index))
        self.register_buffer(
            "limit_joint_upper", torch.FloatTensor(limit_joint_upper)[None, :]
        )
        self.register_buffer(
            "limit_joint_lower", torch.FloatTensor(limit_joint_lower)[None, :]
        )
        self.register_buffer(
            "limit_joint_weight", torch.FloatTensor(limit_joint_weight)[None, :]
        )

    def forward(self, params, params_transformed):
        params_limited = params[:, self.limit_param_index]
        params_ge = (params_limited > self.limit_param_upper).float().detach()
        params_le = (params_limited < self.limit_param_lower).float().detach()
        params_wt = self.limit_param_weight

        params_limit_error = (
            params_ge * (params_limited - self.limit_param_upper).abs()
            + params_le * (params_limited - self.limit_param_lower).abs()
        ) * params_wt

        joints_limited = params_transformed[:, self.limit_joint_index]
        joints_ge = (joints_limited > self.limit_joint_upper).float().detach()
        joints_le = (joints_limited < self.limit_joint_lower).float().detach()
        joints_wt = self.limit_joint_weight

        joints_limit_error = (
            joints_ge * (joints_limited - self.limit_joint_upper).abs()
            + joints_le * (joints_limited - self.limit_joint_lower).abs()
        ) * joints_wt

        return params_limit_error, joints_limit_error
