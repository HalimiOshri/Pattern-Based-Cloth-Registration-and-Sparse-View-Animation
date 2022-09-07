import json
import numpy as np
import re
import torch
import torch.nn as nn

from kinematic_modeling.lbs.quaternion import Quaternion as quat
from pyutils.io import load_ply

import logging
log = logging.getLogger(__name__)

"""
A clean Linear Blend Skinning layer in pytorch, with momentum_python integrated.

Majority of codes come from momentum_python by Carsten Stoll.
Json model loader optimized using torch's advance indexing.
"""

__author__ = "Yuan Dong and Carsten Stoll"
__copyright__ = "Copyright 2004-present Facebook. All Rights Reserved."

__license__ = "Proprietary"
__version__ = "1.0.0"
__maintainer__ = "Yuan Dong"
__email__ = "ydong2@andrew.cmu.edu"
__status__ = "Development"


class ParameterTransform(nn.Module):
    def __init__(self, path=None, lbs_cfg_dict=None):
        super().__init__()

        if path is not None:
            log.warning("npz-based config is deprecated")
            npzfile = np.load(path)
            self.limits = []
            self.pose_names = list(npzfile["joint_names"])
            self.channel_names = list(npzfile["channel_names"])

            try:
                transform_offsets = torch.FloatTensor(npzfile["transform_offsets"])[
                    None, :
                ]
                transform = torch.sparse_coo_tensor(
                    torch.LongTensor(npzfile["transform_indices"]),
                    torch.FloatTensor(npzfile["transform_values"]),
                    torch.Size(npzfile["transform_shape"]),
                ).to_dense()
            except RuntimeError as error:
                # TODO: ugly hack due to npz file bug
                log.warning(f"cannot load from {path}, trying different format")
                transform_offsets = torch.FloatTensor(npzfile["transform_offsets"])
                transform = torch.sparse_coo_tensor(
                    torch.LongTensor(npzfile["transform_indices"].T),
                    torch.FloatTensor(npzfile["transform_values"]),
                    torch.Size(npzfile["transform_shape"]),
                ).to_dense()
                log.warning("done!")
        else:
            self.pose_names = list(lbs_cfg_dict["joint_names"])
            self.channel_names = list(lbs_cfg_dict["channel_names"])
            transform_offsets = torch.FloatTensor(lbs_cfg_dict["transform_offsets"])
            transform = torch.FloatTensor(lbs_cfg_dict["transform"])
            self.limits = lbs_cfg_dict["limits"]

        self.register_buffer("transform_offsets", transform_offsets)
        self.register_buffer("transform", transform)

    def forward(self, pose):
        """
        :param pose: raw pose inputs, shape (batch_size, len(pose_names))
        :return: skeleton parameters, shape (batch_size, len(channel_names)*nr_skeleton_joints)
        """
        return self.transform.mm(pose.t()).t() + self.transform_offsets


class LinearBlendSkinning(nn.Module):
    def __init__(
        self,
        model_json_path,
        model_npz_path=None,
        model_txt_path=None,
        num_max_skin_joints=8,
        optimize_mesh=True,
        template_path=None,
        scale_path=None,
    ):
        super().__init__()

        with open(model_json_path, "r") as fh:
            model = json.load(fh)

        if model_npz_path is not None:
            self.param_transform = ParameterTransform(path=model_npz_path)
        else:
            # TODO: just unroll the parameters?
            lbs_cfg_dict = load_momentum_cfg(model, lbs_config_txt_path=model_txt_path)
            self.param_transform = ParameterTransform(lbs_cfg_dict=lbs_cfg_dict)

        self.joint_names = []

        nr_joints = len(model["Skeleton"]["Bones"])
        joint_parents = torch.zeros((nr_joints, 1), dtype=torch.int64)
        joint_rotation = torch.zeros((nr_joints, 4), dtype=torch.float32)
        joint_offset = torch.zeros((nr_joints, 3), dtype=torch.float32)
        for idx, bone in enumerate(model["Skeleton"]["Bones"]):
            self.joint_names.append(bone["Name"])
            if bone["Parent"] > nr_joints:
                joint_parents[idx] = -1
            else:
                joint_parents[idx] = bone["Parent"]
            joint_rotation[idx, :] = torch.FloatTensor(bone["PreRotation"])
            joint_offset[idx, :] = torch.FloatTensor(bone["TranslationOffset"])

        skin_model = model["SkinnedModel"]
        mesh_vertices = torch.FloatTensor(skin_model["RestPositions"])
        mesh_normals = torch.FloatTensor(skin_model["RestVertexNormals"])

        weights = torch.FloatTensor([e[1] for e in skin_model["SkinningWeights"]])
        indices = torch.LongTensor([e[0] for e in skin_model["SkinningWeights"]])
        offsets = torch.LongTensor(skin_model["SkinningOffsets"])

        nr_vertices = len(offsets) - 1
        skin_weights = torch.zeros(
            (nr_vertices, num_max_skin_joints), dtype=torch.float32
        )
        skin_indices = torch.zeros(
            (nr_vertices, num_max_skin_joints), dtype=torch.int64
        )

        offset_right = offsets[1:]
        for offset in range(num_max_skin_joints):
            offset_left = offsets[:-1] + offset
            skin_weights[offset_left < offset_right, offset] = weights[
                offset_left[offset_left < offset_right]
            ]
            skin_indices[offset_left < offset_right, offset] = indices[
                offset_left[offset_left < offset_right]
            ]

        mesh_faces = torch.IntTensor(skin_model["Faces"]["Indices"]).view(-1, 3)
        mesh_texture_faces = torch.IntTensor(
            skin_model["Faces"]["TextureIndices"]
        ).view(-1, 3)
        mesh_texture_coords = torch.FloatTensor(skin_model["TextureCoordinates"]).view(
            -1, 2
        )

        zero_pose = torch.zeros(
            (1, len(self.param_transform.pose_names)), dtype=torch.float32
        )
        zero_state = solve_skeleton_state(
            self.param_transform(zero_pose), joint_offset, joint_rotation, joint_parents
        )

        # self.register_buffer('mesh_vertices', mesh_vertices) # we want to train on rest pose
        # self.mesh_vertices = nn.Parameter(mesh_vertices, requires_grad=optimize_mesh)
        self.register_buffer("mesh_vertices", mesh_vertices)

        self.register_buffer("joint_parents", joint_parents)
        self.register_buffer("joint_rotation", joint_rotation)
        self.register_buffer("joint_offset", joint_offset)
        self.register_buffer("mesh_normals", mesh_normals)
        self.register_buffer("mesh_faces", mesh_faces)
        self.register_buffer("mesh_texture_faces", mesh_texture_faces)
        self.register_buffer("mesh_texture_coords", mesh_texture_coords)
        self.register_buffer("skin_weights", skin_weights)
        self.register_buffer("skin_indices", skin_indices)
        self.register_buffer("zero_state", zero_state)
        self.register_buffer("rest_vertices", mesh_vertices)

        # pre-compute joint weights
        self.register_buffer("joints_weights", self.compute_joints_weights())

        # TODO: there should be only one version of mesh_vertices?
        if template_path is not None:
            template_verts, _ = load_ply(template_path)
            self.register_buffer("template_verts", template_verts)

        if scale_path is not None:
            scale = np.loadtxt(scale_path).astype(np.float32)[np.newaxis]
            scale = scale[:, 0, :] if len(scale.shape) == 3 else scale
            self.register_buffer("scale", torch.tensor(scale))

    @property
    def num_verts(self):
        return self.mesh_vertices.size(0)

    @property
    def num_joints(self):
        return self.joint_offset.size(0)

    @property
    def num_params(self):
        return self.skin_weights.shape[-1]

    def compute_rigid_transforms(self, global_pose, local_pose, scale):
        """Returns rigid transforms."""
        params = torch.cat([global_pose, local_pose, scale], axis=-1)
        params = self.param_transform(params)
        return solve_skeleton_state(
            params, self.joint_offset, self.joint_rotation, self.joint_parents
        )

    def compute_joints_weights(self, drop_empty=False):
        """Compute weights per joint given flattened weights-indices."""
        idxs_verts = torch.arange(self.num_verts)[:, np.newaxis].expand(
            -1, self.num_params
        )
        weights_joints = torch.zeros(
            (self.num_joints, self.num_verts),
            dtype=torch.float32,
            device=self.skin_weights.device,
        )
        weights_joints[self.skin_indices, idxs_verts] = self.skin_weights

        if drop_empty:
            weights_joints = weights_joints[weights_joints.sum(axis=-1).abs() > 0]

        return weights_joints

    def compute_root_rigid_transform(self, poses):
        """Get a transform of the root joint."""
        scales = torch.zeros(
            (poses.shape[0], len(self.param_transform.pose_names) - poses.shape[1]),
            dtype=poses.dtype,
            device=poses.device,
        )
        params = torch.cat((poses, scales), 1)
        states = solve_skeleton_state(
            self.param_transform(params),
            self.joint_offset,
            self.joint_rotation,
            self.joint_parents,
        )
        mat = states_to_matrix(self.zero_state, states)
        return mat[:, 1, :, 3], mat[:, 1, :, :3]

    def compute_relative_rigid_transforms(self, global_pose, local_pose, scale):
        params = torch.cat([global_pose, local_pose, scale], axis=-1)
        params = self.param_transform(params)

        batch_size = params.shape[0]

        joint_offset = self.joint_offset
        joint_rotation = self.joint_rotation

        # batch processing for parameters
        jp = params.view((batch_size, -1, 7))
        lt = jp[:, :, 0:3] + joint_offset.unsqueeze(0)
        lr = quat.batchMul(
            joint_rotation.unsqueeze(0), quat.batchFromXYZ(jp[:, :, 3:6])
        )
        return torch.cat([lt, lr], axis=-1)

    def skinning(self, bind_state, vertices, target_states):
        """
        Apply skinning to a set of states

        Args:
            b/bind_state: 1 x nr_joint x 8 bind state
            v/vertices: 1 x nr_vertices x 3 vertices
            t/target_states: batch_size x nr_joint x 8 current states

        Returns:
            batch_size x nr_vertices x 3 skinned vertices
        """
        assert target_states.size()[1:] == bind_state.size()[1:]

        mat = states_to_matrix(bind_state, target_states)

        # apply skinning to vertices
        vs = torch.matmul(
            mat[:, self.skin_indices],
            torch.cat(
                (vertices, torch.ones_like(vertices[:, :, 0]).unsqueeze(2)), dim=2
            )
            .unsqueeze(2)
            .unsqueeze(4),
        )
        ws = self.skin_weights.unsqueeze(2).unsqueeze(3)
        res = (vs * ws).sum(dim=2).squeeze(3)

        return res

    def unpose(self, poses, scales, verts, dynamic_indices=None):
        """
        :param poses: 100 (tx ty tz rx ry rz) params in blueman
        :param scales: 29 (s) params in blueman
        :return:
        """
        # check shape of poses and scales
        params = torch.cat((poses, scales), 1)
        states = solve_skeleton_state(
            self.param_transform(params),
            self.joint_offset,
            self.joint_rotation,
            self.joint_parents,
        )

        return self.unskinning(self.zero_state, states, verts, dynamic_indices)

    def unskinning(self, bind_state, target_states, verts, dynamic_indices=None):
        """
        Apply skinning to a set of states

        Args:
            b/bind_state: 1 x nr_joint x 8 bind state
            v/vertices: 1 x nr_vertices x 3 vertices
            t/target_states: batch_size x nr_joint x 8 current states

        Returns:
            batch_size x nr_vertices x 3 skinned vertices
        """
        assert target_states.size()[1:] == bind_state.size()[1:]

        mat = states_to_matrix(bind_state, target_states)

        if dynamic_indices is not None:
            ws = self.skin_weights[None, dynamic_indices, :, None, None]
            si = self.skin_indices[dynamic_indices, :]
        else:
            ws = self.skin_weights[None, :, :, None, None]
            si = self.skin_indices
        sum_mat = (mat[:, si] * ws).sum(dim=2)

        sum_mat4x4 = torch.cat((sum_mat, torch.zeros_like(sum_mat[:, :, :1, :])), dim=2)
        sum_mat4x4[:, :, 3, 3] = 1.0

        verts_4d = torch.cat(
            (verts, torch.ones_like(verts[:, :, :1])), dim=2
        ).unsqueeze(3)

        resmesh = []
        for i in range(sum_mat.shape[0]):
            newmat = sum_mat4x4[i, :, :, :].contiguous()
            invnewmat = newmat.inverse()
            tmpvets = invnewmat.matmul(verts_4d[i])
            resmesh.append(tmpvets.unsqueeze(0))
        resmesh = torch.cat(resmesh)

        return resmesh.squeeze(3)[..., :3].contiguous()

    def forward(self, poses, scales, templmesh=None, output_all=False):
        """
        :param poses: 100 (tx ty tz rx ry rz) params in blueman
        :param scales: 29 (s) params in blueman
        :return:
        """
        params = torch.cat((poses, scales), 1)
        params_transformed = self.param_transform(params)
        states = solve_skeleton_state(
            params_transformed,
            self.joint_offset,
            self.joint_rotation,
            self.joint_parents,
        )
        if templmesh is None:
            mesh = self.skinning(
                self.zero_state, self.mesh_vertices.unsqueeze(0), states
            )
        else:
            mesh = self.skinning(self.zero_state, templmesh, states)
        # return mesh
        if output_all:
            return params, params_transformed, states, mesh
        else:
            return mesh


def solve_skeleton_state(param, joint_offset, joint_rotation, joint_parents):
    """
    :param param: batch_size x (7*nr_skeleton_joints) ParamTransform Outputs.
    :return: batch_size x nr_skeleton_joints x 8 Skeleton States
        8 stands form 3 translation + 4 rotation (quat) + 1 scale
    """
    batch_size = param.shape[0]
    # batch processing for parameters
    jp = param.view((batch_size, -1, 7))
    lt = jp[:, :, 0:3] + joint_offset.unsqueeze(0)
    lr = quat.batchMul(joint_rotation.unsqueeze(0), quat.batchFromXYZ(jp[:, :, 3:6]))
    ls = torch.pow(
        torch.tensor([2.0], dtype=torch.float32, device=param.device),
        jp[:, :, 6].unsqueeze(2),
    )

    state = []
    for index, parent in enumerate(joint_parents):
        if int(parent) != -1:
            gr = quat.batchMul(state[parent][:, :, 3:7], lr[:, index, :].unsqueeze(1))
            gt = (
                quat.batchRot(
                    state[parent][:, :, 3:7],
                    lt[:, index, :].unsqueeze(1) * state[parent][:, :, 7].unsqueeze(2),
                )
                + state[parent][:, :, 0:3]
            )
            gs = state[parent][:, :, 7].unsqueeze(2) * ls[:, index, :].unsqueeze(1)
            state.append(torch.cat((gt, gr, gs), dim=2))
        else:
            state.append(
                torch.cat(
                    (lt[:, index, :], lr[:, index, :], ls[:, index, :]), dim=1
                ).view((batch_size, 1, 8))
            )

    return torch.cat(state, dim=1)


def states_to_matrix(bind_state, target_states, return_transform=False):
    # multiply bind inverse with states
    br = quat.batchInvert(bind_state[:, :, 3:7])
    bs = bind_state[:, :, 7].unsqueeze(2).reciprocal()
    bt = quat.batchRot(br, -bind_state[:, :, 0:3]) * bs

    # applying rotation
    tr = quat.batchMul(target_states[:, :, 3:7], br)
    # applying scaling
    ts = target_states[:, :, 7].unsqueeze(2) * bs
    # applying transformation
    tt = (
        quat.batchRot(
            target_states[:, :, 3:7], bt * target_states[:, :, 7].unsqueeze(2)
        )
        + target_states[:, :, 0:3]
    )

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
    if return_transform:
        return mat, (tr, tt, ts)
    return mat


def get_influence_map(
    transform_raw, pose_length=None, num_params_per_joint=7, eps=1.0e-6
):
    num_joints = transform_raw.shape[0] // num_params_per_joint
    num_params = transform_raw.shape[-1]

    if pose_length is None:
        pose_length = num_params
    assert pose_length <= num_params

    transform_raw = transform_raw.reshape(
        (num_joints, num_params_per_joint, num_params)
    )

    return [
        torch.where(torch.abs(transform_raw[i, :, :pose_length]) > eps)[1].tolist()
        for i in range(num_joints)
    ]


def get_influence_mask(
    transform_raw, pose_length=None, num_params_per_joint=7, eps=1.0e-6
):
    """The same as *map but returns a binary mask."""
    num_joints = transform_raw.shape[0] // num_params_per_joint
    num_params = transform_raw.shape[-1]

    if pose_length is None:
        pose_length = num_params
    assert pose_length <= num_params

    transform_raw = transform_raw.reshape(
        (num_joints, num_params_per_joint, num_params)
    )

    return None

    # return torch.cat([
    #     torch.abs(transform_raw[i, :, :pose_length]) > eps)
    #     for i in range(num_joints)
    # ]


def compute_weights_joints_slow(lbs_weights, lbs_indices, num_joints):
    num_verts = lbs_weights.shape[0]
    weights_joints = torch.zeros((num_joints, num_verts), dtype=torch.float32)
    for i in range(num_verts):
        idx = lbs_indices[i, :]
        weights_joints[idx, i] = lbs_weights[i, :]
    return weights_joints


def load_momentum_cfg(model, lbs_config_txt_path, nr_scaling_params=None):
    def find(l, x):
        try:
            return l.index(x)
        except ValueError:
            return None

    """Load a parameter configuration file"""
    channelNames = ["tx", "ty", "tz", "rx", "ry", "rz", "sc"]
    paramNames = []
    joint_names = []
    for idx, bone in enumerate(model["Skeleton"]["Bones"]):
        joint_names.append(bone["Name"])

    def findJointIndex(x):
        return find(joint_names, x)

    def findParameterIndex(x):
        return find(paramNames, x)

    limits = []

    # create empty result
    transform_triplets = []
    with open(lbs_config_txt_path, "r") as cfg_f:
        lines = cfg_f.readlines()

    # read until end
    for line in lines:
        # strip comments
        line = line[: line.find("#")]

        if line.find("limit") != -1:
            r = re.search("limit ([\\w.]+) (\\w+) (.*)", line)
            if r is None:
                continue

            if len(r.groups()) != 3:
                log.info("Failed to parse limit configuration line :\n   " + line)
                continue

            # find parameter and/or joint index
            fullname = r.groups()[0]
            type = r.groups()[1]
            remaining = r.groups()[2]

            parameterIndex = findParameterIndex(fullname)
            jointName = fullname.split(".")
            jointIndex = findJointIndex(jointName[0])
            channelIndex = -1

            if jointIndex is not None and len(jointName) == 2:
                # find matching channel name
                channelIndex = channelNames.index(jointName[1])
                if channelIndex is None:
                    log.info(
                        "Unknown joint channel name "
                        + jointName[1]
                        + " in parameter configuration line :\n   "
                        + line
                    )
                    continue

            # only parse passive limits for now
            if type == "minmax_passive" or type == "minmax":
                # match [<float> , <float>] <optional weight>
                rp = re.search(
                    "\\[\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*\\](\\s*[-+]?[0-9]*\\.?[0-9]+)?",
                    remaining,
                )

                if len(rp.groups()) != 3:
                    log.info(
                        f"Failed to parse passive limit configuration line :\n {line}"
                    )
                    continue

                minVal = float(rp.groups()[0])
                maxVal = float(rp.groups()[1])
                weightVal = 1.0
                if len(rp.groups()) == 3 and not rp.groups()[2] is None:
                    weightVal = float(rp.groups()[2])

                # result.limits.append([jointIndex * 7 + channelIndex, minVal, maxVal])

                if channelIndex >= 0:
                    valueIndex = jointIndex * 7 + channelIndex
                    limit = {
                        "type": "LimitMinMaxJointValue",
                        "str": fullname,
                        "valueIndex": valueIndex,
                        "limits": [minVal, maxVal],
                        "weight": weightVal,
                    }
                    limits.append(limit)
                else:
                    if parameterIndex is None:
                        log.info(
                            f"Unknown parameterIndex : {fullname}\n  {line} {paramNames} "
                        )
                        continue
                    limit = {
                        "type": "LimitMinMaxParameter",
                        "str": fullname,
                        "parameterIndex": parameterIndex,
                        "limits": [minVal, maxVal],
                        "weight": weightVal,
                    }
                    limits.append(limit)
            # continue the remaining file
            continue

        # check for parameterset definitions and ignore
        if line.find("parameterset") != -1:
            continue

        # use regex to parse definition
        r = re.search("(\w+).(\w+)\s*=\s*(.*)", line)
        if r is None:
            continue

        if len(r.groups()) != 3:
            log.info("Failed to parse parameter configuration line :\n   " + line)
            continue

        # find joint name and parameter
        jointIndex = findJointIndex(r.groups()[0])
        if jointIndex is None:
            log.info(
                "Unknown joint name "
                + r.groups()[0]
                + " in parameter configuration line :\n   "
                + line
            )
            continue

        # find matching channel name
        channelIndex = channelNames.index(r.groups()[1])
        if channelIndex is None:
            log.info(
                "Unknown joint channel name "
                + r.groups()[1]
                + " in parameter configuration line :\n   "
                + line
            )
            continue

        valueIndex = jointIndex * 7 + channelIndex

        # parse parameters
        parameterList = r.groups()[2].split("+")
        for parameterPair in parameterList:
            parameterPair = parameterPair.strip()

            r = re.search("\s*([+-]?[0-9]*\.?[0-9]*)\s\*\s(\w+)\s*", parameterPair)
            if r is None or len(r.groups()) != 2:
                log.info(
                    "Malformed parameter description "
                    + parameterPair
                    + " in parameter configuration line :\n   "
                    + line
                )
                continue

            val = float(r.groups()[0])
            parameter = r.groups()[1]

            # check if parameter exists
            parameterIndex = findParameterIndex(parameter)
            if parameterIndex is None:
                # no, create new parameter entry
                parameterIndex = len(paramNames)
                paramNames.append(parameter)
            transform_triplets.append((valueIndex, parameterIndex, val))

    # set (dense) parameter_transformation matrix
    transform = np.zeros(
        (len(channelNames) * len(joint_names), len(paramNames)), dtype=np.float32
    )
    for i, j, v in transform_triplets:
        transform[i, j] = v

    outputs = {
        "joint_names": paramNames,
        "channel_names": channelNames,
        "limits": limits,
        "transform": transform,
        "transform_offsets": np.zeros(
            (1, len(channelNames) * len(joint_names)), dtype=np.float32
        ),
    }
    # set number of scales automatically
    if nr_scaling_params is None:
        outputs.update(
            nr_scaling_params=len([s for s in paramNames if s.startswith("scale")])
        )
        outputs.update(
            nr_position_params=len(paramNames) - outputs["nr_scaling_params"]
        )

    return outputs
