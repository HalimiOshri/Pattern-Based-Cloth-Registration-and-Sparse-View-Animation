import torch
import torch.nn as nn


class Quaternion:
    """Torch Tensor based Quaternion class"""

    @staticmethod
    def identity(dtype=torch.double):
        """
        Create identity quaternion
        """
        return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype)

    @staticmethod
    def mul(q, r):
        """
        mul two quaternions, expects those to be double tesnors of length 4
        """
        return torch.stack(
            [
                (q * torch.tensor([1.0, 1.0, -1.0, 1.0], dtype=q.dtype)).dot(
                    r[[3, 2, 1, 0]]
                ),
                (q * torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=q.dtype)).dot(
                    r[[2, 3, 0, 1]]
                ),
                (q * torch.tensor([1.0, -1.0, 1.0, 1.0], dtype=q.dtype)).dot(
                    r[[1, 0, 3, 2]]
                ),
                (q * torch.tensor([-1.0, -1.0, -1.0, 1.0], dtype=q.dtype)).dot(
                    r[[0, 1, 2, 3]]
                ),
            ]
        )

    @staticmethod
    def rot(q, v):
        """
        Rotate 3d-vector v given with quaternion q
        """
        axis = q[:3]
        av = torch.cross(axis, v)
        aav = torch.cross(axis, av)
        return v + 2 * (av * q[3] + aav)

    @staticmethod
    def invert(q):
        """
        Get the inverse of quaternion q
        """
        return (
            q * torch.tensor([-1.0, -1.0, -1.0, 1.0], dtype=q.dtype) * (1.0 / q.dot(q))
        )

    @staticmethod
    def fromAxisAngle(axis, angle):
        """
        Generate a quaternion representing a rotation around axis by angle
        """
        s = torch.sin(angle * 0.5)
        c = torch.cos(angle * 0.5).view([1])
        return torch.cat((axis * s, c), 0)

    @staticmethod
    def fromXYZ(angles):
        """
        Generate a quaternion representing a rotation defined by a XYZ-Euler
        rotation.
        This is faster than creating three separate quaternions and muling
        them.
        """
        rc = torch.cos(angles * torch.tensor([-0.5, 0.5, 0.5], dtype=angles.dtype))
        rs = torch.sin(angles * torch.tensor([-0.5, 0.5, 0.5], dtype=angles.dtype))

        return torch.stack(
            [
                -rs[0] * rc[1] * rc[2] - rc[0] * rs[1] * rs[2],
                rc[0] * rs[1] * rc[2] - rs[0] * rc[1] * rs[2],
                rc[0] * rc[1] * rs[2] + rs[0] * rs[1] * rc[2],
                rc[0] * rc[1] * rc[2] - rs[0] * rs[1] * rs[2],
            ]
        )

    @staticmethod
    def toMatrix(q):
        """
        Convert quaternion q to 3x3 rotation matrix
        """
        result = torch.empty([3, 3], dtype=q.dtype)

        tx = q[0] * 2.0
        ty = q[1] * 2.0
        tz = q[2] * 2.0
        twx = tx * q[3]
        twy = ty * q[3]
        twz = tz * q[3]
        txx = tx * q[0]
        txy = ty * q[0]
        txz = tz * q[0]
        tyy = ty * q[1]
        tyz = tz * q[1]
        tzz = tz * q[2]

        result[0, 0] = 1.0 - (tyy + tzz)
        result[0, 1] = txy - twz
        result[0, 2] = txz + twy
        result[1, 0] = txy + twz
        result[1, 1] = 1.0 - (txx + tzz)
        result[1, 2] = tyz - twx
        result[2, 0] = txz - twy
        result[2, 1] = tyz + twx
        result[2, 2] = 1.0 - (txx + tyy)

        return result

    @staticmethod
    def toMatrixBatch(q):
        tx = q[..., 0] * 2.0
        ty = q[..., 1] * 2.0
        tz = q[..., 2] * 2.0
        twx = tx * q[..., 3]
        twy = ty * q[..., 3]
        twz = tz * q[..., 3]
        txx = tx * q[..., 0]
        txy = ty * q[..., 0]
        txz = tz * q[..., 0]
        tyy = ty * q[..., 1]
        tyz = tz * q[..., 1]
        tzz = tz * q[..., 2]

        return torch.stack(
            (
                torch.stack((1.0 - (tyy + tzz), txy + twz, txz - twy), dim=2),
                torch.stack((txy - twz, 1.0 - (txx + tzz), tyz + twx), dim=2),
                torch.stack((txz + twy, tyz - twx, 1.0 - (txx + tyy)), dim=2),
            ),
            dim=3,
        )

    @staticmethod
    def toMatrixBatchDim1(q):
        tx = q[..., 0] * 2.0
        ty = q[..., 1] * 2.0
        tz = q[..., 2] * 2.0
        twx = tx * q[..., 3]
        twy = ty * q[..., 3]
        twz = tz * q[..., 3]
        txx = tx * q[..., 0]
        txy = ty * q[..., 0]
        txz = tz * q[..., 0]
        tyy = ty * q[..., 1]
        tyz = tz * q[..., 1]
        tzz = tz * q[..., 2]

        return torch.stack(
            (
                torch.stack((1.0 - (tyy + tzz), txy + twz, txz - twy), dim=1),
                torch.stack((txy - twz, 1.0 - (txx + tzz), tyz + twx), dim=1),
                torch.stack((txz + twy, tyz - twx, 1.0 - (txx + tyy)), dim=1),
            ),
            dim=2,
        )

    # @staticmethod
    # def batchMul(q, r):
    #     """
    #     mul two quaternions, expects those to be double tesnors of length 4

    #     Args:
    #         q: N x K x 4 quaternions
    #         r: N x K x 4 quaternions

    #     Returns:
    #         N x K x 4 multiplied quaternions
    #     """
    #     return torch.stack([
    #             torch.sum(torch.mul(torch.mul(q, torch.tensor([[[ 1., 1.,-1., 1.]]], dtype=q.dtype)), r[:, :, (3, 2, 1, 0)]), dim=-1),
    #             torch.sum(torch.mul(torch.mul(q, torch.tensor([[[-1., 1., 1., 1.]]], dtype=q.dtype)), r[:, :, (2, 3, 0, 1)]), dim=-1),
    #             torch.sum(torch.mul(torch.mul(q, torch.tensor([[[ 1.,-1., 1., 1.]]], dtype=q.dtype)), r[:, :, (1, 0, 3, 2)]), dim=-1),
    #             torch.sum(torch.mul(torch.mul(q, torch.tensor([[[-1.,-1.,-1., 1.]]], dtype=q.dtype)), r[:, :, (0, 1, 2, 3)]), dim=-1)
    #     ], dim = 2)

    @staticmethod
    def batchMul(q, r):
        """
        mul two quaternions, expects those to be double tesnors of length 4

        Args:
            q: N x K x 4 quaternions
            r: N x K x 4 quaternions

        Returns:
            N x K x 4 multiplied quaternions
        """
        return torch.stack(
            [
                torch.sum(
                    torch.mul(
                        torch.mul(
                            q,
                            torch.tensor(
                                [[[1.0, 1.0, -1.0, 1.0]]],
                                dtype=q.dtype,
                                device=q.device,
                            ),
                        ),
                        r[:, :, (3, 2, 1, 0)],
                    ),
                    dim=-1,
                ),
                torch.sum(
                    torch.mul(
                        torch.mul(
                            q,
                            torch.tensor(
                                [[[-1.0, 1.0, 1.0, 1.0]]],
                                dtype=q.dtype,
                                device=q.device,
                            ),
                        ),
                        r[:, :, (2, 3, 0, 1)],
                    ),
                    dim=-1,
                ),
                torch.sum(
                    torch.mul(
                        torch.mul(
                            q,
                            torch.tensor(
                                [[[1.0, -1.0, 1.0, 1.0]]],
                                dtype=q.dtype,
                                device=q.device,
                            ),
                        ),
                        r[:, :, (1, 0, 3, 2)],
                    ),
                    dim=-1,
                ),
                torch.sum(
                    torch.mul(
                        torch.mul(
                            q,
                            torch.tensor(
                                [[[-1.0, -1.0, -1.0, 1.0]]],
                                dtype=q.dtype,
                                device=q.device,
                            ),
                        ),
                        r[:, :, (0, 1, 2, 3)],
                    ),
                    dim=-1,
                ),
            ],
            dim=2,
        )

    @staticmethod
    def batchRot(q, v):
        """
        Rotate 3d-vector v given with quaternion q

        Args:
            q: N x K x 4 quaternions
            v: N x K x 3 vectors
        
        Returns:
            N x K x 3 rotated vectors
        """
        av = torch.cross(q[:, :, :3], v, dim=2)
        aav = torch.cross(q[:, :, :3], av, dim=2)
        return torch.add(v, 2 * torch.add(torch.mul(av, q[:, :, 3].unsqueeze(2)), aav))

    # @staticmethod
    # def batchInvert(q):
    #     """
    #     Get the inverse of quaternion q

    #     Args:
    #         q: N x K x 4 quaternions

    #     Returns:
    #         N x K x 4 inverted quaternions
    #     """
    #     return q * torch.tensor([-1., -1., -1., 1.], dtype=q.dtype) * (torch.reciprocal(torch.sum(q * q, dim=2).unsqueeze(2)))

    @staticmethod
    def batchInvert(q):
        """
        Get the inverse of quaternion q

        Args:
            q: N x K x 4 quaternions

        Returns:
            N x K x 4 inverted quaternions
        """
        return (
            q
            * torch.tensor([-1.0, -1.0, -1.0, 1.0], dtype=q.dtype, device=q.device)
            * (torch.reciprocal(torch.sum(q * q, dim=2).unsqueeze(2)))
        )

    # @staticmethod
    # def batchFromXYZ(r):
    #     """
    #     Generate a quaternion representing a rotation defined by a XYZ-Euler
    #     rotation.

    #     Args:
    #         r: N x K x 3 rotation vectors

    #     Returns:
    #         N x K x 4 quaternions
    #     """
    #     rm = r * torch.tensor([[[-0.5, 0.5, 0.5]]], dtype=r.dtype)
    #     rc = torch.cos(rm)
    #     rs = torch.sin(rm)

    #     return torch.stack([
    #         torch.sub(torch.mul(torch.neg(rs[:, :, 0]), torch.mul(rc[:, :, 1], rc[:, :, 2])),
    #                   torch.mul(rc[:, :, 0], torch.mul(rs[:, :, 1], rs[:, :, 2]))),
    #         torch.sub(torch.mul(rc[:, :, 0], torch.mul(rs[:, :, 1], rc[:, :, 2])),
    #                   torch.mul(rs[:, :, 0], torch.mul(rc[:, :, 1], rs[:, :, 2]))),
    #         torch.add(torch.mul(rc[:, :, 0], torch.mul(rc[:, :, 1], rs[:, :, 2])),
    #                   torch.mul(rs[:, :, 0], torch.mul(rs[:, :, 1], rc[:, :, 2]))),
    #         torch.sub(torch.mul(rc[:, :, 0], torch.mul(rc[:, :, 1], rc[:, :, 2])),
    #                   torch.mul(rs[:, :, 0], torch.mul(rs[:, :, 1], rs[:, :, 2])))
    #     ], dim=2)

    @staticmethod
    def batchFromXYZ(r):
        """
        Generate a quaternion representing a rotation defined by a XYZ-Euler
        rotation.

        Args:
            r: N x K x 3 rotation vectors

        Returns:
            N x K x 4 quaternions
        """
        rm = r * torch.tensor([[[-0.5, 0.5, 0.5]]], dtype=r.dtype, device=r.device)
        rc = torch.cos(rm)
        rs = torch.sin(rm)

        return torch.stack(
            [
                torch.sub(
                    torch.mul(
                        torch.neg(rs[:, :, 0]), torch.mul(rc[:, :, 1], rc[:, :, 2])
                    ),
                    torch.mul(rc[:, :, 0], torch.mul(rs[:, :, 1], rs[:, :, 2])),
                ),
                torch.sub(
                    torch.mul(rc[:, :, 0], torch.mul(rs[:, :, 1], rc[:, :, 2])),
                    torch.mul(rs[:, :, 0], torch.mul(rc[:, :, 1], rs[:, :, 2])),
                ),
                torch.add(
                    torch.mul(rc[:, :, 0], torch.mul(rc[:, :, 1], rs[:, :, 2])),
                    torch.mul(rs[:, :, 0], torch.mul(rs[:, :, 1], rc[:, :, 2])),
                ),
                torch.sub(
                    torch.mul(rc[:, :, 0], torch.mul(rc[:, :, 1], rc[:, :, 2])),
                    torch.mul(rs[:, :, 0], torch.mul(rs[:, :, 1], rs[:, :, 2])),
                ),
            ],
            dim=2,
        )

    @staticmethod
    def batchMatrixFromXYZ(r):
        """
        Generate a matrix representing a rotation defined by a XYZ-Euler
        rotation.

        Args:
            r: N x 3 rotation vectors

        Returns:
            N x 3 x 3 rotation matrices
        """
        rc = torch.cos(r)
        rs = torch.sin(r)
        cx = rc[:, 0]
        cy = rc[:, 1]
        cz = rc[:, 2]
        sx = rs[:, 0]
        sy = rs[:, 1]
        sz = rs[:, 2]

        result = torch.stack(
            (
                cy * cz,
                -cx * sz + sx * sy * cz,
                sx * sz + cx * sy * cz,
                cy * sz,
                cx * cz + sx * sy * sz,
                -sx * cz + cx * sy * sz,
                -sy,
                sx * cy,
                cx * cy,
            ),
            dim=1,
        ).view(-1, 3, 3)
        return result

    @staticmethod
    def batchQuatFromMatrix(m):
        """
        :param m: B*3*3
        :return: B*4, order xyzw
        """
        assert len(m.shape) == 3
        b, j, k = m.shape
        assert j == 3
        assert k == 3
        result = torch.zeros((b, 4), dtype=torch.float32).to(m.device)
        S = torch.zeros((b,), dtype=torch.float32).to(m.device)

        m00 = m[:, 0, 0]
        m01 = m[:, 0, 1]
        m02 = m[:, 0, 2]
        m10 = m[:, 1, 0]
        m11 = m[:, 1, 1]
        m12 = m[:, 1, 2]
        m20 = m[:, 2, 0]
        m21 = m[:, 2, 1]
        m22 = m[:, 2, 2]

        tr = m00 + m11 + m22
        flag = tr > 0
        S[flag] = 2 * torch.sqrt(1 + tr[flag])
        result[flag, 0] = (m21 - m12)[flag] / S[flag]
        result[flag, 1] = (m02 - m20)[flag] / S[flag]
        result[flag, 2] = (m10 - m01)[flag] / S[flag]
        result[flag, 3] = 0.25 * S[flag]

        flag = ~flag & (m00 > m11) & (m00 > m22)
        S[flag] = 2 * torch.sqrt(1.0 + m00[flag] - m11[flag] - m22[flag])
        result[flag, 0] = 0.25 * S[flag]
        result[flag, 1] = (m01 + m10)[flag] / S[flag]
        result[flag, 2] = (m02 + m20)[flag] / S[flag]
        result[flag, 3] = (m21 - m12)[flag] / S[flag]

        flag = ~flag & (m11 > m22)
        S[flag] = 2 * torch.sqrt(1.0 + m11[flag] - m00[flag] - m22[flag])
        result[flag, 0] = (m01 + m10)[flag] / S[flag]
        result[flag, 1] = 0.25 * S[flag]
        result[flag, 2] = (m12 + m21)[flag] / S[flag]
        result[flag, 3] = (m02 - m20)[flag] / S[flag]

        flag = ~flag
        S[flag] = 2 * torch.sqrt(1.0 + m22[flag] - m00[flag] - m11[flag])
        result[flag, 0] = (m02 + m20)[flag] / S[flag]
        result[flag, 1] = (m12 + m21)[flag] / S[flag]
        result[flag, 2] = 0.25 * S[flag]
        result[flag, 3] = (m10 - m01)[flag] / S[flag]

        return result


class RodriguesVecBatch(nn.Module):
    def __init__(self):
        super(RodriguesVecBatch, self).__init__()
        self.register_buffer("eye", (torch.eye(3)))
        self.register_buffer("zero", (torch.zeros(1,)))
        # mat = torch.zeros((nbat,3,3),dtype=torch.float32,device=r.device,requires_grad=True)

    def forward(
        self, v0, v1
    ):  # assuming v0 and v1 are already normalized, compute matrix aligning v0 to v1
        nbat = v0.size(0)
        cosn = (v0 * v1).sum(dim=1, keepdim=True).unsqueeze(2)
        # r = v0.cross(v1,dim=1)
        r = v1.cross(v0, dim=1)
        sinn = r.pow(2).sum(1, keepdim=True).sqrt().unsqueeze(2)
        rn = r.unsqueeze(2) / (sinn + 1e-10)
        R = cosn * self.eye.unsqueeze(0).expand(nbat, 3, 3)
        R = R + (1.0 - cosn) * rn.bmm(rn.permute(0, 2, 1))
        R[:, 0, 1] = R[:, 0, 1] + rn[:, 2, 0] * sinn[:, 0, 0]
        R[:, 1, 0] = R[:, 0, 1] - rn[:, 2, 0] * sinn[:, 0, 0]
        R[:, 0, 2] = R[:, 0, 2] - rn[:, 1, 0] * sinn[:, 0, 0]
        R[:, 2, 0] = R[:, 2, 0] + rn[:, 1, 0] * sinn[:, 0, 0]
        R[:, 1, 2] = R[:, 1, 2] + rn[:, 0, 0] * sinn[:, 0, 0]
        R[:, 2, 1] = R[:, 2, 1] - rn[:, 0, 0] * sinn[:, 0, 0]
        return R


class RodriguesBatch(nn.Module):
    def __init__(self):
        super(RodriguesBatch, self).__init__()
        self.register_buffer("eye", (torch.eye(3)))
        self.register_buffer("zero", (torch.zeros(1,)))

    def forward(self, r):
        # pdb.set_trace()
        nbat = r.size(0)
        n = ((r * r).sum(dim=1, keepdim=True) + 1e-10).sqrt()
        rn = torch.div(r, n).unsqueeze(2)

        cosn = torch.cos(n).unsqueeze(2)
        sinn = torch.sin(n).unsqueeze(2)
        R = cosn * self.eye.unsqueeze(0).expand(nbat, 3, 3)
        R = R + (1.0 - cosn) * rn.bmm(rn.permute(0, 2, 1))

        R[:, 0, 1] = R[:, 0, 1] + rn[:, 2, 0] * sinn[:, 0, 0]
        R[:, 1, 0] = R[:, 0, 1] - rn[:, 2, 0] * sinn[:, 0, 0]
        R[:, 0, 2] = R[:, 0, 2] - rn[:, 1, 0] * sinn[:, 0, 0]
        R[:, 2, 0] = R[:, 2, 0] + rn[:, 1, 0] * sinn[:, 0, 0]
        R[:, 1, 2] = R[:, 1, 2] + rn[:, 0, 0] * sinn[:, 0, 0]
        R[:, 2, 1] = R[:, 2, 1] - rn[:, 0, 0] * sinn[:, 0, 0]
        return R


class NormalComputer(nn.Module):
    def __init__(self, height, width, maskin=None):
        super(NormalComputer, self).__init__()
        # self.register_buffer('eye', (torch.eye(3)))
        # self.register_buffer('zero', (torch.zeros(1,)))

        patchttnum = 5  # neighbor + self
        patchmatch_uvpos = np.zeros((height, width, patchttnum, 2), dtype=np.int32)
        vec_standuv = (
            np.indices((height, width))
            .swapaxes(0, 2)
            .swapaxes(0, 1)
            .astype(np.int32)
            .reshape(height, width, 1, 2)
        )
        patchmatch_uvpos = patchmatch_uvpos + vec_standuv
        localpatchcoord = np.zeros((patchttnum, 2), dtype=np.int32)
        localpatchcoord = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]]).astype(
            np.int32
        )

        patchmatch_uvpos = patchmatch_uvpos + localpatchcoord.reshape(
            1, 1, patchttnum, 2
        )
        patchmatch_uvpos[..., 0] = np.clip(patchmatch_uvpos[..., 0], 0, height - 1)
        patchmatch_uvpos[..., 1] = np.clip(patchmatch_uvpos[..., 1], 0, width - 1)

        # geoemtry mask , apply simiilar to texture mask
        # mesh_mask_int = mesh_mask.reshape(height,width).astype(np.int32)
        if maskin is None:
            maskin = np.ones((height, width), dtype=np.int32)
        mesh_mask_int = maskin.reshape(height, width).astype(
            np.int32
        )  # using all pixel valid mask; can use a tailored mask
        patchmatch_mask = mesh_mask_int[
            patchmatch_uvpos[..., 0], patchmatch_uvpos[..., 1]
        ].reshape(height, width, patchttnum, 1)
        patch_indicemap = (
            patchmatch_uvpos * patchmatch_mask + (1 - patchmatch_mask) * vec_standuv
        )

        tensor_patch_geoindicemap = torch.from_numpy(patch_indicemap).long()
        tensor_patch_geoindicemap1d = (
            tensor_patch_geoindicemap[..., 0] * width
            + tensor_patch_geoindicemap[..., 1]
        )

        self.register_buffer("tensor_patch_geoindicemap1d", tensor_patch_geoindicemap1d)
        # tensor_patchmatch_uvpos = torch.from_numpy(patchmatch_uvpos).long()
        # tensor_vec_standuv = torch.from_numpy(vec_standuv).long()

    def forward(self, t_georecon):  # in: N 3 H W
        # pdb.set_trace()
        geometry_in = index_selection_nd(
            t_georecon.view(t_georecon.size(0), t_georecon.size(1), -1),
            self.tensor_patch_geoindicemap1d,
            2,
        ).permute(0, 2, 3, 4, 1)
        normal = (geometry_in[..., 0, :] - geometry_in[..., 4, :]).cross(
            geometry_in[..., 1, :] - geometry_in[..., 4, :], dim=3
        )
        normal = normal + (geometry_in[..., 1, :] - geometry_in[..., 4, :]).cross(
            geometry_in[..., 2, :] - geometry_in[..., 4, :], dim=3
        )
        normal = normal + (geometry_in[..., 2, :] - geometry_in[..., 4, :]).cross(
            geometry_in[..., 3, :] - geometry_in[..., 4, :], dim=3
        )
        normal = normal + (geometry_in[..., 3, :] - geometry_in[..., 4, :]).cross(
            geometry_in[..., 0, :] - geometry_in[..., 4, :], dim=3
        )
        normal = normal / torch.clamp(
            normal.pow(2).sum(3, keepdim=True).sqrt(), min=1e-6
        )
        return normal.permute(0, 3, 1, 2)


def pointcloud_rigid_registration(src_pointcloud, dst_pointcloud):
    """
    Calculate RT and residual L2 loss for two pointclouds
    :param src_pointcloud: x (b, v, 3)
    :param dst_pointcloud: y (b, v, 3)
    :return: loss, R, t  s.t. ||Rx+t-y||_2^2 minimal.
    """
    if len(src_pointcloud.shape) == 2:
        src_pointcloud = src_pointcloud.unsqueeze(0)
    if len(dst_pointcloud.shape) == 2:
        dst_pointcloud = dst_pointcloud.unsqueeze(0)
    bn = src_pointcloud.shape[0]

    assert src_pointcloud.shape == dst_pointcloud.shape
    assert src_pointcloud.shape[2] == 3

    X = src_pointcloud - src_pointcloud.mean(dim=1, keepdim=True)
    Y = dst_pointcloud - dst_pointcloud.mean(dim=1, keepdim=True)

    XYT = torch.einsum("nji,njk->nik", X, Y)
    muX = src_pointcloud.mean(dim=1)
    muY = dst_pointcloud.mean(dim=1)

    R = torch.zeros((bn, 3, 3), dtype=src_pointcloud.dtype).to(src_pointcloud.device)
    t = torch.zeros((bn, 1, 3), dtype=src_pointcloud.dtype).to(src_pointcloud.device)
    loss = torch.zeros((bn,), dtype=src_pointcloud.dtype).to(src_pointcloud.device)

    for i in range(bn):
        u_, s_, v_ = torch.svd(XYT[i, :, :])
        detvut = torch.det(v_.mm(u_.t()))
        diag_m = torch.ones_like(s_)
        diag_m[-1] = detvut

        r_ = v_.mm(torch.diag(diag_m)).mm(u_.t())
        t_ = muY[i, :] - r_.mm(muX[i, :, None])[:, 0]

        R[i, :, :] = r_
        t[i, 0, :] = t_
        loss[i] = (torch.einsum("ij,nj->ni", r_, X[i]) - Y[i]).pow(2).sum(1).mean(0)

    return loss.mean(0), R, t


def pointcloud_rigid_registration_balanced(src_pointcloud, dst_pointcloud, weight):
    """
    Calculate RT and residual L2 loss for two pointclouds
    :param src_pointcloud: x (b, v, 3)
    :param dst_pointcloud: y (b, v, 3)
    :param weight:  (v, ), duplication of vertices
    :return: loss, R, t  s.t. ||w(Rx+t-y)||_2^2 minimal.
    """
    if len(src_pointcloud.shape) == 2:
        src_pointcloud = src_pointcloud.unsqueeze(0)
    if len(dst_pointcloud.shape) == 2:
        dst_pointcloud = dst_pointcloud.unsqueeze(0)
    bn = src_pointcloud.shape[0]

    assert src_pointcloud.shape == dst_pointcloud.shape
    assert src_pointcloud.shape[2] == 3
    assert src_pointcloud.shape[1] == weight.shape[0]
    assert len(weight.shape) == 1
    w = weight[None, :, None]

    def s1(a):
        return a.sum(dim=1, keepdim=True)

    w2 = w.pow(2)
    sw2 = s1(w2)
    X = src_pointcloud
    Y = dst_pointcloud

    wXYT = torch.einsum("nji,njk->nik", w2 * (sw2 - w2) * X, Y)
    U, s, V = batch_svd(wXYT)
    UT = U.permute(0, 2, 1).contiguous()
    det = batch_det(V.bmm(UT))
    diag = torch.ones_like(s).to(s.device)
    diag[:, -1] = det

    R = V.bmm(batch_diag(diag)).bmm(UT)
    RX = torch.einsum("bij,bnj->bni", R, X)
    t = torch.sum(w * (Y - RX), dim=1, keepdim=True) / sw2
    loss = w * (RX + t - Y)
    loss = F.mse_loss(loss, torch.zeros_like(loss)) * 3

    return loss, R, t


def batch_dot(x, y):
    assert x.shape == y.shape
    assert len(x.shape) == 2
    return torch.einsum("ni,ni->n", x, y)


def batch_svd(x):
    assert len(x.shape) == 3
    bn, m, n = x.shape
    U = torch.zeros((bn, m, m), dtype=torch.float32).to(x.device)
    s = torch.zeros((bn, min(n, m)), dtype=torch.float32).to(x.device)
    V = torch.zeros((bn, n, n), dtype=torch.float32).to(x.device)
    for i in range(bn):
        u_, s_, v_ = torch.svd(x[i, :, :])
        U[i] = u_
        s[i] = s_
        V[i] = v_
    return U, s, V


def batch_diag(x):
    if len(x.shape) == 2:
        bn, n = x.shape
        res = torch.zeros((bn, n, n), dtype=torch.float32).to(x.device)
        res[:, range(n), range(n)] = x
        return res
    elif len(x.shape) == 3:
        assert x.shape[1] == x.shape[2]
        n = x.shape[1]
        return x[:, range(n), range(n)]
    else:
        raise ValueError("dim of batch_diag should be 2 or 3")


def batch_det(x):
    assert len(x.shape) == 3
    assert x.shape[1] == x.shape[2]
    bn, _, _ = x.shape
    res = torch.zeros((bn,), dtype=torch.float32).to(x.device)
    for i in range(bn):
        res[i] = torch.det(x[i])
    return res
