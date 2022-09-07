"""Basic rendering utilities."""

import torch
import torch.nn.functional as F

from drtk.render_cuda_ext import render_forward
from drtk.rasterizer import rasterize_packed as rasterize

# TODO: should we have a more pytorch-style lookup?
from sklearn.neighbors import KDTree

import numpy as np

import logging

logger = logging.getLogger(__name__)


def render_index_image(uv, uv_faces, uv_size, rounding=False):
    """Render an index image in the uv space.

    Args:
        uv: tensor of size [N_uv, 2], uv coordinates in [0;1]
        uv_faces: tensor of size [F_uv, 3], uv triangles
        uv_size: int, the size of the uv image

    Returns:
        a tuple (index_img, bary_img), where
            index_img is a int tensor of shape [uv_size, uv_size]
            bary_img is a float tensor of shape [uv_size, uv_size, 3] of
            barycentric coordinates

    """
    device = uv.device

    uv = uv.to(device)
    uv_faces = uv_faces.to(device)

    B, H, W = 1, uv_size, uv_size

    if rounding:
        verts = torch.floor(uv * uv_size)
    else:
        verts = uv * uv_size - 0.5
    # verts = uv * uv_size - 0.5
    # verts = uv * uv_size
    verts_ones = torch.ones((verts.shape[0], 1), device=device)
    verts = torch.cat([verts, verts_ones], axis=-1)[np.newaxis]
    verts = verts.contiguous()

    index_img = torch.empty((B, H, W), dtype=torch.int32, device=device)
    depth_img = torch.empty((B, H, W), dtype=torch.float32, device=device)
    packed_index_img = torch.empty((1, H, W, 2), dtype=torch.int32, device=device)

    rasterize(verts, uv_faces, depth_img, index_img, packed_index_img)

    mask = torch.isnan(depth_img)
    depth_img[mask] = -1
    index_img[mask] = -1

    bary_img = torch.empty((B, H, W, 3), dtype=torch.float32, device=device)
    uv_img = torch.empty((B, H, W, 2), dtype=torch.float32, device=device)
    bary_img[mask] = -1
    # TODO: can we avoid doing this fully?
    render_forward(
        verts,
        uv,
        None,
        uv_faces,
        uv_faces,
        index_img,
        depth_img,
        bary_img,
        uv_img,
        None,
    )
    return index_img[0].to(torch.int64), bary_img[0].detach()


def render_vertex_index_image(
    uv, faces_uv, v2uv, uv_size, dtype=torch.int64, rounding=False
):
    """Renders a vertex index image.

    Args:
        uv: [N_uv, 2] uv coordinates
        faces_uv: [F, 3] uv faces
        v2uv: [N_v, K] mapping from mesh vertices to two (at most) uv coordinates

    Returns:
        a tuple (vertex_index_img, bary_img), where
            vertex_index_img: is an int tensor of shape [uv_size, uv_size, 3] and
                contains original vertices.
            bary_img: is a float tensor of shape [uv_size, uv_size, 3] and contains
                barycentrics coordinates of the vertices within triangles.

    """
    device = uv.device

    num_uvs = uv.shape[0]
    num_verts = v2uv.shape[0]

    uv = uv.to(device)
    faces_uv = faces_uv.to(device)
    v2uv = v2uv.to(device)

    uv2v = torch.zeros(num_uvs, dtype=torch.int64, device=device)
    # NOTE: this is not the most efficient thing, maybe just remap?
    for c in range(v2uv.shape[-1]):
        uv2v[v2uv[:, c]] = torch.arange(num_verts, device=device)

    # get the index image and the barycentric coordinates
    face_uv_index_img, bary_img = render_index_image(
        uv, faces_uv, uv_size, rounding=rounding
    )
    mask = face_uv_index_img != -1
    # image vertices. TODO: should it be 64?
    vertex_index_img = -1 * torch.ones(
        (uv_size, uv_size, 3), dtype=dtype, device=device
    )
    # get in UV space
    vertex_indices = faces_uv[face_uv_index_img[mask]].to(torch.int64)
    # map back to the original mesh vertex space
    vertex_index_img[mask] = uv2v[vertex_indices].to(dtype)

    return vertex_index_img, bary_img


def render_vertex_index_image_ext(
    uv_coords, uv_faces, v2uv, uv_size, dtype=torch.int64
):
    index_image, bary_image = render_vertex_index_image(
        uv_coords, uv_faces, v2uv, uv_size, dtype=dtype, rounding=True,
    )

    # defined pixels
    mask_def = torch.all(index_image > -1, axis=-1)
    # undefined pixels
    mask_undef = ~mask_def
    # put all the uvs into a search structure
    lookup = KDTree(uv_coords.cpu().numpy())

    ys, xs = torch.where(mask_undef)
    # TODO: check this
    uvs_absent = torch.stack([xs, ys], axis=-1) / float(uv_size)
    # looking up nearest neighbours
    nbs_idxs = lookup.query(uvs_absent.cpu().numpy(), return_distance=False)[..., 0]
    # their (normalized to [-1, 1]) UV coords
    nbs_coords = uv_coords[nbs_idxs]

    def _sample_uv(image, coords):
        return (
            F.grid_sample(
                input=image.permute((2, 0, 1))[np.newaxis].to(torch.float32),
                grid=(2.0 * coords - 1.0)[np.newaxis, :, np.newaxis],
                align_corners=False,
                mode="nearest",
            )[0, ..., 0]
            .transpose(1, 0)
            .to(image.dtype)
        )

    index_undef = _sample_uv(index_image, nbs_coords)
    bary_undef = _sample_uv(bary_image, nbs_coords)

    index_image_ext = index_image.clone()
    index_image_ext[mask_undef] = index_undef

    bary_image_ext = bary_image.clone()
    bary_image_ext[mask_undef] = bary_undef

    return index_image_ext, bary_image_ext


def render_vertex_image(verts, uv_coords, uv_faces, uv_mapping, uv_size):
    """Renders a uv image of xyz coordinates."""
    index_image, bary_image = render_vertex_index_image(
        uv_coords, uv_faces, uv_mapping, uv_size
    )
    index_image = index_image.to(torch.int64)
    mask = index_image[..., 0] != -1

    verts_uv_masked_mean = (
        (bary_image[mask][..., np.newaxis] * verts[:, index_image[mask], :])
        .sum(axis=2)
        .permute(0, 2, 1)
    )

    verts_uv_mean = torch.zeros(
        (verts.shape[0], verts.shape[-1], uv_size, uv_size),
        dtype=verts.dtype,
        device=verts.device,
    )
    verts_uv_mean[:, :, mask] = verts_uv_masked_mean
    return verts_uv_mean


def values_to_uv(values, index_img, bary_img):
    uv_size = index_img.shape[0]
    index_mask = torch.all(index_img != -1, axis=-1)
    idxs_flat = index_img[index_mask].to(torch.int64)
    bary_flat = bary_img[index_mask].to(torch.float32)
    # NOTE: here we assume
    values_flat = torch.sum(
        values[:, idxs_flat].permute(0, 3, 1, 2) * bary_flat, axis=-1
    )
    values_uv = torch.zeros(
        values.shape[0],
        values.shape[-1],
        uv_size,
        uv_size,
        dtype=values.dtype,
        device=values.device,
    )
    values_uv[:, :, index_mask] = values_flat
    return values_uv


def sample_uv(values_uv, uv_coords, v2uv, align_corners=True, return_var=False):
    batch_size = values_uv.shape[0]
    uv_coords_norm = (uv_coords * 2.0 - 1.0)[np.newaxis, :, np.newaxis].expand(
        batch_size, -1, -1, -1
    )
    values = (
        F.grid_sample(values_uv, uv_coords_norm, align_corners=align_corners)
        .squeeze(-1)
        .permute((0, 2, 1))
    )
    values_duplicate = values[:, v2uv]
    values = values_duplicate.mean(2)

    if return_var:
        values_var = values_duplicate.var(2)
        return values, values_var

    return values
