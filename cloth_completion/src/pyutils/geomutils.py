import numpy as np
import torch
import torch as th

import torch.nn as nn
import torch.nn.functional as F

# from renderlayer.projection import project_points
# import igl


def index(x, I, dim):
    target_shape = [*x.shape]
    del target_shape[dim]
    target_shape[dim:dim] = [*I.shape]
    return x.index_select(dim, I.view(-1)).reshape(target_shape)


def face_normals(v, vi):
    b = v.shape[0]
    vi = vi.expand(b, -1, -1)

    p0 = th.stack([index(v[i], vi[i, :, 0], 0) for i in range(b)])
    p1 = th.stack([index(v[i], vi[i, :, 1], 0) for i in range(b)])
    p2 = th.stack([index(v[i], vi[i, :, 2], 0) for i in range(b)])
    v0 = p1 - p0
    v1 = p2 - p0
    n = th.cross(v0, v1, dim=-1)
    norm = th.norm(n, dim=-1, keepdim=True)
    norm[norm < 1e-5] = 1
    n /= norm
    return n


def vert_normals(v, vi):
    fnorms = face_normals(v, vi)
    fnorms = fnorms[:, :, None].expand(-1, -1, 3, -1).reshape(fnorms.shape[0], -1, 3)
    vi_flat = vi.view(vi.shape[0], -1).expand(v.shape[0], -1)
    vnorms = th.zeros_like(v)

    for j in range(3):
        vnorms[..., j].scatter_add_(1, vi_flat, fnorms[..., j])
    norm = th.norm(vnorms, dim=-1, keepdim=True)
    norm[norm < 1e-5] = 1
    vnorms /= norm
    return vnorms


def face_normals_v2(v, vi):
    pts = v[:, vi]
    v0 = pts[:, :, 1] - pts[:, :, 0]
    v1 = pts[:, :, 2] - pts[:, :, 0]
    n = th.cross(v0, v1, dim=-1)
    norm = th.norm(n, dim=-1, keepdim=True)
    # norm[norm < 1e-5] = 1
    eps_mask = (norm < 1e-5)
    norm = (~eps_mask) * norm + eps_mask * 1.0
    # n /= norm
    n = n / norm
    return n


def vert_normals_v2(v, vi):
    fnorms = face_normals_v2(v, vi)
    fnorms = fnorms[:, :, None].expand(-1, -1, 3, -1).reshape(fnorms.shape[0], -1, 3)
    vi_flat = vi.view(1, -1).expand(v.shape[0], -1)
    vnorms = th.zeros_like(v)

    for j in range(3):
        vnorms[..., j].scatter_add_(1, vi_flat, fnorms[..., j])
    norm = th.norm(vnorms, dim=-1, keepdim=True)
    # norm[norm < 1e-5] = 1
    eps_mask = (norm < 1e-5)
    norm = (~eps_mask) * norm + eps_mask * 1.0
    # vnorms /= norm
    vnorms = vnorms / norm
    return vnorms


def compute_view_cos(verts, faces, camera_pos):
    vn = F.normalize(vert_normals_v2(verts, faces), dim=-1)
    v2c = F.normalize(verts - camera_pos[:, np.newaxis], dim=-1)
    return th.einsum("bnd,bnd->bn", vn, v2c)


def compute_tbn_uv(tri_xyz, tri_uv, eps=1e-5):
    """Compute tangents, bitangents, normals.
    Args:
        tri_xyz: [B,N,3,3] vertex coordinates
        tri_uv: [N,2] texture coordinates
    Returns:
        tangents, bitangents, normals
    """

    tri_uv = tri_uv[np.newaxis]

    v01 = tri_xyz[:, :, 1] - tri_xyz[:, :, 0]
    v02 = tri_xyz[:, :, 2] - tri_xyz[:, :, 0]

    normals = th.cross(v01, v02, dim=-1)
    normals = normals / th.norm(normals, dim=-1, keepdim=True).clamp(min=eps)

    vt01 = tri_uv[:, :, 1] - tri_uv[:, :, 0]
    vt02 = tri_uv[:, :, 2] - tri_uv[:, :, 0]

    f = 1.0 / (vt01[..., 0] * vt02[..., 1] - vt01[..., 1] * vt02[..., 0])

    tangents = f[..., np.newaxis] * (
        v01 * vt02[..., 1][..., np.newaxis] - v02 * vt01[..., 1][..., np.newaxis]
    )
    tangents = tangents / th.norm(tangents, dim=-1, keepdim=True).clamp(min=eps)

    bitangents = th.cross(normals, tangents, dim=-1)
    bitangents = bitangents / th.norm(bitangents, dim=-1, keepdim=True).clamp(
        min=eps
    ).clamp(min=eps)
    return tangents, bitangents, normals


def compute_dX_uv(tri_xyz, tri_uv):
    """ Similar to TBN, but does not normalize vector; also compute dX/dv instead of B = N x T
    Args:
        tri_xyz: [B,N,3,3] vertex coordinates
        tri_uv: [N,2] texture coordinates
    Returns:
        tangents, bitangents, normals
    """

    tri_uv = tri_uv[np.newaxis]

    v01 = tri_xyz[:, :, 1] - tri_xyz[:, :, 0]
    v02 = tri_xyz[:, :, 2] - tri_xyz[:, :, 0]

    vt01 = tri_uv[:, :, 1] - tri_uv[:, :, 0]
    vt02 = tri_uv[:, :, 2] - tri_uv[:, :, 0]

    f = 1.0 / (vt01[..., 0] * vt02[..., 1] - vt01[..., 1] * vt02[..., 0])

    dXu = f[..., np.newaxis] * (
        v01 * vt02[..., 1][..., np.newaxis] - v02 * vt01[..., 1][..., np.newaxis]
    )
    dXv = f[..., np.newaxis] * (
        -v01 * vt02[..., 0][..., np.newaxis] + v02 * vt01[..., 0][..., np.newaxis]
    )

    return dXu, dXv


def compute_dN_uv(tri_normals, tri_uv):
    """
    Args:
        tri_normals: [B,N,3,3] vertex normals
        tri_uv: [N,2] texture coordinates
    Returns:
        tangents, bitangents, normals
    """

    tri_uv = tri_uv[np.newaxis]

    dN01 = tri_normals[:, :, 1] - tri_normals[:, :, 0]
    dN02 = tri_normals[:, :, 2] - tri_normals[:, :, 0]

    vt01 = tri_uv[:, :, 1] - tri_uv[:, :, 0]
    vt02 = tri_uv[:, :, 2] - tri_uv[:, :, 0]

    f = 1.0 / (vt01[..., 0] * vt02[..., 1] - vt01[..., 1] * vt02[..., 0])

    dNu = f[..., np.newaxis] * (
        dN01 * vt02[..., 1][..., np.newaxis] - dN02 * vt01[..., 1][..., np.newaxis]
    )
    dNv = f[..., np.newaxis] * (
        -dN01 * vt02[..., 0][..., np.newaxis] + dN02 * vt01[..., 0][..., np.newaxis]
    )

    return dNu, dNv


def compute_subd_meta(idxs):
    assert idxs.dim() == 2
    sort_edge = lambda e: ((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
    idxs = idxs.data.cpu().numpy()

    i01 = idxs[:, :2]
    i12 = idxs[:, 1:3]
    i20 = np.hstack([idxs[:, 2:3], idxs[:, 0:1]])
    midpoint_idxs = list({sort_edge(idx) for idx in np.vstack([i01, i12, i20])})
    midpoint_map = {edge: i for i, edge in enumerate(midpoint_idxs)}
    i01 = th.LongTensor([midpoint_map[sort_edge(e)] for e in i01])
    i12 = th.LongTensor([midpoint_map[sort_edge(e)] for e in i12])
    i20 = th.LongTensor([midpoint_map[sort_edge(e)] for e in i20])
    return th.LongTensor(midpoint_idxs), (i01, i12, i20)


def subdivide_(p, i, meta=None):
    assert i.dim() == 2
    assert p.dim() in [2, 3]
    expd = False
    if p.dim() == 2:
        expd = True
        p = p[None]

    meta = meta if meta is not None else compute_subd_meta(i)
    mpt_i, (i01, i12, i20) = meta

    p_subd = th.stack([index(p_, mpt_i, 0).sum(1) / 2 for p_ in p])
    p_subd = th.cat([p, p_subd], dim=1)
    i01 += p.shape[1]
    i12 += p.shape[1]
    i20 += p.shape[1]

    new_i = th.cat(
        [
            th.stack([i[:, 0], i01, i20], dim=1),
            th.stack([i01, i[:, 1], i12], dim=1),
            th.stack([i12, i[:, 2], i20], dim=1),
            th.stack([i01, i12, i20], dim=1),
        ]
    )

    if expd:
        p_subd = p_subd[0]
    return p_subd, new_i


def subdivide(v, vt, vi, vti, only_verts=False, metas=None):
    vmeta = metas[0] if metas is not None else compute_subd_meta(vi)
    v_subd, vi_subd = subdivide_(v, vi, vmeta)
    if only_verts:
        return v_subd, vi_subd

    vtmeta = metas[1] if metas is not None else compute_subd_meta(vti)
    vt_subd, vti_subd = subdivide_(vt, vti, vtmeta)
    return v_subd, vt_subd, vi_subd, vti_subd


def subdivide_n(v, vt, vi, vti, n, only_verts=False, metas=None):
    if metas is not None:
        assert len(metas) == n

    for l in range(n):
        meta = metas[l] if metas is not None else None
        out = subdivide(v, vt, vi, vti, only_verts, meta)
        if only_verts:
            v, vi = out
        else:
            v, vt, vi, vti = out

    return v if only_verts else (v, vt, vi, vti)


def compute_subd_meta_n(v, vt, vi, vti, n):
    metas = [(compute_subd_meta(vi), compute_subd_meta(vti))]
    for _ in range(n - 1):
        v, vt, vi, vti = subdivide(v, vt, vi, vti)
        metas.append((compute_subd_meta(vi), compute_subd_meta(vti)))
    return metas


def face_visibility(index_img, n_faces):
    # TODO: can we do some sort of broadcasting here?
    masks = th.zeros(
        [index_img.shape[0], n_faces], dtype=th.int64, device=index_img.device
    )
    for b in range(index_img.shape[0]):
        masks[b][index_img[b][index_img[b] != -1]] = 1
    return masks


def compute_vertex_visibility(index_img, faces, n_verts):
    batch_size = index_img.shape[0]
    verts_mask = th.zeros((batch_size, n_verts), dtype=th.bool, device=index_img.device)
    # getting the visibility mask per vertex
    # TODO: use advanced indexing instead?
    for b in range(batch_size):
        visible_verts = th.unique(faces[index_img[b][index_img[b] != -1]].reshape(-1,))
        verts_mask[b, visible_verts] = 1.0
    return verts_mask


def compute_uv_visiblity(face_index_image, faces, index_image_uv, n_verts):
    batch_size = face_index_image.shape[0]
    uv_size = index_image_uv.shape[0]
    visibility = th.zeros(
        (batch_size, uv_size, uv_size), dtype=th.bool, device=face_index_image.device
    )
    verts_mask = compute_vertex_visibility(face_index_image, faces, n_verts)
    # getting the visibility mask per vertex
    # TODO: use advanced indexing instead?
    for b in range(batch_size):
        mask = th.any(index_image_uv != -1, axis=-1)
        visibility[b][mask] = th.any(verts_mask[b][index_image_uv[mask]], axis=-1)
    return visibility


def compute_view_texture(
    verts,
    faces,
    image,
    face_index_image,
    camera,
    index_image_uv,
    bary_image_uv,
    intensity_threshold=None,
):
    batch_size = verts.shape[0]
    uv_size = index_image_uv.shape[0]
    H, W = image.shape[2:4]

    uv_mask = index_image_uv[..., 0] != -1
    index_flat = index_image_uv[uv_mask]
    bary_flat = bary_image_uv[uv_mask]

    xyz_w = th.sum(
        verts[:, index_flat] * bary_flat[np.newaxis, :, :, np.newaxis], axis=2
    )
    v_pix, v_cam = project_points(xyz_w, **camera)

    yxs = 2.0 * th.stack((v_pix[:, :, 0] / W, v_pix[:, :, 1] / H), axis=-1) - 1.0

    # TODO: add an extra channel here?
    visibility_mask = compute_uv_visiblity(
        face_index_image, faces, index_image_uv, verts.shape[1]
    )

    verts_rgb = F.grid_sample(image, yxs[:, np.newaxis], align_corners=False)[:, :, 0]

    # TODO: should we shuffle things around here?
    tex = th.zeros(
        (batch_size, 3, uv_size, uv_size), dtype=verts_rgb.dtype, device=verts.device
    )
    tex[:, :, uv_mask] = verts_rgb
    tex = tex * visibility_mask[:, np.newaxis]
    # NOTE: we are filtering out pixels that are too white
    if intensity_threshold:
        tex = tex * th.all(tex <= intensity_threshold, axis=1, keepdims=True)

    return tex


def depth_discontuity_mask(depth, threshold=40.0, kscale=4.0, pool_ksize=3):
    device = depth.device

    with th.no_grad():
        # TODO: pass the kernel?
        kernel = th.as_tensor(
            [
                [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
            ],
            dtype=th.float32,
            device=device,
        )

        disc_mask = (
            th.norm(F.conv2d(depth, kernel, bias=None, padding=1), dim=1) > threshold
        )[:, np.newaxis]
        disc_mask = (
            F.avg_pool2d(
                disc_mask.float(), pool_ksize, stride=1, padding=pool_ksize // 2
            )
            > 0.0
        )

    return disc_mask


def gaussian_kernel(ksize, std):
    assert ksize % 2 == 1
    radius = ksize // 2
    xrange = th.linspace(-radius, radius, ksize)
    x, y = th.meshgrid(xrange, xrange)
    xy = th.stack([x, y], axis=2)
    gk = th.exp(-(xy ** 2).sum(-1) / (2 * std ** 2))
    return gk / gk.sum()


class GaussianBlur2d(nn.Module):
    def __init__(self, ksize, std):
        super().__init__()
        self.ksize = ksize
        self.std = std
        kernel = gaussian_kernel(ksize, std)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        w = self.kernel[np.newaxis, np.newaxis].expand(x.shape[1], -1, -1, -1)
        return F.conv2d(x, w, padding=self.ksize // 2, groups=x.shape[1])


def igl_signed_distance(Q, V, F):
    # signed distance of point in Q to the mesh surface represented by V and F
    # gradient is back propogated through Q, not V
    assert len(Q.shape) == 2 and len(V.shape) == 2
    assert isinstance(Q, torch.Tensor)
    assert isinstance(V, torch.Tensor)
    assert isinstance(F, np.ndarray)

    S, I, C, N = igl.signed_distance(Q.data.cpu().numpy(), V.data.cpu().numpy(), F, return_normals=True)
    thN = torch.from_numpy(N).to(Q.device)
    thC = torch.from_numpy(C).to(Q.device)
    signed_distance = torch.einsum('ij,ij->i', Q - thC, thN)

    return signed_distance

