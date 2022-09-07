import torch
import torch.nn as nn

import numpy as np

import logging

logger = logging.getLogger(__name__)


class BaseTotalLoss(nn.Module):
    def __init__(self, weights, inverse_rendering, nbs_idxs, nbs_weights):
        super().__init__()
        self.weights = weights
        self.inverse_rendering = inverse_rendering
        self.nbs_idxs = nbs_idxs
        self.nbs_weights = nbs_weights
        # self.register_buffer("nbs_idxs", nbs_idxs)
        # self.register_buffer("nbs_weights", nbs_weights)

    def forward(self, preds, targets, inputs=None):
        return base_total_loss(
            preds,
            targets,
            inverse_rendering=self.inverse_rendering,
            weights=self.weights,
            nbs_idxs=self.nbs_idxs,
            nbs_weights=self.nbs_weights,
        )


def base_total_loss(
    preds, targets, weights, inverse_rendering, nbs_idxs, nbs_weights,
):
    """It is a TOTAL LOSS."""

    sample_weight = targets["sample_weight"]
    # TODO: this should be a separate class?
    losses_dict = dict()

    # geometry
    loss_verts_total, losses_verts = geometry_loss(
        preds=preds,
        targets=targets,
        sample_weight=sample_weight,
        nbs_idxs=nbs_idxs,
        nbs_weights=nbs_weights,
        weight_rec=weights.geometry_rec,
        weight_laplacian=weights.geometry_laplacian,
        weight_seam=weights.geometry_seam,
    )
    loss = loss_verts_total
    losses_dict.update(**losses_verts)

    # texture
    loss_tex_total, losses_tex = texture_loss(
        preds=preds,
        targets=targets,
        sample_weight=sample_weight,
        weight_rec=weights.texture_rec,
        weight_seam=weights.texture_seam,
    )
    loss += loss_tex_total
    losses_dict.update(**losses_tex)

    # keypoint-based loss for skeleton refinement
    if "kpts_3d" in targets:
        loss_kpt = (
            (preds["kpts_3d"] - targets["kpts_3d"][:, :, :3]).abs()
            * targets["kpts_3d"][:, :, 3:4]
        ).mean()
        loss += weights.kpt * loss_kpt
        losses_dict.update(loss_kpt=loss_kpt[np.newaxis])

    if inverse_rendering:
        # LOSS: inverse rendering loss for entire image
        loss_ir_total, losses_ir = inverse_rendering_loss(
            preds=preds,
            targets=targets,
            sample_weight=sample_weight,
            weights_rgb=weights.ir_rgb,
            weights_mask=weights.ir_mask,
        )
        loss += loss_ir_total
        losses_dict.update(**losses_ir)

    return loss, losses_dict


def geometry_loss(
    preds,
    targets,
    nbs_idxs,
    nbs_weights,
    weight_rec,
    weight_laplacian,
    weight_seam=None,
    eps=1.0e-6,
    sample_weight=None,
):
    # vertex-wise 3D position error
    loss_verts_rec = torch.mean((preds["verts"] - targets["verts"]).pow(2).sum(2))

    # ground truth surface Laplacian of ground truth mesh (target_verts)
    loss_verts_laplacian = laplacian_loss(
        preds["verts"], targets["verts"], nbs_idxs, nbs_weights
    )

    loss = weight_rec * loss_verts_rec + weight_laplacian * loss_verts_laplacian

    losses_dict = dict(
        loss_verts_rec=loss_verts_rec, loss_verts_laplacian=loss_verts_laplacian,
    )

    if weight_seam is not None:
        mask_verts_rec_var = (preds["verts_var"] > eps).float()
        loss_verts_seam_var = (preds["verts_var"] * mask_verts_rec_var).sum() / (
            mask_verts_rec_var.sum()
        )
        loss += weight_seam * loss_verts_seam_var
        losses_dict.update(loss_verts_seam_var=loss_verts_seam_var)

    return loss, losses_dict


def texture_loss(preds, targets, sample_weight, weight_rec, weight_seam, eps=1.0e-6):
    sample_weight = sample_weight.view(preds["tex"].shape[0], 1, 1, 1)

    weight_tex = torch.ne(targets["tex"], 0.0).float() * targets["tex_mask"]

    loss_tex_rec = torch.sum(
        weight_tex * (torch.abs(preds["tex"] - targets["tex"]) * sample_weight)
    ) / torch.sum(weight_tex)

    # loss on texture variance of seam vertices
    mask_tex_rec_var = (preds["tex_var"] > eps).float()
    loss_tex_seam_var = (preds["tex_var"] * mask_tex_rec_var).sum() / (
        mask_tex_rec_var.sum()
    )

    loss_tex_total = loss_tex_rec * weight_rec + loss_tex_seam_var * weight_seam

    return (
        loss_tex_total,
        dict(loss_tex_rec=loss_tex_rec, loss_tex_seam_var=loss_tex_seam_var),
    )


def inverse_rendering_loss(
    preds, targets, sample_weight=None, weight_rgb=1.0, weight_mask=1.0
):

    if sample_weight is not None:
        sample_weight = sample_weight.view(preds["rendered_rgb"].shape[0], 1, 1, 1)
    else:
        sample_weight = 1.0

    # LOSS: inverse rendering loss for entire image
    loss_rgb = (
        ((preds["rendered_rgb"] - targets["capture_image"]) * sample_weight).abs()
    ).mean()

    # LOSS: inverse rendering loss for mask
    loss_mask = (
        (
            (
                (preds["rendered_mask"] - targets["capture_image_mask"][:, np.newaxis])
                * sample_weight
            )
        ).abs()
    ).mean()

    loss_total = loss_rgb * weight_rgb + loss_mask * weight_mask

    return loss_total, dict(loss_ir_rgb=loss_rgb, loss_ir_mask=loss_mask)


def index_selection_nd(x, I, dim):
    target_shape = [*x.shape]
    del target_shape[dim]
    target_shape[dim:dim] = [*I.shape]
    return x.index_select(dim, I.view(-1)).reshape(target_shape)


def compute_laplacian(x, nbs_idxs, nbs_weights):
    lapval = index_selection_nd(x, nbs_idxs, 1) * nbs_weights.unsqueeze(2).unsqueeze(0)
    return lapval.sum(2) + x


def laplacian_loss(preds, targets, nbs_idxs, nbs_weights, mask=None):
    l_preds = compute_laplacian(preds, nbs_idxs, nbs_weights)
    l_targets = compute_laplacian(targets, nbs_idxs, nbs_weights)

    delta = (l_preds - l_targets).pow(2)
    if mask is not None:
        delta = delta[:, mask]
    return delta.mean()


def geometry_masked_loss(
    preds,
    targets,
    mask,
    nbs_idxs,
    nbs_weights,
    weight_rec,
    weight_laplacian,
    eps=1.0e-6,
):
    # vertex-wise 3D position error
    loss_verts_rec = torch.mean(
        (preds["verts"][:, mask] - targets["verts"][:, mask]).pow(2).sum(2)
    )

    # ground truth surface Laplacian of ground truth mesh (target_verts)
    loss_verts_laplacian = laplacian_loss(
        preds["verts"], targets["verts"], nbs_idxs, nbs_weights, mask=mask
    )

    loss = weight_rec * loss_verts_rec + weight_laplacian * loss_verts_laplacian

    losses_dict = dict(
        loss_verts_rec=loss_verts_rec, loss_verts_laplacian=loss_verts_laplacian,
    )

    return loss, losses_dict


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1.0 + logvar - mu ** 2 - torch.exp(logvar))


def laplacian_loss_sample(preds, targets, nbs_idxs, nbs_weights, mask=None):
    l_preds = compute_laplacian(preds, nbs_idxs, nbs_weights)
    l_targets = compute_laplacian(targets, nbs_idxs, nbs_weights)

    delta = l_preds - l_targets
    if mask is not None:
        delta = delta[:, mask]
    return delta.pow(2).mean(dim=(1, 2))


class RecLoss(nn.Module):
    """Reconstruction loss."""

    def __init__(self, weights, nbs_idxs, nbs_weights):
        super().__init__()
        self.weights = weights

        logger.info("loading ")
        if isinstance(nbs_idxs, str):
            nbs_idxs = np.loadtxt(nbs_idxs).astype(np.int64)
        if isinstance(nbs_weights, str):
            nbs_weights = np.loadtxt(nbs_weights).astype(np.float32)
        logger.info("done!")

        self.register_buffer("nbs_idxs", torch.tensor(nbs_idxs))
        self.register_buffer("nbs_weights", torch.tensor(nbs_weights))

    def forward(self, preds, targets, inputs=None, iteration=None):
        # TODO: we should start IR only after a certain step?

        loss_verts_rec = (preds["verts"] - targets["verts"]).pow(2).mean()
        loss_verts_laplacian = laplacian_loss(
            preds["verts"], targets["verts"], self.nbs_idxs, self.nbs_weights
        )
        loss_tex_rec = (preds["tex"] - targets["tex"]).abs().mean()

        loss = (
            loss_verts_rec * self.weights.geometry_rec
            + loss_verts_laplacian * self.weights.geometry_laplacian
            + loss_tex_rec * self.weights.tex_rec
        )

        # these will be printed as individual losses
        losses_dict = dict(
            loss_total=loss,
            loss_verts_rec=loss_verts_rec,
            loss_verts_laplacian=loss_verts_laplacian,
            loss_tex_rec=loss_tex_rec,
        )

        return loss, losses_dict
