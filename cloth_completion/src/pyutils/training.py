import os
import torch
import cv2
from models.aux.io import save_model
from src.pyutils import io
import logging

from copy import deepcopy

logger = logging.getLogger(__name__)


class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def skip_nones_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def skip_nones_allow_incomplete_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return th.utils.data.dataloader.default_collate(batch)


def skip_nones_multi_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) <= 1:
        return None
    return th.utils.data.dataloader.default_collate(batch)


def _process_batch(batch, device):
    def _process(v):
        return v.to(device) if isinstance(v, torch.Tensor) else v

    return {k: _process(v) for k, v in batch.items()}


def to_device(values, device=None, non_blocking=True):
    """Transfer a set of values to the device.

    Args:
        values: a nested dict/list/tuple of tensors
        device: argument to `to()` for the underlying vector

    NOTE:
        if the device is not specified, using `torch.cuda()`
    """
    if device is None:
        device = torch.device("cuda")

    if isinstance(values, dict):
        return {k: to_device(v) for k, v in values.items()}
    elif isinstance(values, tuple):
        return (to_device(v) for v in values)
    elif isinstance(values, list):
        return [to_device(v) for v in values]
    elif isinstance(values, torch.Tensor):
        return values.to(device, non_blocking=non_blocking)
    else:
        return values


def _process_losses(loss_dict, reduce=True, detach=True):
    """Preprocess the outputs.

    NOTE: in practice, this just computes losses
    """
    result = {
        k.replace("loss_", ""): v for k, v in loss_dict.items() if k.startswith("loss_")
    }
    if detach:
        result = {k: v.detach() for k, v in result.items()}
    if reduce:
        result = {k: float(v.mean().item()) for k, v in result.items()}
    return result


def train(
    model,
    loss_fn,
    optimizer,
    device,
    config,
    train_loader,
    train_writer,
    val_loader=None,
    val_writer=None,
    test_loader=None,
    test_writer=None,
    lr_scheduler=None,
    epoch=None,
    iteration=None,
):
    """Run an epoch of training training.

    Returns:
        integer, updated `iteration`
    """

    if iteration is None:
        iteration = 0

    model.train()
    for (inputs, targets) in train_loader:
        # TODO: move this to collate_fn?
        inputs = _process_batch(inputs, device)
        targets = _process_batch(targets, device)

        # TODO: we also want to save the actual outputs?
        preds = model(**inputs)

        loss, losses_dict = loss_fn(preds, targets, inputs)

        losses_dict = _process_losses(losses_dict)

        # compute gradients and update
        optimizer.zero_grad()
        # NOTE: why do we need these ones?
        loss.backward()
        # loss.backward()
        optimizer.step()

        if iteration == config.training.num_max_iters:
            logger.info(f"finishing training after {iteration} iterations")
            break

        if iteration % config.training.log_every == 0:
            if config.training.verbose_logging:
                line = " ".join([f"{k}={v:.4f}" for k, v in losses_dict.items()])
            else:
                line = f"L={losses_dict['total']:.4f}"

            logger.info(f"epoch={epoch}, iter={iteration}: {line}")

        if iteration % config.training.summary_every == 0:
            # adding losses
            for name, value in losses_dict.items():
                train_writer.add_scalar(f"Losses/{name}", value, global_step=iteration)
            train_writer.flush()
            # adding hyperparameters
            train_writer.add_scalar(
                "Hyperparams/learning_rate",
                lr_scheduler.get_lr()[0],
                global_step=iteration,
            )
            #

        if (
            lr_scheduler
            and iteration
            and iteration % config.training.update_lr_every == 0
        ):
            logger.info(f"updating the learning rate after {iteration} iterations")
            lr_scheduler.step()
            # TODO: this could be
            logger.info(f"lr: {lr_scheduler.get_lr()[0]}")

        # TODO: we should change the frame_index thing?
        if val_loader and iteration and iteration % config.training.val_every == 0:
            logger.info(f"running validation after {iteration} iterations")
            # TODO: should it return things and log them?
            model.eval()
            validate(model, loss_fn, val_loader, val_writer, iteration, device)
            model.train()

        if test_loader and iteration and iteration % config.training.test_every == 0:
            logger.info(f"running testing after {iteration} iterations")
            # TODO: should it return things and log them?
            model.eval()
            validate(
                model, loss_fn, test_loader, test_writer, iteration, device, tag="test"
            )
            model.train()

        if iteration and iteration % config.training.save_every == 0:
            out_models_dir = os.path.join(
                config.training.output_dir, "models", f"{iteration:06d}"
            )
            os.makedirs(out_models_dir, exist_ok=True)
            logger.info(f"saving model to {out_models_dir}")
            # saving for this particular iteration
            save_model(model, out_models_dir)
            logger.info(f"done!")

        iteration += 1
    return iteration


def validate(
    model,
    loss_fn,
    loader,
    output_dir,
    writer=None,
    iteration=None,
    device=None,
    tag="val",
    verbose=False,
):
    """Runs valiadtion."""
    with torch.no_grad():
        losses_list = []
        for i, batch in enumerate(loader):
            if batch is None:
                logger.warning("empty batch, skipping")
                continue

            inputs, targets = batch
            inputs = _process_batch(inputs, device)
            targets = _process_batch(targets, device)
            # predict
            preds = model(**inputs)
            # save visuals
            save_visuals(preds, batch, loader.dataset.clothes_faces, output_dir, iteration)
            # compute loss
            _, losses_dict = loss_fn(preds, targets, inputs)
            # NOTE: here we do not actually need to average things twice?
            losses_list.append(deepcopy(_process_losses(losses_dict, reduce=False)))
        # aggregate across batches
        names = losses_list[0].keys()
        losses_dict = {
            name: torch.stack([l[name] for l in losses_list]).mean() for name in names
        }

        if writer:
            for name, value in losses_dict.items():
                writer.add_scalar(f"Losses/{name}", value, global_step=iteration)
            writer.flush()

        logger.info(f"{tag} L={losses_dict['total']:.6f}")

        if verbose:
            line = " ".join([f"{k}={v:.4f}" for k, v in losses_dict.items()])
            logger.info(f"{tag}: {line}")

    return losses_dict

def save_visuals(preds, batch, clothes_faces, output_dir, iteration):
    mesh_dir = os.path.join(output_dir, 'cloth_recon_meshes', f'iteration-{iteration}')
    os.makedirs(mesh_dir, exist_ok=True)

    frames = batch[0]["frame"]
    # cameras = batch[0]["camera_id"]
    for i in range(len(frames)):
        mesh_filename = os.path.join(mesh_dir, f'frame_{frames[i]}.ply')
        io.save_ply(mesh_filename, preds['clothes_verts'][i].detach().cpu(), torch.Tensor(clothes_faces).to(dtype=torch.int))

def save_output_summaries(inputs, preds, targets, writer):
    """Saving outputs (and inputs) as summaries.

    # TODO: this has to be an overridable function?

    Args:
        inputs: a dict of pytorch tensors, model inputs
        outputs: a dict of pytorch tensors, model outputs
        writer: an instance of SummaryWriter

    """
    # TODO: add images?
    pass
