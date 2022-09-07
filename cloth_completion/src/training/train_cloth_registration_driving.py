import os
import sys

import numpy as np
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import FastDataLoader as DataLoader

from omegaconf import OmegaConf, DictConfig

# TODO: this should be specifyable?
from dataset.dataset_clothes import Dataset

from models.aux.io import (
    config_to_instance,
    config_to_model,
    load_pretrained_model,
    filter_params,
    save_model,
    config_to_optimizer,
)

import pyutils
from pyutils import logging
logger = logging.get_logger()

from pyutils.training import (
    validate,
    _process_losses,
    to_device,
    skip_nones_collate_fn,
)


import torchvision

def train(
    model,
    loss_fn,
    loss_fn_disc,
    optimizer,
    optimizer_disc,
    device,
    config,
    train_loader,
    train_writer=None,
    val_loader=None,
    val_writer=None,
    test_loader=None,
    test_writer=None,
    lr_scheduler=None,
    epoch=None,
    iteration=None,
    logging_enabled=True,
):
    """Run an epoch of training training.

    Returns:
        integer, updated `iteration`
    """

    if iteration is None:
        iteration = 0

    model.train()
    for batch in train_loader:
        if batch is None:
            logger.warning("empty batch, skipping")
            continue
        # TODO: move this to collate_fn?
        inputs, targets = to_device(batch, device)

        if loss_fn_disc and optimizer_disc:
            preds = model(**inputs, mode="disc")
            loss_disc, losses_dict_disc = loss_fn_disc(preds, targets, inputs)
            optimizer_disc.zero_grad()
            loss_disc.backward()

            if config.clip_gradient is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_gradient)

            optimizer_disc.step()

        # TODO: we also want to save the actual outputs?
        preds = model(**inputs)
        loss, losses_dict = loss_fn(preds, targets, inputs, iteration=iteration)

        # compute gradients and update
        optimizer.zero_grad()
        # NOTE: why do we need these ones?
        loss.backward()

        if config.clip_gradient is not None:
            norm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_gradient)
            logger.info(f"gradient norm = {norm}")
        # loss.backward()
        optimizer.step()

        if iteration == config.num_max_iters:
            logger.info(f"finishing training after {iteration} iterations")
            break

        if lr_scheduler and iteration and iteration % config.update_lr_every == 0:
            logger.info(f"updating the learning rate after {iteration} iterations")
            lr_scheduler.step()
            # TODO: this could be
            logger.info(f"lr: {lr_scheduler.get_last_lr()[0]}")

        if not logging_enabled:
            continue

        if iteration % config.log_every == 0 or iteration % config.summary_every == 0:
            losses_dict = _process_losses(losses_dict)

        if iteration % config.log_every == 0:
            if config.verbose_logging:
                line = " ".join([f"{k}={v:.4f}" for k, v in losses_dict.items()])
            else:
                line = f"L={losses_dict['total']:.4f}"

            logger.info(f"epoch={epoch}, iter={iteration}: {line}")

        if train_writer and iteration % config.summary_every == 0:
            # NOTE
            # adding losses
            for name, value in losses_dict.items():
                train_writer.add_scalar(f"Losses/{name}", value, global_step=iteration)
            train_writer.flush()
            # adding hyperparameters
            train_writer.add_scalar(
                "Hyperparams/learning_rate",
                lr_scheduler.get_last_lr()[0],
                global_step=iteration,
            )

        if (
            config.image_summary_every is not None
            and train_writer is not None
            and iteration % config.image_summary_every == 0
        ):
            images = torch.cat(
                [preds["rendered_rgb"].detach(), targets["capture_image"].detach()],
                axis=3,
            )
            grid = torchvision.utils.make_grid(images.cpu() / 255.0, nrow=2)
            train_writer.add_image("images", grid, iteration)
            #

        if iteration and iteration % config.save_every == 0:
            # NOTE: new format
            os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
            ckpt_path = os.path.join(
                config.output_dir, "checkpoints", f"{iteration:06d}.pt"
            )
            logger.info(f"saving a training checkpoint to {ckpt_path}")
            ckpt_dict = dict(
                epoch=epoch,
                iteration=iteration,
                optimizer_state_dict=optimizer.state_dict(),
                # NOTE: here we assume it is a DataParallel (!)
                model_state_dict=model.module.state_dict(),
            )
            if lr_scheduler:
                ckpt_dict.update(lr_scheduler_state_dict=lr_scheduler.state_dict())
            if optimizer_disc:
                ckpt_dict.update(optimizer_disc_state_dict=optimizer_disc.state_dict())

            torch.save(ckpt_dict, ckpt_path)
            logger.info(f"done!")
            # disc?

        # TODO: we should change the frame_index thing?
        if val_loader and iteration and iteration % config.val_every == 0:
            logger.info(f"running validation after {iteration} iterations")
            # TODO: should it return things and log them?
            model.eval()
            validate(model, loss_fn, val_loader, config.output_dir, val_writer, iteration, device)
            model.train()

        if test_loader and iteration and iteration % config.test_every == 0:
            logger.info(f"running testing after {iteration} iterations")
            # TODO: should it return things and log them?
            model.eval()
            validate(
                model, loss_fn, test_loader, config.output_dir, test_writer, iteration, device, tag="test"
            )
            model.train()
    
        iteration += 1
    return iteration


def main(config):
    """Main training function."""
    pyutils.logging.setup()

    # getting the rank
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    logger.info(f"current device: {device}")

    if 'resume_iteration' not in config and local_rank == 0:
        logger.info(f"saving the model config to {config.training.output_dir}")
        os.makedirs(config.training.output_dir, exist_ok=config.overwrite)
        OmegaConf.save(config, os.path.join(config.training.output_dir, "config.yml"))

    DatasetClass = Dataset

    train_data = DatasetClass(**config.split.train, **config.dataset)

    if 'frame_list_weights' in config.split.train:
        logger.info('Sampling Weights provided, using weigted random sampler (not distributed)')
        frame_list_weights = np.load(config.split.train.frame_list_weights)
        assert frame_list_weights.shape[0] == len(train_data)
        train_sampler = torch.utils.data.WeightedRandomSampler(frame_list_weights, len(train_data), replacement=True)
    else:
        logger.info('Using default distributed sampler')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    train_loader = DataLoader(
        train_data,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=False,
        sampler=train_sampler,
        collate_fn=skip_nones_collate_fn,
    )

    train_writer = None
    if local_rank == 0:
        train_writer = SummaryWriter(
            log_dir=os.path.join(config.training.output_dir, "train")
        )

    # validation data
    if "val" in config.split and local_rank == 0:
        val_data = DatasetClass(
            **config.split.val, **config.dataset
        )
        val_loader = DataLoader(
            val_data,
            batch_size=config.training.batch_size,
            shuffle=False,
            # TODO: have a separate setting for this?
            num_workers=config.training.num_workers,
            collate_fn=skip_nones_collate_fn,
        )
        val_writer = SummaryWriter(
            log_dir=os.path.join(config.training.output_dir, "val"),
        )
    else:
        val_loader, val_writer = None, None

    # NOTE: testing is supposed to be more rare, and run on the entire dataset
    test_loader, test_writer = None, None
    if "test" in config.split and local_rank == 0:
        test_data = DatasetClass(
            **config.split.test, **config.dataset
        )
        test_loader = DataLoader(
            test_data,
            batch_size=config.training.batch_size,
            shuffle=False,
            # TODO: have a separate setting for this?
            num_workers=config.training.num_workers,
            collate_fn=skip_nones_collate_fn,
        )
        test_writer = SummaryWriter(
            log_dir=os.path.join(config.training.output_dir, "test"),
        )

    logger.info("building the model")
    model = config_to_model(config.model, dataset=train_data).to(device)
    logger.info("done!")

    if config.training.device_ids is not None:
        logger.warning("in DDP mode, ignoring `training.device_ids` settings")

    # TODO: make a proper restore flag that would look up

    # TODO: should this be just a load_model function?
    if 'resume_iteration' in config:
        assert type(config.resume_iteration) == int
        # start from a existing checkpoint of this model
        ckpt_path = os.path.join(
            config.training.output_dir, "checkpoints", f"{(config.resume_iteration):06d}.pt"
        )
        ckpt_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        params = filter_params(
            ckpt_dict['model_state_dict'],
            [
                "decoder_view.renderer_rgb.*",
                "decoder_view.renderer_mask.*",
                "decoder_view.body_soft_render.*",
                "decoder_view.clothes_soft_render.*",
                "decoder_view.renderer_hard.*",
            ]
        )
        model.load_state_dict(params)
    else:
        if isinstance(config.training.pretrained_model, DictConfig):
            logger.info(
                f"loading pretrained model: {config.training.pretrained_model.pretty()}"
            )
            load_pretrained_model(
                model,
                **config.training.pretrained_model,
                ignore_names=[
                    "decoder_view.renderer_rgb.*",
                    "decoder_view.renderer_mask.*",
                    "decoder_view.body_soft_render.*",
                    "decoder_view.clothes_soft_render.*",
                    "decoder_view.renderer_hard.*",
                ],  # ignore the DRTK buffers
                device=device
            )
            logger.info("done!")

    # TODO: set this configurable?
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=True
    )

    optimizer_disc = None
    loss_fn_disc = None
    if config.training.optimizer_disc is not None:
        optimizer_disc = config_to_instance(
            config.training.optimizer_disc,
            params=model.module.discriminator.parameters(),
        )
        loss_fn_disc = config_to_instance(config.loss_discriminator)

    # TODO: these should probably be read directly in the loss (!)
    # OR: the loss should actually be a part of bigger model?
    loss_fn = config_to_instance(config.loss, dataset=train_data).to(device)

    optimizer = config_to_optimizer(config.training.optimizer, model)

    lr_scheduler = config_to_instance(config.training.lr_scheduler, optimizer=optimizer)

    # testing data

    # TODO: epochs?
    iteration = 0
    init_epoch = 0

    if 'resume_iteration' in config:  # can be specified by the command line input
        iteration = config.resume_iteration + 1
        optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
        lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler_state_dict'])
        init_epoch = ckpt_dict['epoch']

    # NOTE: we are syncing here in order to avoid thrashing I/O
    logger.info(f"{local_rank} ready to train, waiting for others")
    torch.distributed.barrier()

    for epoch in range(init_epoch, config.training.num_epochs):
        logger.info(f"starting epoch {epoch}")
        iteration = train(
            model=model,
            loss_fn=loss_fn,
            loss_fn_disc=loss_fn_disc,
            optimizer=optimizer,
            optimizer_disc=optimizer_disc,
            device=device,
            config=config.training,
            train_loader=train_loader,
            train_writer=train_writer,
            val_loader=val_loader,
            val_writer=val_writer,
            test_loader=test_loader,
            test_writer=test_writer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            iteration=iteration,
            logging_enabled=local_rank == 0,
        )
        if iteration >= config.training.num_max_iters:
            logger.info("reached maximum number of iterations, finishing")
            break

    # saving the final
    if local_rank == 0:
        logger.info(f"saving the final model to {config.training.output_dir}")
        save_model(model, config.training.output_dir)
        logger.info("done!")

        train_writer.close()
        val_writer.close()


def setup(group_name=None):
    logger.info("initializing process group")
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", group_name=group_name
    )


def cleanup():
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # load configuration
    if len(sys.argv) < 2:
        logger.error("usage: python <config.yml> [group.value=<value> ...]")
        sys.exit(-1)

    config = OmegaConf.load(str(sys.argv[1]))
    config_cli = OmegaConf.from_cli(args_list=sys.argv[2:])

    if config_cli:
        logger.info("overriding with following values from args:")
        logger.info(f"{config_cli.pretty()}")
        config = OmegaConf.merge(config, config_cli)

    if config.random_seed:
        pyutils.io.seed_torch(config.random_seed)

    setup(config.group_name)
    main(config)
    cleanup()
