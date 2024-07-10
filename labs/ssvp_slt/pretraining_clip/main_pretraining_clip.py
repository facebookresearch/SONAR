# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import datetime
import time

import ssvp_slt.util.misc as misc
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf

from engine_pretraining_clip import train_one_epoch
from utils_pretraining_clip import (create_clip_model_and_tokenizer,
                                    create_criterion, create_dataloader,
                                    create_gradnorm_model,
                                    create_optimizer_and_loss_scaler)


def main(cfg: DictConfig):
    misc.init_distributed_mode(cfg)

    misc.seed_all(cfg.common.seed + misc.get_rank())
    cudnn.benchmark = True
    device = torch.device(cfg.common.device)

    print(OmegaConf.to_yaml(cfg))

    clip, tokenizer = create_clip_model_and_tokenizer(cfg)
    criterion = create_criterion()

    dataloader_train = create_dataloader(cfg, tokenizer)

    if cfg.wandb.enabled:
        misc.setup_wandb(cfg, clip)

    print(f"Model = {str(clip)}")
    print(
        f"Number of params (M): {(sum(p.numel() for p in clip.parameters() if p.requires_grad) / 1.0e6):.2f}"
    )
    print(f"Learning rate: {cfg.optim.lr:.2e}")
    print(f"Effective batch size: {cfg.optim.batch_size * misc.get_world_size()}")

    clip.to(device)
    if cfg.dist.enabled:
        clip = torch.nn.parallel.DistributedDataParallel(
            clip, device_ids=[torch.cuda.current_device()], find_unused_parameters=True
        )
        clip_without_ddp = clip.module
    else:
        clip_without_ddp = clip

    gradnorm_model = create_gradnorm_model(cfg, clip, dataloader_train, criterion, device)
    gradnorm_model.to(device)

    if cfg.dist.enabled and not cfg.model.no_mae:
        gradnorm_model = torch.nn.parallel.DistributedDataParallel(
            gradnorm_model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True
        )
        gradnorm_model_without_ddp = gradnorm_model.module
    else:
        gradnorm_model_without_ddp = gradnorm_model

    optimizer, loss_scaler = create_optimizer_and_loss_scaler(cfg, clip_without_ddp)
    optimizer_gradnorm, loss_scaler_gradnorm = create_optimizer_and_loss_scaler(
        cfg, gradnorm_model_without_ddp
    )

    misc.load_checkpoint(
        cfg=cfg,
        container={
            "clip": clip_without_ddp,
            "optimizer": optimizer,
            "scaler": loss_scaler,
            "gradnorm": gradnorm_model_without_ddp,
            "optimizer_gradnorm": optimizer_gradnorm,
            "scaler_gradnorm": loss_scaler_gradnorm,
        },
    )

    checkpoint_path = ""
    print(f"Start training for {cfg.optim.epochs} epochs")
    start_time = time.time()

    for epoch in range(cfg.optim.start_epoch, cfg.optim.epochs):
        if cfg.dist.enabled:
            dataloader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            cfg=cfg,
            epoch=epoch,
            clip=clip,
            dataloader=dataloader_train,
            optimizer=optimizer,
            scaler=loss_scaler,
            gradnorm=gradnorm_model,
            optimizer_gradnorm=optimizer_gradnorm,
            scaler_gradnorm=loss_scaler_gradnorm,
            criterion=criterion,
            device=device,
        )

        if cfg.common.output_dir:
            checkpoint_path = misc.save_checkpoint(
                cfg=cfg,
                epoch=epoch,
                checkpoint={
                    "clip": clip_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": loss_scaler.state_dict(),
                    "gradnorm": gradnorm_model_without_ddp.state_dict(),
                    "optimizer_gradnorm": optimizer_gradnorm.state_dict(),
                    "scaler_gradnorm": loss_scaler_gradnorm.state_dict(),
                },
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "train_weight_contrastive": gradnorm_model_without_ddp.contrastive_weight.data,
            "train_weight_mae": gradnorm_model_without_ddp.mae_weight.data,
        }

        if misc.is_main_process() and cfg.wandb.enabled:
            misc.wandb_log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]
