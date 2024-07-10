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

from engine_pretraining import train_one_epoch
from utils_pretraining import (create_dataloader, create_model,
                               create_optimizer_and_loss_scaler)


def main(cfg: DictConfig):
    if cfg.data.video_backend == "cuda":
        torch.multiprocessing.set_start_method("spawn")

    misc.init_distributed_mode(cfg)

    misc.seed_all(cfg.common.seed + misc.get_rank())
    cudnn.benchmark = True
    device = torch.device(cfg.common.device)

    print(OmegaConf.to_yaml(cfg))

    model = create_model(cfg)

    if misc.get_last_checkpoint(cfg) is None and cfg.common.load_model:
        misc.load_model(model, cfg.common.load_model)

    dataloader_train = create_dataloader(cfg)

    if cfg.wandb.enabled:
        misc.setup_wandb(cfg, model)

    print(f"Model = {str(model)}")
    print(
        f"Number of params (M): {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1.0e6):.2f}"
    )
    print(f"Learning rate: {cfg.optim.lr:.2e}")
    print(f"Accumulate grad iterations: {cfg.optim.gradient_accumulation_steps}")
    print(
        f"Effective batch size: {cfg.optim.batch_size * cfg.optim.gradient_accumulation_steps * misc.get_world_size()}"
    )

    model.to(device)
    if cfg.dist.enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    optimizer, loss_scaler = create_optimizer_and_loss_scaler(cfg, model_without_ddp)

    misc.load_checkpoint(
        cfg=cfg,
        container={
            "model": model_without_ddp,
            "optimizer": optimizer,
            "scaler": loss_scaler,
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
            model=model,
            dataloader=dataloader_train,
            optimizer=optimizer,
            device=device,
            loss_scaler=loss_scaler,
        )
        if cfg.common.output_dir:
            checkpoint_path = misc.save_checkpoint(
                cfg=cfg,
                epoch=epoch,
                checkpoint={
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": loss_scaler.state_dict(),
                },
                max_checkpoints=cfg.common.max_checkpoints,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if misc.is_main_process() and cfg.wandb.enabled:
            misc.wandb_log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]
