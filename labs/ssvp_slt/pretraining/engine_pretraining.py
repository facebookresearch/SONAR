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
import math
import time
from typing import Dict, Iterable

import ssvp_slt.util.lr_sched as lr_sched
import ssvp_slt.util.misc as misc
import torch
from omegaconf import DictConfig
from ssvp_slt.util.batch_sampling import skip_first_batches


def train_one_epoch(
    cfg: DictConfig,
    epoch: int,
    model: torch.nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_scaler: misc.NativeScalerWithGradNormCount,
    device: torch.device,
) -> Dict[str, float]:
    model.train(True)
    optimizer.zero_grad()

    num_batches_in_epoch = len(dataloader)
    data_iter_step = 0

    if cfg.optim.epoch_offset is not None:
        print(f"Skipping the first {cfg.optim.epoch_offset} batches in epoch {epoch}")
        dataloader = skip_first_batches(dataloader, cfg.optim.epoch_offset)
        data_iter_step += cfg.optim.epoch_offset

    prefetcher = misc.Prefetcher(dataloader, device)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    space_fmt = ":" + str(len(str(num_batches_in_epoch))) + "d"
    log_msg = [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}"]
    log_msg = metric_logger.delimiter.join(log_msg)

    start_time = time.time()
    end = time.time()

    batch = next(prefetcher)
    while batch is not None:
        metric_logger.update(data=time.time() - end)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % cfg.optim.gradient_accumulation_steps == 0:
            lr_sched.adjust_learning_rate(
                cfg, optimizer, data_iter_step / num_batches_in_epoch + epoch
            )

        frames = batch["frames"].float()
        padding = batch["padding"]
        if len(frames.shape) == 6:
            b, r, c, t, h, w = frames.shape
            frames = frames.reshape(b * r, c, t, h, w)

        if len(padding.shape) == 2:
            b, r = padding.shape
            padding = padding.reshape(b * r)

        with torch.cuda.amp.autocast(enabled=cfg.common.fp16):
            outputs = model(frames, mask_ratio=cfg.model.mask_ratio, padding=padding)
            loss = outputs[0]

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            raise Exception(f"Loss is {loss_value}, stopping training")

        loss /= cfg.optim.gradient_accumulation_steps

        grad_norm = loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % cfg.optim.gradient_accumulation_steps == 0,
            clip_grad=cfg.optim.clip_grad,
        )

        if (data_iter_step + 1) % cfg.optim.gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(grad_norm=grad_norm)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        metric_logger.update(time=time.time() - end)
        if (
            data_iter_step % (cfg.optim.gradient_accumulation_steps * cfg.common.print_steps) == 0
            or data_iter_step == num_batches_in_epoch - 1
        ):
            eta_seconds = metric_logger.meters["time"].global_avg * (
                num_batches_in_epoch - data_iter_step
            )
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                log_msg.format(
                    data_iter_step,
                    num_batches_in_epoch,
                    eta=eta_string,
                    meters=str(metric_logger),
                )
            )

        if (data_iter_step + 1) % (
            cfg.optim.gradient_accumulation_steps * cfg.common.logging_steps
        ) == 0:
            """We use epoch_1000x as the x-axis, which calibrates different curves when batch size changes."""
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            epoch_1000x = int(
                (data_iter_step / num_batches_in_epoch + epoch) * 1000 * cfg.data.repeat_aug
            )
            if misc.is_main_process() and cfg.wandb.enabled:
                log_stats = {
                    f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()
                }
                log_stats.update({"train/loss": loss_value_reduce, "train/step": epoch_1000x})
                misc.wandb_log(log_stats)

        if (data_iter_step + 1) % (
            cfg.optim.gradient_accumulation_steps * cfg.common.save_steps
        ) == 0:
            misc.save_checkpoint(
                cfg=cfg,
                epoch=epoch,
                checkpoint={
                    "model": misc.unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": loss_scaler.state_dict(),
                },
                max_checkpoints=cfg.common.max_checkpoints,
                epoch_offset=data_iter_step,
            )

        data_iter_step += 1
        end = time.time()
        batch = next(prefetcher)

    # Reset epoch_offset after the first completed epoch
    cfg.optim.epoch_offset = None

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"{header} Total time: {total_time_str} ({total_time / len(dataloader):.4f} s / it)")

    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
