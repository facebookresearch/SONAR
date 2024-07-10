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
    dataloader: Iterable,
    clip: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    criterion: torch.nn.Module,
    device: torch.device,
    gradnorm: torch.nn.Module,
    optimizer_gradnorm: torch.optim.Optimizer,
    scaler_gradnorm: torch.cuda.amp.GradScaler,
) -> Dict[str, float]:
    clip.train(True)
    clip_without_ddp = misc.unwrap_model(clip)
    gradnorm.train(True)
    gradnorm_without_ddp = misc.unwrap_model(gradnorm)

    optimizer.zero_grad()
    optimizer_gradnorm.zero_grad()

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

        # We use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(
            cfg, optimizer, data_iter_step / num_batches_in_epoch + epoch
        )
        lr_sched.adjust_learning_rate(
            cfg,
            optimizer_gradnorm,
            data_iter_step / num_batches_in_epoch + epoch,
            suffix="_gradnorm",
        )

        with torch.cuda.amp.autocast(enabled=cfg.common.fp16):
            clip_output_dict = clip(
                text=batch["labels"],
                video=batch["video"],
                video_padding=batch["padding"],
                video_mask_ratio=cfg.model.mask_ratio,
            )
            mae_loss = clip_output_dict["mae_loss"]
            contrastive_loss = criterion(
                image_features=clip_output_dict["video_features"],
                text_features=clip_output_dict["text_features"],
                logit_scale=clip_output_dict["logit_scale"],
            )

            loss = (
                gradnorm_without_ddp.contrastive_weight * contrastive_loss
                + gradnorm_without_ddp.mae_weight * mae_loss
            )

        loss_value = loss.item()
        mae_loss_value = mae_loss.item()
        contrastive_loss_value = contrastive_loss.item()

        if not math.isfinite(loss_value):
            raise Exception(f"Loss is {loss_value}, stopping training")

        optimizer.zero_grad()
        scaler.scale(loss).backward(retain_graph=True)

        gradnorm_loss = gradnorm(
            contrastive_loss, mae_loss, clip_without_ddp.visual.shared_parameters
        )
        gradnorm_loss_value = gradnorm_loss.item()

        if not cfg.model.no_mae:
            optimizer_gradnorm.zero_grad()
            scaler_gradnorm.scale(gradnorm_loss).backward()
            scaler_gradnorm.step(optimizer_gradnorm)
            scaler_gradnorm.update()

        scaler.unscale_(optimizer)

        if cfg.optim.clip_grad:
            torch.nn.utils.clip_grad_norm_(clip.parameters(), cfg.optim.clip_grad)

        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(mae_loss=mae_loss_value)
        metric_logger.update(contrastive_loss=contrastive_loss_value)
        metric_logger.update(gradnorm_loss=gradnorm_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        gradnorm_lr = optimizer_gradnorm.param_groups[0]["lr"]
        metric_logger.update(gradnorm_lr=gradnorm_lr)

        metric_logger.update(time=time.time() - end)
        if (
            data_iter_step % cfg.common.print_steps == 0
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

        if (data_iter_step + 1) % cfg.common.logging_steps == 0:
            # Contrastive loss is already the same across all ranks, so we don't need to all-reduce
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            mae_loss_value_reduce = misc.all_reduce_mean(mae_loss_value)
            gradnorm_loss_value_reduce = misc.all_reduce_mean(gradnorm_loss_value)

            epoch_1000x = int((data_iter_step / num_batches_in_epoch + epoch) * 1000)
            if misc.is_main_process() and cfg.wandb.enabled:
                log_stats = {
                    f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()
                }
                log_stats.update(
                    {
                        "train/loss": loss_value_reduce,
                        "train/mae_loss": mae_loss_value_reduce,
                        "train/contrastive_loss": contrastive_loss_value,
                        "train/gradnorm_loss": gradnorm_loss_value_reduce,
                        "train/loss_weight_contrastive": gradnorm_without_ddp.contrastive_weight.data,
                        "train/loss_weight_mae": gradnorm_without_ddp.mae_weight.data,
                        "train/step": epoch_1000x,
                    }
                )
                misc.wandb_log(log_stats)

        if not cfg.model.no_mae:
            gradnorm_without_ddp.rescale_weights()

        if (data_iter_step + 1) % cfg.common.save_steps == 0:
            misc.save_checkpoint(
                cfg=cfg,
                epoch=epoch,
                checkpoint={
                    "clip": clip_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "gradnorm": gradnorm_without_ddp.state_dict(),
                    "optimizer_gradnorm": optimizer_gradnorm.state_dict(),
                    "scaler_gradnorm": scaler_gradnorm.state_dict(),
                },
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
