# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE-ST: https://github.com/facebookresearch/mae_st
# --------------------------------------------------------

import math

import torch
from omegaconf import DictConfig


def adjust_learning_rate(
    cfg: DictConfig, optimizer: torch.optim.Optimizer, epoch: int, suffix: str = ""
):
    """Decay the learning rate with half-cycle cosine after warmup"""

    if (
        not hasattr(cfg.optim, f"lr{suffix}")
        and not hasattr(cfg.optim, f"min_lr{suffix}")
        and not hasattr(cfg.optim, f"warmup_epochs{suffix}")
    ):
        raise RuntimeError(
            "Cannot adjust learning rate. "
            f"Check that `cfg.optim` contains `lr{suffix}`, `min_lr{suffix}`, and `warmup_epochs{suffix}`."
        )

    peak_lr = getattr(cfg.optim, f"lr{suffix}")
    min_lr = getattr(cfg.optim, f"min_lr{suffix}")
    warmup_epochs = getattr(cfg.optim, f"warmup_epochs{suffix}")

    if epoch < warmup_epochs:
        lr = peak_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (peak_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (cfg.optim.epochs - warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr
