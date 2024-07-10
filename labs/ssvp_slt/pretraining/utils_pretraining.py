# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os
from typing import Tuple

import ssvp_slt.modeling.sign_hiera_mae as sign_hiera_mae
import ssvp_slt.util.misc as misc
import torch
from omegaconf import DictConfig, OmegaConf
from ssvp_slt.data.video_dataset import VideoDataset
from ssvp_slt.modeling.sign_hiera_mae import MaskedAutoencoderSignHiera
from torch.utils.data import (ConcatDataset, DataLoader, DistributedSampler,
                              RandomSampler)


def create_model(cfg: DictConfig) -> MaskedAutoencoderSignHiera:
    return sign_hiera_mae.__dict__[cfg.model.name](
        **(OmegaConf.to_container(cfg.model)), pretrained=True, strict=False
    )


def create_dataloader(cfg: DictConfig) -> DataLoader:
    dataset = ConcatDataset(
        [
            VideoDataset(
                mode="pretrain",
                data_dir=os.path.join(os.path.join(cfg.data.base_data_dir, dataset_name)),
                gpu=cfg.dist.gpu if cfg.data.video_backend == "cuda" else None,
                video_backend=cfg.data.video_backend,
                target_fps=cfg.data.target_fps,
                sampling_rate=cfg.data.sampling_rate,
                num_frames=cfg.data.num_frames,
                repeat_aug=cfg.data.repeat_aug,
                rand_aug=cfg.data.rand_aug,
                do_normalize=True,
                min_duration=cfg.data.min_duration,
                max_duration=cfg.data.max_duration,
                max_num_samples=(
                    cfg.data.num_train_samples // len(cfg.data.dataset_names.split(","))
                    if cfg.data.num_train_samples is not None
                    else None
                ),
            )
            for dataset_name in cfg.data.dataset_names.split(",")
        ]
    )

    if cfg.dist.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=True,
        )
        print(f"Sampler = {sampler}")
    else:
        sampler = RandomSampler(dataset)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.optim.batch_size,
        num_workers=cfg.common.num_workers,
        pin_memory=cfg.common.pin_mem,
        persistent_workers=cfg.common.persistent_workers,
        drop_last=True,
    )
    return dataloader


def create_optimizer_and_loss_scaler(
    cfg: DictConfig, model_without_ddp: MaskedAutoencoderSignHiera
) -> Tuple[torch.optim.Optimizer, misc.NativeScalerWithGradNormCount]:
    # Following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(
        model_without_ddp,
        cfg.optim.weight_decay,
        bias_wd=cfg.optim.bias_wd,
    )
    optimizer = torch.optim._multi_tensor.AdamW(
        param_groups,
        lr=cfg.optim.lr,
        betas=(cfg.optim.adam_beta1, cfg.optim.adam_beta2),
    )
    loss_scaler = misc.NativeScalerWithGradNormCount(fp32=not cfg.common.fp16)

    return optimizer, loss_scaler
