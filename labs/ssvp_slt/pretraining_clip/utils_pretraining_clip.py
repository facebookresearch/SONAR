# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# OpenCLIP: https://github.com/mlfoundations/open_clip
# --------------------------------------------------------

import os
from typing import Any, Callable, Dict, List, Tuple, Union

import ssvp_slt.util.misc as misc
import torch
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from ssvp_slt.data.video_dataset import VideoDataset
from ssvp_slt.modeling.clip import CLIP, CLIPTextCfg, CLIPVisionCfg
from ssvp_slt.modeling.gradnorm import GradNorm
from torch import distributed as dist
from torch.utils.data import (ConcatDataset, DataLoader, DistributedSampler,
                              RandomSampler)
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class ClipLoss(nn.Module):
    """Adapted from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py"""

    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device: torch.device, num_logits: int) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.nn.Parameter,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.nn.Parameter,
        output_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


def gather_features(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    local_loss: bool = False,
    gather_with_grad: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py"""

    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


def create_criterion() -> ClipLoss:
    return ClipLoss(rank=misc.get_rank(), world_size=misc.get_world_size())


def create_clip_model_and_tokenizer(
    cfg: DictConfig,
) -> Tuple[CLIP, PreTrainedTokenizerFast]:
    vision_cfg = CLIPVisionCfg(
        model_id=cfg.model.vision_model_name,
        pretrained_name_or_path=cfg.common.load_model,
        proj=cfg.model.vision_model_proj,
    )
    text_cfg = CLIPTextCfg(
        hf_model_name=cfg.model.text_model_name_or_path,
        hf_model_pretrained=True,
        proj=cfg.model.text_model_proj,
        pooler_type=cfg.model.text_model_pooler,
    )
    clip = CLIP(
        embed_dim=cfg.model.embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        output_dict=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_model_name_or_path, use_fast=True)

    return clip, tokenizer


def create_gradnorm_model(
    cfg: DictConfig,
    clip: CLIP,
    dataloader: DataLoader,
    criterion: ClipLoss,
    device: torch.device,
) -> GradNorm:
    clip.train()

    # Compute initial losses on the first batch
    batch = next(iter(dataloader))
    for k, v in batch.items():
        batch[k] = v.to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=cfg.common.fp16):
        clip_output_dict = clip(
            text=batch["labels"],
            video=batch["video"],
            video_padding=batch["padding"],
            video_mask_ratio=cfg.model.mask_ratio,
        )
        contrastive_loss = criterion(
            image_features=clip_output_dict["video_features"],
            text_features=clip_output_dict["text_features"],
            logit_scale=clip_output_dict["logit_scale"],
        )
        mae_loss = clip_output_dict["mae_loss"]

    gradnorm = GradNorm(
        alpha=cfg.model.gradnorm_alpha,
        initial_contrastive_loss=contrastive_loss,
        initial_mae_loss=mae_loss,
    )

    if cfg.model.no_mae:
        # Set fixed weights
        gradnorm.mae_weight.data.zero_()
        gradnorm.contrastive_weight.data.fill_(1.0)

        # Freeze GradNorm
        for p in gradnorm.parameters():
            p.requires_grad = False

    return gradnorm


def get_collate_fn(tokenizer: PreTrainedTokenizerFast) -> Callable:
    def collate_fn(samples: List[Dict[str, Any]]):
        frames = torch.stack(
            [sample["frames"].squeeze() for sample in samples]
        )  # squeeze `repeat_aug` dim
        padding = torch.stack([sample["padding"] for sample in samples])
        batch = {"video": frames, "padding": padding}

        encoding = tokenizer(
            [sample["labels"] for sample in samples],
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        batch["input_ids"] = encoding["input_ids"]
        batch["labels"] = encoding["input_ids"]
        batch["attention_mask"] = encoding["attention_mask"]

        return batch

    return collate_fn


def create_dataloader(cfg: DictConfig, tokenizer: PreTrainedTokenizerFast) -> DataLoader:
    dataset = ConcatDataset(
        [
            VideoDataset(
                mode="pretrain",
                data_dir=os.path.join(os.path.join(cfg.data.base_data_dir, dataset_name)),
                target_fps=cfg.data.target_fps,
                return_labels=True,
                gpu=cfg.dist.gpu if cfg.data.video_backend == "cuda" else None,
                video_backend=cfg.data.video_backend,
                sampling_rate=cfg.data.sampling_rate,
                num_frames=cfg.data.num_frames,
                repeat_aug=1,  # repeat_aug is not directly applicable to CLIP training
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

    collate_fn = get_collate_fn(tokenizer)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        batch_size=cfg.optim.batch_size,
        num_workers=cfg.common.num_workers,
        pin_memory=cfg.common.pin_mem,
        persistent_workers=cfg.common.persistent_workers,
        drop_last=True,
    )
    return dataloader


def create_optimizer_and_loss_scaler(
    cfg: DictConfig, model_without_ddp: Union[CLIP, GradNorm]
) -> Tuple[torch.optim.Optimizer, torch.cuda.amp.GradScaler]:
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
    loss_scaler = torch.cuda.amp.GradScaler(enabled=cfg.common.fp16)

    return optimizer, loss_scaler
