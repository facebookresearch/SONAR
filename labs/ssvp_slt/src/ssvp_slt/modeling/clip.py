# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# References:
#
# OpenCLIP: https://github.com/mlfoundations/open_clip
# --------------------------------------------------------

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sign_hiera_clip import SignHieraForClip
from .text_encoder_clip import HFTextEncoderForClip

"""
CLIP implementation for SignHiera and HuggingFace-based text encoders
Adapted from https://github.com/mlfoundations/open_clip
"""


@dataclass
class CLIPVisionCfg:
    model_id: str
    pretrained_name_or_path: Optional[str] = None
    proj: Optional[str] = None


@dataclass
class CLIPTextCfg:
    hf_model_name: str = None
    hf_model_pretrained: bool = True
    proj: str = "mlp"
    pooler_type: str = "mean_pooler"


def _build_text_tower(
    embed_dim: int,
    text_cfg: CLIPTextCfg,
) -> nn.Module:
    """
    Creates a text tower for CLIP based on the config provided
    """

    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    text = HFTextEncoderForClip(
        text_cfg.hf_model_name,
        output_dim=embed_dim,
        proj=text_cfg.proj,
        pooler_type=text_cfg.pooler_type,
        pretrained=text_cfg.hf_model_pretrained,
    )
    return text


def _build_vision_tower(
    embed_dim: int,
    vision_cfg: CLIPVisionCfg,
) -> nn.Module:
    """
    Creates a vision tower for CLIP based on the config provided
    """

    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    if "hiera" not in vision_cfg.model_id:
        raise NotImplementedError("Models other than `Hiera` are currently not supported")

    visual = SignHieraForClip(
        model_id=vision_cfg.model_id,
        output_dim=embed_dim,
        pretrained_name_or_path=vision_cfg.pretrained_name_or_path,
        proj=vision_cfg.proj,
    )
    return visual


class CLIP(nn.Module):
    """
    CLIP/FLIP model class that creates a vision tower for video encoding and a text tower for text encoding
    """

    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg)
        self.text = _build_text_tower(embed_dim, text_cfg)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def encode_video(
        self,
        video: torch.Tensor,
        padding: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        normalize: bool = False,
        return_mae_loss: bool = True,
    ):
        mae_loss, features = self.visual(video, padding=padding, mask_ratio=mask_ratio, mask=mask)

        if return_mae_loss:
            return F.normalize(features, dim=-1) if normalize else features, mae_loss

        return (F.normalize(features, dim=-1) if normalize else features,)

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(
        self,
        text: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        video_padding: Optional[torch.Tensor] = None,
        video_lengths: Optional[torch.Tensor] = None,
        video_attention_mask: Optional[torch.Tensor] = None,
        video_mask_ratio: Optional[float] = None,
        video_mask: Optional[torch.Tensor] = None,
        return_mae_loss: bool = True,
    ):
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        video_outputs = (
            self.encode_video(
                video,
                padding=video_padding,
                lengths=video_lengths,
                attention_mask=video_attention_mask,
                mask_ratio=video_mask_ratio,
                mask=video_mask,
                return_mae_loss=return_mae_loss,
                normalize=True,
            )
            if video is not None
            else None
        )
        video_features = video_outputs[0]

        if self.output_dict:
            out_dict = {
                "video_features": video_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
            if self.logit_bias is not None:
                out_dict["logit_bias"] = self.logit_bias

            if return_mae_loss:
                out_dict["mae_loss"] = video_outputs[-1]

            return out_dict

        outputs = video_features, text_features, self.logit_scale.exp()
        if self.logit_bias is not None:
            outputs += (self.logit_bias,)
        if return_mae_loss:
            outputs += (video_outputs[-1],)
        return outputs
