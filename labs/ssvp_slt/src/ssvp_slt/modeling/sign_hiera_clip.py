# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# References:
#
# OpenCLIP: https://github.com/mlfoundations/open_clip
# Hiera: https://github.com/facebookresearch/hiera/
# slowfast: https://github.com/facebookresearch/SlowFast
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from . import sign_hiera_mae

"""
Wraps SignHiera models for use as a vision tower in CLIP model.
"""


def is_shared_hiera_param(name: str) -> bool:
    unshared = [
        "encoder_norm",
        "multi_scale_fusion_heads",
        "decoder",
        "mask_token",
    ]
    for n in unshared:
        if n in name:
            return False
    return True


class SignHieraMeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        masked_output = x * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


class SignHieraForClip(nn.Module):
    def __init__(
        self,
        model_id: str,
        output_dim: int,
        pretrained_name_or_path: Optional[str] = None,
        proj: Optional[str] = None,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.transformer = sign_hiera_mae.__dict__[model_id](pretrained=True, strict=False)
        if pretrained_name_or_path is not None:
            print(f"Loading pretrained model from {pretrained_name_or_path}")
            self.transformer.load_weights(pretrained_name_or_path)

        self.pooler = SignHieraMeanPooler()

        d_model = self.transformer.feature_dim
        if (d_model == output_dim) and (proj is None):
            self.proj = nn.Identity()
        elif proj == "linear":
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj == "mlp":
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=False),
            )

    def forward(
        self,
        x: torch.Tensor,
        padding: torch.Tensor,
        mask_ratio: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_mask = self.transformer.get_attention_mask(padding, device=x.device)
        outputs = self.transformer(
            x, mask_ratio=mask_ratio, mask=mask, attn_mask=attn_mask, return_last_intermediate=True
        )

        # Get MAE loss
        mae_loss = outputs[0]

        # Get pooled encoder features
        x = outputs[-1]
        mask = outputs[-2]
        x = self.pooler(x=x.squeeze(), attention_mask=attn_mask[mask].view(attn_mask.shape[0], -1))
        x = self.proj(x)

        return mae_loss, x

    @property
    def shared_parameters(self) -> List[nn.Parameter]:
        return [
            p
            for n, p in self.transformer.named_parameters()
            if is_shared_hiera_param(n) and p.requires_grad
        ]
