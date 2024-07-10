# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# References:
#
# Hiera: https://github.com/facebookresearch/hiera/
# slowfast: https://github.com/facebookresearch/SlowFast
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

"""
MaskedAutoencoderSignHiera PyTorch model
Adapted from https://github.com/facebookresearch/hiera/blob/v0.1.2/hiera/hiera_mae.py

Changes made:
- made clip size variable (to increase from 16 to 128)
- added temporal attention masking at mask unit (MU) level
"""

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .sign_hiera import SignHiera, SignHieraBlock
from .sign_hiera_utils import conv_nd, pretrained_model, undo_windowing


def apply_fusion_head(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(head, nn.Identity):
        return x

    B, num_mask_units = x.shape[0:2]
    # Apply head, e.g [B, #MUs, My, Mx, C] -> head([B * #MUs, C, My, Mx])
    permute = [0] + [len(x.shape) - 2] + list(range(1, len(x.shape) - 2))
    x = head(x.reshape(B * num_mask_units, *x.shape[2:]).permute(permute))

    # Restore original layout, e.g. [B * #MUs, C', My', Mx'] -> [B, #MUs, My', Mx', C']
    permute = [0] + list(range(2, len(x.shape))) + [1]
    x = x.permute(permute).reshape(B, num_mask_units, *x.shape[2:], x.shape[1])
    return x


class MaskedAutoencoderSignHiera(SignHiera):
    """Masked Autoencoder with SignHiera backbone"""

    def __init__(
        self,
        in_chans: int = 3,
        patch_stride: Tuple[int, ...] = (4, 4),
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 8,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        **kwdargs,
    ):
        super().__init__(
            in_chans=in_chans,
            patch_stride=patch_stride,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            **kwdargs,
        )

        del self.norm, self.head
        encoder_dim_out = self.blocks[-1].dim_out
        self.encoder_norm = norm_layer(encoder_dim_out)
        self.mask_unit_spatial_shape_final = [
            i // s ** (self.q_pool) for i, s in zip(self.mask_unit_size, self.q_stride)
        ]
        self.tokens_spatial_shape_final = [
            i // s ** (self.q_pool) for i, s in zip(self.tokens_spatial_shape, self.q_stride)
        ]

        # --------------------------------------------------------------------------
        # Multi-scale fusion heads
        curr_mu_size = self.mask_unit_size
        self.multi_scale_fusion_heads = nn.ModuleList()

        for i in self.stage_ends[: self.q_pool]:  # resolution constant after q_pool
            kernel = [i // s for i, s in zip(curr_mu_size, self.mask_unit_spatial_shape_final)]
            curr_mu_size = [i // s for i, s in zip(curr_mu_size, self.q_stride)]
            self.multi_scale_fusion_heads.append(
                conv_nd(len(self.q_stride))(
                    self.blocks[i].dim_out,
                    encoder_dim_out,
                    kernel_size=kernel,
                    stride=kernel,
                )
            )
        self.multi_scale_fusion_heads.append(nn.Identity())  # final stage, no transform

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_dim_out, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, math.prod(self.tokens_spatial_shape_final), decoder_embed_dim)
        )

        self.decoder_blocks = nn.ModuleList(
            [
                SignHieraBlock(
                    dim=decoder_embed_dim,
                    dim_out=decoder_embed_dim,
                    heads=decoder_num_heads,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_stride = patch_stride[-1] * (
            self.q_stride[-1] ** self.q_pool
        )  # patch stride of prediction
        self.patch_stride = patch_stride

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            (self.pred_stride ** min(2, len(self.q_stride))) * in_chans,
        )  # predictor
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._mae_init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _mae_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pixel_label_2d(
        self, input_img: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*
        input_img = input_img.permute(0, 2, 3, 1)

        size = self.pred_stride
        label = input_img.unfold(1, size, size).unfold(2, size, size)
        label = label.flatten(1, 2).flatten(2)
        label = label[mask]
        if norm:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def get_pixel_label_3d(
        self, input_vid: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*

        # We use time strided loss, only take the first frame from each token
        input_vid = input_vid[:, :, :: self.patch_stride[0], :, :]

        size = self.pred_stride
        label = input_vid.unfold(3, size, size).unfold(4, size, size)
        label = label.permute(
            0, 2, 3, 4, 5, 6, 1
        )  # Different from 2d, mistake during training lol
        label = label.flatten(1, 3).flatten(2)
        label = label[mask]

        if norm:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def super_forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        if mask is None:
            mask = self.get_random_mask(x, mask_ratio, attn_mask=attn_mask)  # [B, #MUs_all]

        # Get multi-scale representations from encoder
        _, intermediates = super().forward(x, mask, attn_mask=attn_mask, return_intermediates=True)

        return intermediates[-1], mask

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_last_intermediate: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = self.get_random_mask(x, mask_ratio, attn_mask=attn_mask)  # [B, #MUs_all]

        # Get multi-scale representations from encoder
        _, intermediates = super().forward(x, mask, attn_mask=attn_mask, return_intermediates=True)
        if return_last_intermediate:
            last_intermediate = intermediates[-1]

        # Resolution unchanged after q_pool x, so skip those features
        intermediates = intermediates[: self.q_pool] + intermediates[-1:]

        # Multi-scale fusion
        x = 0.0
        for head, interm_x in zip(self.multi_scale_fusion_heads, intermediates):
            x += apply_fusion_head(head, interm_x)

        x = self.encoder_norm(x)

        if return_last_intermediate:
            return x, mask, last_intermediate
        return x, mask

    def forward_decoder(
        self, x: torch.Tensor, mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Embed tokens
        x = self.decoder_embed(x)

        # Create attention mask if not provided
        if attn_mask is None:
            attn_mask = torch.ones(
                (x.shape[0], math.prod(self.mask_spatial_shape)), device=x.device
            )

        # Combine visible and mask tokens

        # x: [B, #MUs, *mask_unit_spatial_shape_final, encoder_dim_out]
        # mask: [B, #MUs_all]

        x_dec = torch.zeros(*mask.shape, *x.shape[2:], device=x.device, dtype=x.dtype)

        mask_tokens = self.mask_token.view((1,) * (len(mask.shape) + len(x.shape[2:-1])) + (-1,))
        mask = mask.reshape(mask.shape + (1,) * len(x.shape[2:]))
        attn_mask = attn_mask.reshape(attn_mask.shape + (1,) * len(x.shape[2:-1]))

        mask = mask.expand((-1,) * 2 + x.shape[2:]).bool()
        attn_mask = attn_mask.expand((-1,) * 2 + x.shape[2:-1]).bool()

        x_dec[mask] = x.flatten()
        x_dec = ~mask * mask_tokens + mask * x_dec

        # Get back spatial order
        x = undo_windowing(
            x_dec,
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )
        mask = undo_windowing(
            mask[..., 0:1],
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )
        attn_mask = undo_windowing(
            attn_mask[..., None],
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )

        # Flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        mask = mask.view(x.shape[0], -1)
        attn_mask = attn_mask.view(x.shape[0], -1)

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x, attn_mask=attn_mask)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        return x, mask, attn_mask

    def forward_loss(
        self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note: in mask, 0 is *visible*, 1 is *masked*

        x: e.g. [B, 3, H, W]
        pred: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        label: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        """
        if len(self.q_stride) == 2:
            label = self.get_pixel_label_2d(x, mask)
        elif len(self.q_stride) == 3:
            label = self.get_pixel_label_3d(x, mask)
        else:
            raise NotImplementedError

        pred = pred[mask]
        loss = (pred - label) ** 2

        return loss.mean(), pred, label

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.9,
        mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        padding: Optional[torch.Tensor] = None,
        return_last_intermediate: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        if padding is not None:
            attn_mask = self.get_attention_mask(padding, device=x.device)

        encoder_outputs = self.forward_encoder(
            x,
            mask_ratio,
            mask=mask,
            attn_mask=attn_mask,
            return_last_intermediate=return_last_intermediate,
        )
        latent = encoder_outputs[0]
        mask = encoder_outputs[1]

        pred, pred_mask, attn_mask = self.forward_decoder(
            latent, mask, attn_mask=attn_mask
        )  # pred_mask is mask at resolution of *prediction*

        # Only compute loss for tokens that are masked *and* are attended to (i.e., no padding)
        combined_mask = torch.logical_and(~pred_mask, attn_mask.bool())

        # Toggle mask, to generate labels for *masked* tokens
        outputs = self.forward_loss(x, pred, combined_mask)

        if return_last_intermediate:
            return *outputs, mask, encoder_outputs[2]
        return *outputs, mask


# Video Models


@pretrained_model(
    {
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth",
    },
    default="mae_k400",
)
def mae_hiera_base_16x224(num_classes: int = 400, **kwdargs) -> MaskedAutoencoderSignHiera:
    return MaskedAutoencoderSignHiera(
        num_classes=num_classes,  # K400 has 400 classes
        input_size=(16, 224, 224),
        q_stride=(1, 2, 2),
        mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        sep_pos_embed=True,
        q_pool=2,
        **kwdargs,
    )


@pretrained_model(
    {
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth",
    },
    default="mae_k400",
)
def mae_hiera_base_128x224(num_classes: int = 400, **kwdargs) -> MaskedAutoencoderSignHiera:
    return MaskedAutoencoderSignHiera(
        num_classes=num_classes,  # K400 has 400 classes
        input_size=(128, 224, 224),
        q_stride=(1, 2, 2),
        mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        sep_pos_embed=True,
        q_pool=3,  # changed from 2 -> 3
        **kwdargs,
    )
