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
SignHiera PyTorch model
Adapted from https://github.com/facebookresearch/hiera/blob/v0.1.2/hiera/hiera.py

Main changes made:
- made clip size variable (to increase from 16 to 128)
- added temporal attention masking at mask unit (MU) level
- added feature extraction
- added loading from pretrained CLIP/FLIP model with Hiera-based vision tower
"""

import math
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, Mlp

from .sign_hiera_utils import (Reroll, Unroll, conv_nd, do_masked_conv,
                               do_pool, pretrained_model)


class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        """
        Args:
        - dim, dim_out: The input and output feature dimensions.
        - heads: The number of attention heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride

        self.head_dim = dim_out // heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Input should be of shape [batch, tokens, channels]."""

        B, N, _ = x.shape
        num_windows = (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1

        qkv = (
            self.qkv(x)
            .reshape(B, -1, num_windows, 3, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # [B, N, 1] -> [B, W, N//W]
        attn_mask = attn_mask.reshape(B, -1, num_windows).permute(0, 2, 1)
        # [B, W, N//W] -> [B, H=1, W, L=1, N//W]
        attn_mask = attn_mask[:, None, :, None, :]
        attn_mask = torch.where(attn_mask == 0, -float("inf"), attn_mask)

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            q = (
                q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            )

        # Only use attention mask when doing global attention.
        # This is because we do attention masking at the level of mask units, so when doing mask unit attention,
        # the attention mask would either be true or false for the full window, which is not useful
        if hasattr(F, "scaled_dot_product_attention"):
            # Note: the original paper did *not* use SDPA, it's a free boost!
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask if not self.use_mask_unit_attn else None
            )
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2)
            if not self.use_mask_unit_attn:
                attn += attn_mask
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = x.transpose(1, 3).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        return x


class SignHieraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.norm1 = norm_layer(dim)
        self.attn = MaskUnitAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn
        )

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Attention + Q Pooling
        x_norm = self.norm1(x)
        if self.dim != self.dim_out:
            x = do_pool(self.proj(x_norm), stride=self.attn.q_stride)
        x = x + self.drop_path(self.attn(x_norm, attn_mask=attn_mask))

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Head(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dropout_rate: float = 0.0,
        act_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.softmax(dim=-1),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.projection = nn.Linear(dim, num_classes)
        # act_fun for eval and testing only
        self.act_func = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.projection(x)
        if not self.training:
            x = self.act_func(x)
        return x


class PatchEmbed(nn.Module):
    """Patch embed that supports any number of spatial dimensions (1d, 2d, 3d)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
    ):
        super().__init__()

        # Support any number of spatial dimensions
        self.spatial_dims = len(kernel)
        self.proj = conv_nd(self.spatial_dims)(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = do_masked_conv(x, self.proj, mask)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        return x


class SignHiera(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, ...] = (224, 224),
        in_chans: int = 3,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        num_classes: int = 1000,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, ...] = (2, 2),
        mask_unit_size: Tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)
        # mask_unit_attn: which stages use mask unit attention?
        mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: Tuple[int, ...] = (7, 7),
        patch_stride: Tuple[int, ...] = (4, 4),
        patch_padding: Tuple[int, ...] = (3, 3),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        head_dropout: float = 0.0,
        head_init_scale: float = 0.001,
        sep_pos_embed: bool = False,
        **kwargs,
    ):
        super().__init__()

        depth = sum(stages)
        self.tokens_spatial_shape = [i // s for i, s in zip(input_size, patch_stride)]
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(mask_unit_size)
        flat_q_stride = math.prod(q_stride)

        assert q_pool < len(stages)
        self.q_pool, self.q_stride = q_pool, q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
        self.mask_spatial_shape = [
            i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size)
        ]
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        self.patch_stride = patch_stride
        self.feature_dim = int(embed_dim * dim_mul * len(self.stage_ends))

        self.patch_embed = PatchEmbed(
            in_chans, embed_dim, patch_kernel, patch_stride, patch_padding
        )

        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    embed_dim,
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[0], embed_dim)
            )
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # Setup roll and reroll modules
        self.unroll = Unroll(input_size, patch_stride, [q_stride] * len(self.stage_ends[:-1]))
        self.reroll = Reroll(
            input_size,
            patch_stride,
            [q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            q_pool,
        )
        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[:q_pool]]
        self.q_pool_blocks = set(q_pool_blocks)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        cur_stage = 0
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # Mask unit or global attention.
            # Lag by 1 block, so that global attention,
            # applied post pooling on lower resolution
            use_mask_unit_attn = mask_unit_attn[cur_stage]

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride

            block = SignHieraBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.norm = norm_layer(embed_dim)
        self.head = Head(embed_dim, num_classes, dropout_rate=head_dropout)

        # Initialize everything
        if sep_pos_embed:
            nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(self._init_weights))
        self.head.projection.weight.data.mul_(head_init_scale)
        self.head.projection.bias.data.mul_(head_init_scale)

    def _init_weights(self, m: nn.Module, init_bias: float = 0.02) -> None:
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, init_bias)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> List[str]:
        if self.sep_pos_embed:
            return ["pos_embed_spatial", "pos_embed_temporal"]
        else:
            return ["pos_embed"]

    def get_random_mask(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        B = x.shape[0]
        # Tokens selected for masking at mask unit level
        num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(B, num_windows, device=x.device)

        # Attention mask indicates tokens that are non-padding (1) or padding (0)
        # Out of the non-padding tokens we take the one with the highest noise
        # And bump up noise to 100 to guarantee that it gets masked
        # We therefore ensure that at least one masked patch is non-padding
        # This is necessary because we only compute loss on non-padding tokens, i.e. loss would otherwise be NaN
        if attn_mask is not None:
            noise_mask = torch.argmax(noise * attn_mask, dim=1)
            noise[torch.arange(noise.size(0)), noise_mask] = 100.0
            # First (1x)x#MUyx#MUx tokens will not be padding, so by setting low value we guarantee that
            # at least one non-padding token will be kept
            noise[
                torch.arange(noise.size(0)),
                torch.randint(
                    self.mask_spatial_shape[-2] * self.mask_spatial_shape[-1],
                    (noise.size(0),),
                ),
            ] = -100

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([B, num_windows], device=x.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.bool()

    def get_attention_mask(self, padding: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Creates a temporal attention mask based on the number of padding frames
        """
        attn_mask = torch.ones(
            (padding.shape[0], math.prod(self.mask_spatial_shape)), device=device
        )

        # #MUs that are padded
        num_padding_mus = (
            (padding // (self.mask_unit_size[0] * self.patch_stride[0]))
            * self.mask_spatial_shape[1]
            * self.mask_spatial_shape[2]
        )

        for i in range(num_padding_mus.shape[0]):
            if num_padding_mus[i] > 0:
                attn_mask[i, -num_padding_mus[i] :] = 0

        return attn_mask

    def get_pos_embed(self) -> torch.Tensor:
        if self.sep_pos_embed:
            return self.pos_embed_spatial.repeat(
                1, self.tokens_spatial_shape[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                dim=1,
            )
        else:
            return self.pos_embed

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        mask should be a boolean tensor of shape [B, #MUt*#MUy*#MUx] where #MU are the number of mask units in that dim.
        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
        """

        # Slowfast training passes in a list
        if isinstance(x, list):
            x = x[0]

        if attn_mask is None:
            attn_mask = torch.ones(
                (x.shape[0], math.prod(self.mask_spatial_shape)), device=x.device
            )

        intermediates = []

        # Zero out both mask tokens and padding (attn_mask == 0) tokens in patch embedding conv
        patch_embed_mask = torch.logical_and(mask, attn_mask) if mask is not None else attn_mask
        x = self.patch_embed(
            x, mask=patch_embed_mask.view(x.shape[0], 1, *self.mask_spatial_shape)
        )

        x = x + self.get_pos_embed()
        x = self.unroll(x)

        # get spatial view of attention mask
        attn_mask = attn_mask.view(attn_mask.shape[0], *self.mask_spatial_shape)

        # upsample by mask unit size, then flatten and unsqueeze channel dimension
        for i, s in enumerate(self.mask_unit_size):
            attn_mask = attn_mask.repeat_interleave(s, i)
        attn_mask = attn_mask.view(attn_mask.shape[0], -1).unsqueeze(-1)
        attn_mask = self.unroll(attn_mask)

        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(
                x.shape[0], -1, x.shape[-1]
            )
            attn_mask = attn_mask[mask[..., None].tile(1, self.mu_size, attn_mask.shape[2])].view(
                attn_mask.shape[0], -1, attn_mask.shape[-1]
            )

        for i, blk in enumerate(self.blocks):
            x = blk(x, attn_mask=attn_mask)

            # Downsample attention mask
            if i in self.q_pool_blocks:
                attn_mask = (
                    attn_mask.view(attn_mask.shape[0], math.prod(self.q_stride), -1, 1)
                    .max(1)
                    .values
                )

            # if return_intermediates and #i in self.stage_ends:
            if i in self.stage_ends:
                intermediates.append(self.reroll(x, i, mask=mask))

        if mask is None:
            x = x.mean(dim=1)
            x = self.norm(x)
            x = self.head(x)

        # x may not always be in spatial order here.
        # e.g. if q_pool = 2, mask_unit_size = (8, 8), and
        # q_stride = (2, 2), not all unrolls were consumed,
        # intermediates[-1] is x in spatial order
        if return_intermediates:
            return x, intermediates

        return x

    def extract_features(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding: Optional[torch.Tensor] = None,
        return_attn_mask: bool = False,
    ) -> torch.Tensor:
        """
        Extract features from a video tensor x

        """

        if attention_mask is not None:
            attn_mask = attention_mask
        elif padding is not None:
            attn_mask = self.get_attention_mask(padding, device=x.device)
        else:
            attn_mask = None

        x = self.forward(x, attn_mask=attn_mask, return_intermediates=True)

        # Take last intermediate features and mean pool over spatial dimensions
        x = x[1][-1]
        x = x.mean(dim=(2, 3))

        # Remove padding
        if padding is not None:
            assert x.dim() == 3
            x_list = []
            num_padding_units = padding // (self.mask_unit_size[0] * self.patch_stride[0])
            for i, features in enumerate(x):
                x_list.append(
                    features[: -num_padding_units[i]] if num_padding_units[i] > 0 else features
                )
            x = torch.concatenate(x_list, dim=0)

        return x.view(-1, x.shape[-1])

    def load_weights(self, checkpoint_path: str) -> None:
        """
        Loads SignHiera weights from a pretrained model checkpoint
        """

        checkpoint = torch.load(checkpoint_path)

        if "model" in checkpoint.keys():
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint["model_state"]

        _mismatch = False
        new_checkpoint_model = {}
        for k, v in checkpoint_model.items():
            if k in self.state_dict() and self.state_dict()[k].shape != v.shape:
                print(f"Pruning {k} due to size mismatch")
                _mismatch = True
            else:
                new_checkpoint_model[k] = v

        if _mismatch:
            print(
                "Warning: Not all parameters from the checkpoint state dict match the target shape. "
                "Please check whether this is intended, e.g. when changing the clip size, or not."
            )

        # load pre-trained model
        msg = self.load_state_dict(new_checkpoint_model, strict=False)
        print(msg)

    @classmethod
    def from_clip_model(cls, model_id: str, clip_model_path: str) -> nn.Module:
        """
        Loads a SignHiera encoder from a pretrained CLIP model with SignHiera vision tower
        """

        import sys

        from ssvp_slt.modeling.clip import CLIP, CLIPTextCfg, CLIPVisionCfg

        checkpoint = torch.load(clip_model_path)
        args = checkpoint["args"]
        model_params = checkpoint["clip"]

        vision_cfg = CLIPVisionCfg(
            model_id=args.model,
            proj="mlp",
        )

        # FIXME: might cause errors if proj and pooler are not `mlp` and `mean_pooler`
        text_cfg = CLIPTextCfg(
            hf_model_name=args.text_model_name_or_path,
            proj="mlp",
            pooler_type="mean_pooler",
        )
        clip = CLIP(embed_dim=768, vision_cfg=vision_cfg, text_cfg=text_cfg, output_dict=True)

        print(f"Loading CLIP weights from {clip_model_path}")
        msg = clip.load_state_dict(model_params)
        print(msg)

        print("Loading SignHiera weights from CLIP vision tower")
        model = sys.modules[__name__].__dict__[model_id](**args.__dict__)
        msg = model.load_state_dict(clip.visual.transformer.state_dict(), strict=False)
        print(msg)

        return model


# Video models


@pretrained_model(
    {
        "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth",
    },
    default="mae_k400_ft_k400",
)
def hiera_base_16x224(num_classes: int = 400, **kwdargs) -> SignHiera:
    return SignHiera(
        num_classes=num_classes,  # K400 has 400 classes
        input_size=(16, 224, 224),
        q_stride=(1, 2, 2),
        mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        sep_pos_embed=True,
        **kwdargs,
    )


@pretrained_model(
    {
        "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth",
    },
    default="mae_k400_ft_k400",
)
def hiera_base_128x224(num_classes: int = 400, **kwdargs) -> SignHiera:
    return SignHiera(
        num_classes=num_classes,  # K400 has 400 classes
        input_size=(128, 224, 224),
        q_stride=(1, 2, 2),
        mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        sep_pos_embed=True,
        q_pool=3,
        **kwdargs,
    )
