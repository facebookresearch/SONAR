# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------

import re

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutput, BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions)

"""
Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models
for use as a text tower in CLIP model.
Adapted from https://github.com/mlfoundations/open_clip/tree/main/src/open_clip
"""

# HF architecture dict:
arch_dict = {
    # https://huggingface.co/docs/transformers/model_doc/t5
    "t5": {
        "config_names": {
            # unlimited seqlen
            # https://github.com/google-research/text-to-text-transfer-transformer/issues/273
            # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/t5/modeling_t5.py#L374
            "context_length": "",
            "vocab_size": "vocab_size",
            "width": "d_model",
            "heads": "num_heads",
            "layers": "num_layers",
            "layer_attr": "block",
            "token_embeddings_attr": "embed_tokens",
        },
        "pooler": "mean_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/bart
    "bart": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "d_model",
            "heads": "encoder_attention_heads",
            "layers": "encoder_layers",
            "layer_attr": "layers",
            "token_embeddings_attr": "embed_tokens",
        },
        "pooler": "mean_pooler",
    },
}


def _camel2snake(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


_POOLERS = {}


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: torch.Tensor):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    """Max pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: torch.Tensor):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, attention_mask: torch.Tensor):
        if (
            self.use_pooler_output
            and isinstance(
                x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)
            )
            and (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x.last_hidden_state[:, self.cls_token_position, :]


@register_pooler
class ClsLastHiddenStatePooler(nn.Module):
    """CLS token pooling
    NOTE: this is equivalent to ClsPooler above with use_pooler_output=False
    """

    def __init__(self):
        super().__init__()
        self.cls_token_position = 0

    def forward(self, x: BaseModelOutput, attention_mask: torch.Tensor):
        return x.last_hidden_state[:, self.cls_token_position, :]


class HFTextEncoderForClip(nn.Module):
    """HuggingFace model adapter"""

    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int,
        config: PretrainedConfig = None,
        pooler_type: str = None,
        proj: str = None,
        pretrained: bool = True,
        output_tokens: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim

        uses_transformer_pooler = pooler_type == "cls_pooler"

        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            create_func, model_args = (
                (AutoModel.from_pretrained, model_name_or_path)
                if pretrained
                else (AutoModel.from_config, self.config)
            )
            if hasattr(self.config, "is_encoder_decoder") and self.config.is_encoder_decoder:
                self.transformer = create_func(model_args)
                self.transformer = self.transformer.get_encoder()
            else:
                self.transformer = create_func(
                    model_args, add_pooling_layer=uses_transformer_pooler
                )
        else:
            self.config = config
            self.transformer = AutoModel.from_config(config)
        if pooler_type is None:  # get default arch pooler
            pooler_type = arch_dict[self.config.model_type]["pooler"]

        self.vocab_size = getattr(self.config, "vocab_size", 0)
        self.context_length = getattr(self.config, "max_position_embeddings", 0)

        self.pooler = _POOLERS[pooler_type]()

        d_model = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["width"])
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

    def forward(self, input_ids: torch.Tensor):
        attn_mask = (input_ids != self.config.pad_token_id).long()
        out = self.transformer(input_ids=input_ids, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
            return

        encoder = (
            self.transformer.encoder if hasattr(self.transformer, "encoder") else self.transformer
        )
        layer_list = getattr(
            encoder, arch_dict[self.config.model_type]["config_names"]["layer_attr"]
        )
        print(f"Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model")
        embeddings = getattr(
            self.transformer,
            arch_dict[self.config.model_type]["config_names"]["token_embeddings_attr"],
        )
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass
