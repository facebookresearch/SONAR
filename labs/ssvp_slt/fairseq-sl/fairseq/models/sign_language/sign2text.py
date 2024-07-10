import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch.nn as nn
from fairseq import checkpoint_utils, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from fairseq.models import FairseqEncoder, FairseqEncoderDecoderModel, register_model
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from omegaconf import II
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class Sign2TextTransformerConfig(FairseqDataclass):
    """Add model-specific arguments to the parser."""

    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(default=0.1, metadata={"help": "dropout probability for attention weights"})
    activation_dropout: float = field(default=0.1, metadata={"help": "dropout probability after activation in FFN."})
    encoder_embed_dim: int = field(default=512, metadata={"help": "encoder embedding dimension"})
    encoder_ffn_embed_dim: int = field(default=2048, metadata={"help": "encoder embedding dimension for FFN"})
    encoder_layers: int = field(default=12, metadata={"help": "num encoder layers"})
    encoder_attention_heads: int = field(default=8, metadata={"help": "num encoder attention heads"})
    encoder_normalize_before: bool = field(
        default=True, metadata={"help": "apply layernorm before each encoder block"}
    )
    encoder_layerdrop: float = field(default=0.0, metadata={"help": "probability of dropping a transformer layer"})
    decoder_embed_dim: int = field(default=512, metadata={"help": "decoder embedding dimension"})
    decoder_ffn_embed_dim: int = field(default=2048, metadata={"help": "decoder embedding dimension for FFN"})
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(default=8, metadata={"help": "num decoder attention heads"})
    decoder_output_dim: int = field(
        default=512,
        metadata={"help": "decoder output dimension (extra linear layer if different from decoder embed dim)"},
    )
    decoder_normalize_before: bool = field(
        default=True, metadata={"help": "apply layernorm before each decoder block"}
    )
    decoder_layerdrop: float = field(default=0.0, metadata={"help": "decoder layerdrop chance"})
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    layernorm_embedding: bool = field(default=False, metadata={"help": "add layernorm to embedding"})
    no_scale_embedding: bool = field(default=False, metadata={"help": "if True, dont scales embeddings"})
    load_pretrained_encoder_from: Optional[str] = field(
        default=None, metadata={"help": "model to take encoder weights from (for initialization)"}
    )
    load_pretrained_decoder_from: Optional[str] = field(
        default=None, metadata={"help": "model to take decoder weights from (for initialization)"}
    )
    max_source_positions: int = II("task.max_source_positions")
    max_target_positions: int = II("task.max_target_positions")
    feature_dim: int = field(default=768, metadata={"help": "Feature dimension"})


@register_model("sign2text_transformer", dataclass=Sign2TextTransformerConfig)
class Sign2TextTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg, feat_dim):
        encoder = Sign2TextTransformerEncoder(cfg, feat_dim)
        pretraining_path = getattr(cfg, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(f"skipped pretraining because {pretraining_path} does not exist")
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_decoder(cls, cfg, task, embed_tokens):
        decoder = TransformerDecoder(cfg, task.target_dictionary, embed_tokens)
        pretraining_path = getattr(cfg, "load_pretrained_decoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(f"skipped pretraining because {pretraining_path} does not exist")
            else:
                decoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=decoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained decoder from: {pretraining_path}")
        return decoder

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        feat_dim = cfg.feature_dim

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(task.target_dictionary, cfg.decoder_embed_dim)
        encoder = cls.build_encoder(cfg, feat_dim)
        decoder = cls.build_decoder(cfg, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, source, padding_mask, prev_output_tokens, **kwargs):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(source=source, padding_mask=padding_mask)
        decoder_out = self.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out)
        return decoder_out


class Sign2TextTransformerEncoder(FairseqEncoder):
    """Sign-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, cfg, feat_dim: int):
        super().__init__(None)

        self.num_updates = 0

        self.dropout_module = FairseqDropout(p=cfg.dropout, module_name=self.__class__.__name__)
        self.embed_scale = math.sqrt(cfg.encoder_embed_dim)
        if cfg.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.feat_proj = nn.Linear(feat_dim, cfg.encoder_embed_dim)

        self.embed_positions = PositionalEmbedding(cfg.max_source_positions, cfg.encoder_embed_dim, self.padding_idx)

        self.transformer_layers = nn.ModuleList([TransformerEncoderLayer(cfg) for _ in range(cfg.encoder_layers)])
        if cfg.encoder_normalize_before:
            self.layer_norm = LayerNorm(cfg.encoder_embed_dim)
        else:
            self.layer_norm = None

    def forward(self, source, padding_mask, return_all_hiddens=False, **kwargs):
        x = source

        # src_tokens B x seq_len x Fs
        x = self.feat_proj(x).transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        # x: seq_len x B x H
        x = self.embed_scale * x

        # encoder_padding_mask: B x seq_len x Fs
        positions = self.embed_positions(padding_mask).transpose(0, 1)
        # positions: seq_len x B x H, --> it's not, last dimension is 262144 instead of H
        x += positions
        x = self.dropout_module(x)

        encoder_states = []

        for layer in self.transformer_layers:
            x = layer(x, padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask] if padding_mask.any() else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
