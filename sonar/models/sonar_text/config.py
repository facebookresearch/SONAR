# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from fairseq2.context import RuntimeContext
from fairseq2.data import VocabularyInfo


@dataclass
class SonarTextEncoderConfig:
    """Holds the configuration of an SonarTextEncoder model."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The expected maximum sequence length.
        Corresponds to `max_source_positions` in fairseq
    """

    vocab_info: VocabularyInfo
    """the vocabulary information."""

    num_encoder_layers: int
    """The number of Transformer encoder layers."""

    num_decoder_layers: int
    """The number of Transformer decoder layers."""

    num_encoder_attn_heads: int
    """The number of attention heads in Transformer encoder layers."""

    num_decoder_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    ffn_inner_dim: int
    """The dimensionality of inner projection layers in Transformer feed-forward
    networks."""

    pooling: str
    """ Pooling option (one of `max`, `mean`, `last`) applied over encoded_output
        to get a fix size sequence representation vector """

    # list of less common parameters interpreted by the builder
    embedding_dim: Optional[int] = None
    """ Dimension of embedding, if it is not the same as `model_dim` (in this case, attention pooling is required) """

    decoder_ffn_inner_dim: Optional[int] = None
    """ Dimensionality of inner projection in the attention pooler"""

    activation_fn: str = "ReLU"
    """ activation function to use in FeedForward network of Transformers; None corresponds to ReLu"""

    layernorm_embedding: bool = False
    """ If True, apply LayerNorm on sequence embeddings"""

    no_scale_embedding: bool = False
    """if False, multiply sequence embeddings by sqrt(model_dim) before positional encoding"""

    no_token_positional_embeddings: bool = False
    """If False, add positional encoding to the sequence embeddings"""

    learned_pos: bool = False
    """if True, use LearnedPositionEncoder instead of SinusoidalPositionEncoder"""

    emb_dropout_p: float = 0.1
    """The dropout probability in embeddings layers."""

    attention_dropout_p: float = 0.1
    """The dropout probability in MHA layers."""

    activation_dropout_p: float = 0.1
    """The dropout probability in FF layers."""

    normalize_before: bool = False
    """Using LayerNorm at the beginning of MHA layer if True"""

    _from_fairseq: bool = False
    """if True, do max_seq_len += pad_idx + 1 for retro-compatibgiility with fairseq trained models"""


def register_sonar_text_encoder_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(SonarTextEncoderConfig)

    arch = registry.decorator

    @arch("basic")
    def basic() -> SonarTextEncoderConfig:
        return SonarTextEncoderConfig(
            model_dim=1024,
            vocab_info=VocabularyInfo(
                size=256206, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=1
            ),
            learned_pos=False,
            no_scale_embedding=False,
            emb_dropout_p=0.1,
            attention_dropout_p=0.1,
            activation_dropout_p=0.1,
            max_seq_len=512,
            pooling="mean",
            no_token_positional_embeddings=False,
            layernorm_embedding=False,
            activation_fn="ReLU",
            normalize_before=False,
            num_encoder_layers=24,
            num_decoder_layers=24,
            num_encoder_attn_heads=16,
            num_decoder_attn_heads=16,
            ffn_inner_dim=1024 * 8,
            _from_fairseq=True,
        )

    @arch("small")
    def small(vocab_size=32005, depth=6, hidden_dim=1024 * 4) -> SonarTextEncoderConfig:
        config = basic()
        config.vocab_info = VocabularyInfo(
            size=vocab_size, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=1
        )
        config.num_encoder_layers = depth
        config.num_decoder_layers = depth
        config.ffn_inner_dim = hidden_dim
        return config


@dataclass
class SonarTextDecoderConfig:
    """Holds the configuration of an SonarDecoder model."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The expected maximum sequence length.
        Corresponds to `max_source_positions` in fairseq
    """

    vocab_info: VocabularyInfo
    """The vocabulary information."""

    activation_fn: str
    """ activation function to use in FeedForward network of Transformers; None corresponds to ReLu"""

    layernorm_embedding: bool
    """ If True, apply LayerNorm on sequence embeddings"""

    no_scale_embedding: bool
    """if False, multiply sequence embeddings by sqrt(model_dim) before positional encoding"""

    no_token_positional_embeddings: bool
    """If False, add positional encoding to the sequence embeddings"""

    learned_pos: bool
    """if True, use LearnedPositionEncoder instead of SinusoidalPositionEncoder"""

    emb_dropout_p: float
    """The dropout probability in embeddings layers."""

    attention_dropout_p: float
    """The dropout probability in MHA layers."""

    activation_dropout_p: float
    """The dropout probability in FF layers."""

    normalize_before: bool
    """Using LayerNorm at the beginning of MHA layer if True"""

    num_encoder_layers: int
    """The number of Transformer encoder layers."""

    num_decoder_layers: int
    """The number of Transformer decoder layers."""

    num_encoder_attn_heads: int
    """The number of attention heads in Transformer encoder layers."""

    num_decoder_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    ffn_inner_dim: int
    """The dimensionality of inner projection layers in Transformer feed-forward
    networks."""

    input_dim: Optional[int] = None
    """The dimensionality of the input. If None, model_dim is used instead."""


def register_sonar_text_decoder_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(SonarTextDecoderConfig)

    arch = registry.decorator

    @arch("basic")
    def basic() -> SonarTextDecoderConfig:
        return SonarTextDecoderConfig(
            model_dim=1024,
            max_seq_len=512,
            vocab_info=VocabularyInfo(
                size=256206, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=1
            ),
            learned_pos=False,
            no_scale_embedding=False,
            emb_dropout_p=0.1,
            attention_dropout_p=0.1,
            activation_dropout_p=0.1,
            no_token_positional_embeddings=False,
            layernorm_embedding=False,
            activation_fn="ReLU",
            normalize_before=True,
            num_encoder_layers=24,
            num_decoder_layers=24,
            num_encoder_attn_heads=16,
            num_decoder_attn_heads=16,
            ffn_inner_dim=1024 * 8,
        )

    @arch("small")
    def small(vocab_size=32005, depth=6, hidden_dim=1024 * 4) -> SonarTextDecoderConfig:
        config = basic()
        config.vocab_info = VocabularyInfo(
            size=vocab_size, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=1
        )
        config.num_encoder_layers = depth
        config.num_decoder_layers = depth
        config.ffn_inner_dim = hidden_dim
        return config

    @arch("toy")
    def toy() -> SonarTextDecoderConfig:
        """A very small decoder (67K parameters), exclusively for testing purposes."""
        return SonarTextDecoderConfig(
            model_dim=32,
            max_seq_len=512,
            vocab_info=VocabularyInfo(
                size=1024, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=1
            ),
            learned_pos=False,
            no_scale_embedding=False,
            emb_dropout_p=0.1,
            attention_dropout_p=0.1,
            activation_dropout_p=0.1,
            no_token_positional_embeddings=False,
            layernorm_embedding=False,
            activation_fn="ReLU",
            normalize_before=True,
            num_encoder_layers=2,
            num_decoder_layers=2,
            num_encoder_attn_heads=4,
            num_decoder_attn_heads=4,
            ffn_inner_dim=128,
        )
