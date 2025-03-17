# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from fairseq2.context import RuntimeContext
from fairseq2.models.w2vbert import W2VBertConfig
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig
from fairseq2.nn.transformer import TransformerNormOrder


@dataclass
class SonarSpeechEncoderConfig:
    """Holds the configuration of a Sonar model."""

    w2v2_encoder_config: Wav2Vec2EncoderConfig
    """The configuration of the wav2vec 2.0 encoder model."""

    final_dropout_p: float
    """The dropout probability applied final projection"""

    model_dim: int
    """The output embedding dimension."""

    max_seq_len: int
    """The expected maximum sequence length."""

    pad_idx: Optional[int]
    """The index of the pad symbol in the vocabulary."""

    bos_idx: int
    """The index of bos symbol used in attention pooling"""

    num_decoder_layers: int
    """The number of Transformer decoder layers."""

    num_decoder_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    decoder_norm_order: TransformerNormOrder
    """Layer norm order in decoder modules."""

    ffn_inner_dim: int
    """The inner dimensionality of Transformer feed-forward networks."""

    dropout_p: float
    """The dropout probability in Transformer layers."""


def register_sonar_speech_encoder_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(SonarSpeechEncoderConfig)

    arch = registry.decorator

    w2vbert_registry = context.get_config_registry(W2VBertConfig)

    @arch("english")
    def basic() -> SonarSpeechEncoderConfig:
        w2vbert_config = w2vbert_registry.get("600m")

        return SonarSpeechEncoderConfig(
            w2v2_encoder_config=w2vbert_config.w2v2_config.encoder_config,
            final_dropout_p=0.1,
            model_dim=1024,
            max_seq_len=1024,
            pad_idx=1,
            bos_idx=2,
            num_decoder_layers=3,
            num_decoder_attn_heads=16,
            decoder_norm_order=TransformerNormOrder.POST,
            ffn_inner_dim=4096,
            dropout_p=0.1,
        )

    @arch("non_english")
    def multilingual() -> SonarSpeechEncoderConfig:
        w2vbert_config = w2vbert_registry.get("600m")

        return SonarSpeechEncoderConfig(
            w2v2_encoder_config=w2vbert_config.w2v2_config.encoder_config,
            final_dropout_p=0.1,
            model_dim=1024,
            max_seq_len=1024,
            pad_idx=1,
            bos_idx=2,
            num_decoder_layers=6,
            num_decoder_attn_heads=16,
            decoder_norm_order=TransformerNormOrder.POST,
            ffn_inner_dim=4096,
            dropout_p=0.1,
        )
