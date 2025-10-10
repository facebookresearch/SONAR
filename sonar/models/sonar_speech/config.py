# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Final, Optional

from fairseq2.models.transformer import TransformerNormOrder
from fairseq2.models.w2vbert import W2VBertConfig
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig
from fairseq2.runtime.config_registry import ConfigRegistrar, get_config
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver

SONAR_SPEECH_FAMILY: Final = "sonar_speech"


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


def _register_sonar_speech_encoder_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, SonarSpeechEncoderConfig)

    @arch("english", advanced=True)
    def basic(resolver: DependencyResolver) -> SonarSpeechEncoderConfig:
        w2vbert_config = get_config(resolver, W2VBertConfig, "600m")

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

    @arch("non_english", advanced=True)
    def multilingual(resolver: DependencyResolver) -> SonarSpeechEncoderConfig:
        w2vbert_config = get_config(resolver, W2VBertConfig, "600m")

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
