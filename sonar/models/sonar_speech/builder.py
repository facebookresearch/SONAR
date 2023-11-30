# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from fairseq2.models.transformer.frontend import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
)
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.models.w2vbert import w2vbert_archs
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderBuilder, Wav2Vec2EncoderConfig
from fairseq2.nn import Linear
from fairseq2.nn.embedding import Embedding, StandardEmbedding, init_scaled_embedding
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.position_encoder import PositionEncoder, SinusoidalPositionEncoder
from fairseq2.nn.transformer import TransformerNormOrder
from fairseq2.nn.transformer.attention import create_default_sdpa
from fairseq2.nn.transformer.decoder import (
    StandardTransformerDecoder,
    TransformerDecoder,
)
from fairseq2.nn.transformer.decoder_layer import (
    StandardTransformerDecoderLayer,
    TransformerDecoderLayer,
)
from fairseq2.nn.transformer.ffn import FeedForwardNetwork, StandardFeedForwardNetwork
from fairseq2.nn.transformer.layer_norm import create_standard_layer_norm
from fairseq2.nn.transformer.multihead_attention import (
    MultiheadAttention,
    StandardMultiheadAttention,
)
from fairseq2.typing import DataType, Device

from sonar.models.sonar_speech.model import SonarSpeechEncoderModel
from sonar.nn.encoder_pooler import AttentionEncoderOutputPooler, EncoderOutputPooler


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


sonar_speech_archs = ArchitectureRegistry[SonarSpeechEncoderConfig]("sonar_speech")

sonar_speech_arch = sonar_speech_archs.decorator


@sonar_speech_arch("english")
def _basic() -> SonarSpeechEncoderConfig:
    w2vbert_config = w2vbert_archs.get_config("600m")

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


@sonar_speech_arch("non_english")
def _multilingual() -> SonarSpeechEncoderConfig:
    w2vbert_config = w2vbert_archs.get_config("600m")

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


class SonarSpeechEncoderBuilder:
    """Builds modules of a SONAR model as described in
    # TODO add correct paper cite :cite:t`URL`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: SonarSpeechEncoderConfig
    w2v2_encoder_builder: Wav2Vec2EncoderBuilder
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: SonarSpeechEncoderConfig,
        w2v2_encoder_builder: Wav2Vec2EncoderBuilder,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param w2v2_encoder_builder:
            The wav2vec2 encoder builder.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        if w2v2_encoder_builder.config.model_dim != config.model_dim:
            raise ValueError(
                f"`model_dim` and `model_dim` of `w2v2_encoder_builder.config` must be equal, but are {config.model_dim} and {w2v2_encoder_builder.config.model_dim} instead."
            )

        self.config = config
        self.w2v2_encoder_builder = w2v2_encoder_builder
        self.device = device
        self.dtype = dtype

    def build_model(self) -> SonarSpeechEncoderModel:
        """Build sonar speech encoder model."""
        return SonarSpeechEncoderModel(
            encoder_frontend=self.w2v2_encoder_builder.build_frontend(),
            encoder=self.w2v2_encoder_builder.build_encoder(),
            layer_norm=self.build_w2v2_final_layer_norm(),
            final_dropout_p=self.config.final_dropout_p,
            encoder_pooler=self.build_attention_pooler(),
        ).to(device=self.device, dtype=self.dtype)

    def build_attention_pooler(self) -> EncoderOutputPooler:
        """Build Attention Pooler"""
        return AttentionEncoderOutputPooler(
            decoder_frontend=self.build_decoder_frontend(),
            decoder=self.build_decoder(),
            projection_out=self.build_projection_out(),
            bos_idx=self.config.bos_idx,
        )

    def build_decoder_frontend(self) -> TransformerFrontend:
        """Build a Transformer decoder front-end."""
        return TransformerEmbeddingFrontend(
            self.build_embedding(),
            self.build_pos_encoder(),
            dropout_p=self.config.dropout_p,
        )

    def build_pos_encoder(self) -> PositionEncoder:
        """Build position encoder"""
        return SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.max_seq_len,
        )

    def build_embedding(self) -> Embedding:
        """Build an embedding table."""
        return StandardEmbedding(
            num_embeddings=self.config.w2v2_encoder_config.model_dim,
            embedding_dim=self.config.model_dim,
            pad_idx=self.config.pad_idx,
            init_fn=init_scaled_embedding,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self.config.num_decoder_layers
        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return StandardTransformerDecoder(
            layers,
            norm_order=self.config.decoder_norm_order,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        num_heads = self.config.num_decoder_attn_heads

        return StandardTransformerDecoderLayer(
            self.build_attention(num_heads),
            self.build_attention(num_heads),
            self.build_ffn(),
            dropout_p=self.config.dropout_p,
            norm_order=self.config.decoder_norm_order,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self.config.dropout_p)

        return StandardMultiheadAttention(
            self.config.model_dim,
            num_heads,
            sdpa=sdpa,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            bias=True,
            norm_order=self.config.decoder_norm_order,
        )

    def build_w2v2_final_layer_norm(self) -> Optional[LayerNorm]:
        """Build w2v2 final layer norm"""
        if not self.config.w2v2_encoder_config.use_conformer:
            return None

        return create_standard_layer_norm(
            self.config.w2v2_encoder_config.model_dim,
        )

    def build_projection_out(self) -> Linear:
        """Build final projection linear layer"""
        return Linear(
            input_dim=self.config.model_dim,
            output_dim=self.config.model_dim,
            bias=False,
        )


def create_sonar_speech_encoder_model(
    config: SonarSpeechEncoderConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> SonarSpeechEncoderModel:
    """Create a SONAR speech encoder model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    w2v2_encoder_builder = Wav2Vec2EncoderBuilder(
        config.w2v2_encoder_config, device=device, dtype=dtype
    )

    sonar_builder = SonarSpeechEncoderBuilder(
        config, w2v2_encoder_builder, device=device, dtype=dtype
    )

    return sonar_builder.build_model()
