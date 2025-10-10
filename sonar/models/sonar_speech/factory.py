# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from fairseq2.models.transformer import (
    FeedForwardNetwork,
    IdentityBias,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEmbeddingFrontend,
    TransformerEncoder,
    TransformerFrontend,
    create_default_sdpa,
)
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderFactory, Wav2Vec2Frontend
from fairseq2.nn import (
    Embedding,
    LayerNorm,
    Linear,
    PositionEncoder,
    SinusoidalPositionEncoder,
    StandardEmbedding,
    StandardLayerNorm,
    init_scaled_embedding,
)

from sonar.models.sonar_speech.config import SonarSpeechEncoderConfig
from sonar.models.sonar_speech.model import SonarSpeechEncoderModel
from sonar.nn.encoder_pooler import AttentionEncoderOutputPooler, EncoderOutputPooler


def _create_sonar_speech_encoder_model(
    config: SonarSpeechEncoderConfig,
) -> SonarSpeechEncoderModel:
    return SonarSpeechEncoderFactory(config).create_model()


class SonarSpeechEncoderFactory:
    config: SonarSpeechEncoderConfig

    def __init__(self, config: SonarSpeechEncoderConfig) -> None:
        if config.w2v2_encoder_config.model_dim != config.model_dim:
            raise ValueError(
                f"`config.model_dim` and `config.w2v2_encoder_config.model_dim` must be equal, but are {config.model_dim} and {config.w2v2_encoder_config.model_dim} instead."
            )

        self.config = config

    def create_model(self) -> SonarSpeechEncoderModel:
        encoder_frontend, encoder = self.create_encoder()

        return SonarSpeechEncoderModel(
            encoder_frontend=encoder_frontend,
            encoder=encoder,
            layer_norm=self.create_w2v2_final_layer_norm(),
            final_dropout_p=self.config.final_dropout_p,
            encoder_pooler=self.create_attention_pooler(),
        )

    def create_encoder(self) -> tuple[Wav2Vec2Frontend, TransformerEncoder]:
        factory = Wav2Vec2EncoderFactory(self.config.w2v2_encoder_config)

        encoder_frontend = factory.create_encoder_frontend()

        encoder = factory.create_encoder()

        return encoder_frontend, encoder

    def create_attention_pooler(self) -> EncoderOutputPooler:
        return AttentionEncoderOutputPooler(
            decoder_frontend=self.create_decoder_frontend(),
            decoder=self.create_decoder(),
            projection_out=self.create_projection_out(),
            bos_idx=self.config.bos_idx,
        )

    def create_decoder_frontend(self) -> TransformerFrontend:
        return TransformerEmbeddingFrontend(
            self.config.model_dim,
            self.create_embedding(),
            self.create_pos_encoder(),
            dropout_p=self.config.dropout_p,
        )

    def create_pos_encoder(self) -> PositionEncoder:
        return SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.max_seq_len,
        )

    def create_embedding(self) -> Embedding:
        return StandardEmbedding(
            num_embeddings=self.config.w2v2_encoder_config.model_dim,
            embed_dim=self.config.model_dim,
            pad_idx=self.config.pad_idx,
            init_fn=init_scaled_embedding,
        )

    def create_decoder(self) -> TransformerDecoder:
        num_layers = self.config.num_decoder_layers
        layers = [self.create_decoder_layer() for _ in range(num_layers)]

        return StandardTransformerDecoder(layers)

    def create_decoder_layer(self) -> TransformerDecoderLayer:
        num_heads = self.config.num_decoder_attn_heads

        return StandardTransformerDecoderLayer(
            self_attn=self.create_attention(num_heads),
            self_attn_layer_norm=self.create_layer_norm(),
            encoder_decoder_attn=self.create_attention(num_heads),
            encoder_decoder_attn_layer_norm=self.create_layer_norm(),
            ffn=self.create_ffn(),
            ffn_layer_norm=self.create_layer_norm(),
            dropout_p=self.config.dropout_p,
            norm_order=self.config.decoder_norm_order,
        )

    def create_attention(self, num_heads: int) -> MultiheadAttention:
        sdpa = create_default_sdpa(bias=IdentityBias(), dropout_p=self.config.dropout_p)

        return StandardMultiheadAttention(
            self.config.model_dim,
            num_heads,
            sdpa=sdpa,
        )

    def create_ffn(self) -> FeedForwardNetwork:
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            bias=True,
        )

    def create_layer_norm(self) -> LayerNorm:
        model_dim = self.config.model_dim
        return StandardLayerNorm(model_dim, bias=True)

    def create_w2v2_final_layer_norm(self) -> Optional[LayerNorm]:
        if not self.config.w2v2_encoder_config.use_conformer:
            return None
        return StandardLayerNorm(self.config.w2v2_encoder_config.model_dim, bias=True)

    def create_projection_out(self) -> Linear:
        return Linear(
            input_dim=self.config.model_dim,
            output_dim=self.config.model_dim,
            bias=False,
        )
