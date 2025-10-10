# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, cast

import torch.nn
from fairseq2.models.transformer import (
    AttentionBias,
    CausalAttentionBias,
    FeedForwardNetwork,
    IdentityBias,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEmbeddingFrontend,
    TransformerEncoderLayer,
    TransformerFrontend,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.nn import (
    LayerNorm,
    LearnedPositionEncoder,
    Linear,
    PositionEncoder,
    SinusoidalPositionEncoder,
    StandardEmbedding,
    StandardLayerNorm,
    TiedProjection,
    init_scaled_embedding,
)
from torch.nn import Parameter

from sonar.models.sonar_text.config import (
    SonarTextDecoderConfig,
    SonarTextEncoderConfig,
)
from sonar.models.sonar_text.model import Pooling, SonarTextTransformerEncoderModel
from sonar.nn.conditional_decoder_model import ConditionalTransformerDecoderModel
from sonar.nn.encoder_pooler import AttentionEncoderOutputPooler, EncoderOutputPooler


def _create_sonar_text_encoder_model(
    config: SonarTextEncoderConfig,
) -> SonarTextTransformerEncoderModel:
    return SonarTextEncoderFactory(config).create_model()


class SonarTextEncoderFactory:
    config: SonarTextEncoderConfig

    def __init__(self, config: SonarTextEncoderConfig) -> None:
        self.config = config

        if self.config._from_fairseq:
            assert self.config.vocab_info.pad_idx is not None

            self.config.max_seq_len += self.config.vocab_info.pad_idx + 1

        self.transformer_normalize_order = (
            TransformerNormOrder.PRE
            if self.config.normalize_before
            else TransformerNormOrder.POST
        )

    @property
    def embedding_dim(self) -> int:
        """Return the embedding_dim, which by default equals model_dim, but may differ with attention pooling"""
        return self.config.embedding_dim or self.config.model_dim

    def create_model(self) -> SonarTextTransformerEncoderModel:
        embed = StandardEmbedding(
            num_embeddings=self.config.vocab_info.size,
            embed_dim=self.config.model_dim,
            pad_idx=self.config.vocab_info.pad_idx,
            init_fn=init_scaled_embedding,
        )

        pos_encoder: Optional[PositionEncoder] = None
        if not self.config.no_token_positional_embeddings:
            if self.config.learned_pos:
                pos_encoder = LearnedPositionEncoder(
                    encoding_dim=self.config.model_dim,
                    max_seq_len=self.config.max_seq_len,
                )
            else:
                pos_encoder = SinusoidalPositionEncoder(
                    encoding_dim=self.config.model_dim,
                    max_seq_len=self.config.max_seq_len,
                    _legacy_pad_idx=self.config.vocab_info.pad_idx,
                )

        embedding_frontend = TransformerEmbeddingFrontend(
            model_dim=self.config.model_dim,
            embed=embed,
            pos_encoder=pos_encoder,
            no_scale=self.config.no_scale_embedding,
            dropout_p=self.config.emb_dropout_p,
        )

        transformer_layers = [
            self.create_encoder_layer() for _ in range(self.config.num_encoder_layers)
        ]
        encoder = StandardTransformerEncoder(transformer_layers)
        pooling = getattr(Pooling, self.config.pooling.upper())
        if pooling == Pooling.ATTENTION:
            pooler = self.create_attention_pooler()
        else:
            pooler = None

        return SonarTextTransformerEncoderModel(
            encoder_frontend=embedding_frontend,
            encoder=encoder,
            layer_norm=self.create_layer_norm(dim=self.config.model_dim),
            pooling=pooling,
            pooler=pooler,
            max_source_seq_len=self.config.max_seq_len,
        )

    def create_encoder_layer(self) -> TransformerEncoderLayer:
        return StandardTransformerEncoderLayer(
            self_attn=self.create_attention(),
            self_attn_layer_norm=self.create_layer_norm(dim=self.config.model_dim),
            ffn=self.create_ffn(),
            ffn_layer_norm=self.create_layer_norm(dim=self.config.model_dim),
            dropout_p=self.config.attention_dropout_p,
            norm_order=TransformerNormOrder.PRE,
        )

    def create_attention(
        self,
        model_dim: Optional[int] = None,
        kv_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
    ) -> MultiheadAttention:
        return StandardMultiheadAttention(
            model_dim=model_dim or self.config.model_dim,
            kv_dim=kv_dim or self.config.model_dim,
            num_heads=num_heads or self.config.num_encoder_attn_heads,
            sdpa=create_default_sdpa(
                bias=IdentityBias(), dropout_p=self.config.attention_dropout_p
            ),
        )

    def create_ffn(
        self, model_dim: Optional[int] = None, inner_dim: Optional[int] = None
    ) -> FeedForwardNetwork:
        return StandardFeedForwardNetwork(
            model_dim=model_dim or self.config.model_dim,
            inner_dim=inner_dim or self.config.ffn_inner_dim,
            bias=True,
            inner_activation=getattr(torch.nn, self.config.activation_fn)(),
            inner_dropout_p=self.config.activation_dropout_p,
        )

    def create_attention_pooler(self) -> EncoderOutputPooler:
        return AttentionEncoderOutputPooler(
            decoder_frontend=self.create_decoder_frontend(),
            decoder=self.create_decoder(),
            projection_out=self.create_projection_out(),
            bos_idx=0,
        )

    def create_layer_norm(self, dim: int) -> LayerNorm:
        return StandardLayerNorm(dim, bias=True)

    # This method, and all methods below, refer only to the attention pooler building.
    # The "decoder" is used for pooling the encoder representations in a smarter way
    def create_decoder_frontend(self) -> TransformerFrontend:
        embed = StandardEmbedding(
            num_embeddings=1,
            embed_dim=self.embedding_dim,
            pad_idx=0,
            init_fn=init_scaled_embedding,
        )
        pos_encoder = SinusoidalPositionEncoder(
            encoding_dim=self.embedding_dim,
            max_seq_len=1,
        )
        return TransformerEmbeddingFrontend(
            model_dim=self.config.model_dim,
            embed=embed,
            pos_encoder=pos_encoder,
            dropout_p=self.config.emb_dropout_p,
        )

    def create_decoder(self) -> TransformerDecoder:
        if self.transformer_normalize_order == TransformerNormOrder.PRE:
            layer_norm = self.create_layer_norm(
                dim=self.config.embedding_dim or self.config.model_dim
            )
        else:
            layer_norm = None

        return StandardTransformerDecoder(
            layers=[
                self.create_decoder_layer()
                for _ in range(self.config.num_decoder_layers)
            ],
            layer_norm=layer_norm,
        )

    def create_decoder_layer(self) -> TransformerDecoderLayer:
        num_heads = self.config.num_decoder_attn_heads

        # TODO: remove self-attention in the pooler, because it does not make sense with a single sequence element
        return StandardTransformerDecoderLayer(
            self_attn=self.create_attention(
                num_heads=num_heads,
                model_dim=self.embedding_dim,
                kv_dim=self.embedding_dim,
            ),
            self_attn_layer_norm=self.create_layer_norm(
                dim=self.config.embedding_dim or self.config.model_dim
            ),
            encoder_decoder_attn=self.create_attention(
                num_heads=num_heads,
                model_dim=self.embedding_dim,
                kv_dim=self.config.model_dim,
            ),
            encoder_decoder_attn_layer_norm=self.create_layer_norm(
                dim=self.config.embedding_dim or self.config.model_dim
            ),
            ffn=self.create_ffn(
                model_dim=self.embedding_dim,
                inner_dim=self.config.decoder_ffn_inner_dim,
            ),
            ffn_layer_norm=self.create_layer_norm(
                dim=self.config.embedding_dim or self.config.model_dim
            ),
            dropout_p=self.config.attention_dropout_p,
            norm_order=self.transformer_normalize_order,
        )

    def create_projection_out(self) -> Linear:
        return Linear(
            input_dim=self.embedding_dim,
            output_dim=self.embedding_dim,
            bias=True,
        )

    def create_projection_in(self) -> Linear:
        return Linear(
            input_dim=self.config.model_dim,
            output_dim=self.embedding_dim,
            bias=True,
        )


def _create_sonar_text_decoder_model(
    config: SonarTextDecoderConfig,
) -> ConditionalTransformerDecoderModel:
    return SonarTextDecoderFactory(config).create_model()


class SonarTextDecoderFactory:
    config: SonarTextDecoderConfig

    def __init__(self, config: SonarTextDecoderConfig) -> None:
        self.config = config

        self.transformer_normalize_order = (
            TransformerNormOrder.PRE
            if self.config.normalize_before
            else TransformerNormOrder.POST
        )

    def create_decoder_frontend(self) -> TransformerFrontend:
        embed = StandardEmbedding(
            num_embeddings=self.config.vocab_info.size,
            embed_dim=self.config.model_dim,
            pad_idx=self.config.vocab_info.pad_idx,
            init_fn=init_scaled_embedding,
        )
        pos_encoder = SinusoidalPositionEncoder(
            encoding_dim=self.config.model_dim,
            max_seq_len=self.config.max_seq_len,
            _legacy_pad_idx=self.config.vocab_info.pad_idx,
        )
        return TransformerEmbeddingFrontend(
            model_dim=self.config.model_dim,
            embed=embed,
            pos_encoder=pos_encoder,
            no_scale=self.config.no_scale_embedding,
            dropout_p=self.config.emb_dropout_p,
        )

    def create_layer_norm(self, dim: int) -> LayerNorm:
        return StandardLayerNorm(dim, bias=True)

    def create_decoder_layer(self) -> TransformerDecoderLayer:
        self_attn = self.create_attention(
            bias=CausalAttentionBias(), kv_dim=self.config.model_dim
        )

        encoder_decoder_attn = self.create_attention(
            bias=IdentityBias(), kv_dim=self.config.input_dim
        )

        ffn = self.create_ffn()

        return StandardTransformerDecoderLayer(
            self_attn=self_attn,
            self_attn_layer_norm=self.create_layer_norm(dim=self.config.model_dim),
            encoder_decoder_attn=encoder_decoder_attn,
            encoder_decoder_attn_layer_norm=self.create_layer_norm(
                dim=self.config.model_dim
            ),
            ffn=ffn,
            ffn_layer_norm=self.create_layer_norm(dim=self.config.model_dim),
            dropout_p=self.config.attention_dropout_p,
            norm_order=TransformerNormOrder.PRE,
        )

    def create_attention(self, bias: AttentionBias, kv_dim=None) -> MultiheadAttention:
        return StandardMultiheadAttention(
            self.config.model_dim,
            self.config.num_encoder_attn_heads,
            kv_dim=kv_dim or self.config.model_dim,
            sdpa=create_default_sdpa(
                bias=bias, dropout_p=self.config.attention_dropout_p
            ),
        )

    def create_ffn(self) -> FeedForwardNetwork:
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            bias=True,
            inner_activation=getattr(torch.nn, self.config.activation_fn)(),
            inner_dropout_p=self.config.activation_dropout_p,
        )

    def create_decoder(self) -> TransformerDecoder:
        return StandardTransformerDecoder(
            layers=[
                self.create_decoder_layer()
                for _ in range(self.config.num_decoder_layers)
            ],
            layer_norm=self.create_layer_norm(
                dim=self.config.model_dim
            ),  # equivalent to TransformerNormOrder.PRE for <fs2:v0.5
        )

    def create_model(self) -> ConditionalTransformerDecoderModel:
        decoder = self.create_decoder()
        decoder_frontend = self.create_decoder_frontend()
        param = cast(Parameter, decoder_frontend.embed.weight)  # type: ignore[union-attr]
        final_proj = TiedProjection(weight=param, bias=None)

        return ConditionalTransformerDecoderModel(
            decoder_frontend=decoder_frontend,
            decoder=decoder,
            final_proj=final_proj,
            max_target_seq_len=self.config.max_seq_len,
        )
