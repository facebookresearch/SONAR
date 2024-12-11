# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch.nn
from fairseq2.config_registry import ConfigRegistry
from fairseq2.data import VocabularyInfo
from fairseq2.models.transformer import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
)
from fairseq2.nn import TiedProjection
from fairseq2.nn.embedding import StandardEmbedding, init_scaled_embedding
from fairseq2.nn.normalization import StandardLayerNorm
from fairseq2.nn.position_encoder import (
    LearnedPositionEncoder,
    PositionEncoder,
    SinusoidalPositionEncoder,
)
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.nn.utils.module import to_device
from fairseq2.typing import CPU, DataType, Device

from sonar.models.sonar_text.model import Pooling, SonarTextTransformerEncoderModel
from sonar.nn.conditional_decoder_model import ConditionalTransformerDecoderModel
from sonar.nn.encoder_pooler import AttentionEncoderOutputPooler, EncoderOutputPooler


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


sonar_text_encoder_archs = ConfigRegistry[SonarTextEncoderConfig]()

sonar_text_encoder_arch = sonar_text_encoder_archs.decorator


@sonar_text_encoder_arch("basic")
def encoder_basic() -> SonarTextEncoderConfig:
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


@sonar_text_encoder_arch("small")
def encoder_small(
    vocab_size=32005, depth=6, hidden_dim=1024 * 4
) -> SonarTextEncoderConfig:
    config = encoder_basic()
    config.vocab_info = VocabularyInfo(
        size=vocab_size, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=1
    )
    config.num_encoder_layers = depth
    config.num_decoder_layers = depth
    config.ffn_inner_dim = hidden_dim
    return config


class SonarTextEncoderBuilder:
    config: SonarTextEncoderConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: SonarTextEncoderConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config
        self.device = device
        self.dtype = dtype
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

    def build_model(self) -> SonarTextTransformerEncoderModel:
        """Build a SonarTextTransformerEncoderModel model."""
        embed = StandardEmbedding(
            num_embeddings=self.config.vocab_info.size,
            embedding_dim=self.config.model_dim,
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
            embed,
            pos_encoder,
            no_scale=self.config.no_scale_embedding,
            layer_norm=self.config.layernorm_embedding,
            dropout_p=self.config.emb_dropout_p,
        )

        transformer_layers = [
            self.build_encoder_layer() for _ in range(self.config.num_encoder_layers)
        ]
        encoder = StandardTransformerEncoder(
            transformer_layers, norm_order=self.transformer_normalize_order
        )
        pooling = getattr(Pooling, self.config.pooling.upper())
        if pooling == Pooling.ATTENTION:
            pooler = self.build_attention_pooler()
        else:
            pooler = None

        model = SonarTextTransformerEncoderModel(
            encoder_frontend=embedding_frontend,
            encoder=encoder,
            layer_norm=StandardLayerNorm(self.config.model_dim, bias=True),
            pooling=pooling,
            pooler=pooler,
        )
        # using model.to(device) may accidentaly untie the parameters if the device is META, but to_device is safe
        to_device(model, device=self.device or CPU)
        return model.to(dtype=self.dtype)

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build a Transformer encoder layer."""
        return StandardTransformerEncoderLayer(
            self_attn=self.build_attention(),
            ffn=self.build_ffn(),
            dropout_p=self.config.attention_dropout_p,
            norm_order=TransformerNormOrder.PRE,
        )

    def build_attention(
        self,
        model_dim: Optional[int] = None,
        kv_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
    ) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        return StandardMultiheadAttention(
            model_dim=model_dim or self.config.model_dim,
            kv_dim=kv_dim or self.config.model_dim,
            num_heads=num_heads or self.config.num_encoder_attn_heads,
            sdpa=create_default_sdpa(attn_dropout_p=self.config.attention_dropout_p),
        )

    def build_ffn(
        self, model_dim: Optional[int] = None, inner_dim: Optional[int] = None
    ) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            model_dim=model_dim or self.config.model_dim,
            inner_dim=inner_dim or self.config.ffn_inner_dim,
            bias=True,
            inner_activation=getattr(torch.nn, self.config.activation_fn)(),
            inner_dropout_p=self.config.activation_dropout_p,
            norm_order=self.transformer_normalize_order,
        )

    def build_attention_pooler(self) -> EncoderOutputPooler:
        return AttentionEncoderOutputPooler(
            decoder_frontend=self.build_decoder_frontend(),
            decoder=self.build_decoder(),
            projection_out=self.build_projection_out(),
            bos_idx=0,
        )

    # This method, and all methods below, refer only to the attention pooler building.
    # The "decoder" is used for pooling the encoder representations in a smarter way
    def build_decoder_frontend(self) -> TransformerFrontend:
        """Build a Transformer decoder front-end."""
        embedding = StandardEmbedding(
            num_embeddings=1,
            embedding_dim=self.embedding_dim,
            pad_idx=0,
            init_fn=init_scaled_embedding,
        )
        pos_encoder = SinusoidalPositionEncoder(
            encoding_dim=self.embedding_dim,
            max_seq_len=1,
        )
        return TransformerEmbeddingFrontend(
            embed=embedding,
            pos_encoder=pos_encoder,
            dropout_p=self.config.emb_dropout_p,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self.config.num_decoder_layers
        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return StandardTransformerDecoder(
            layers,
            norm_order=self.transformer_normalize_order,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        num_heads = self.config.num_decoder_attn_heads

        # TODO: remove self-attention in the pooler, because it does not make sense with a single sequence element
        return StandardTransformerDecoderLayer(
            self_attn=self.build_attention(
                num_heads=num_heads,
                model_dim=self.embedding_dim,
                kv_dim=self.embedding_dim,
            ),
            encoder_decoder_attn=self.build_attention(
                num_heads=num_heads,
                model_dim=self.embedding_dim,
                kv_dim=self.config.model_dim,
            ),
            ffn=self.build_ffn(
                model_dim=self.embedding_dim,
                inner_dim=self.config.decoder_ffn_inner_dim,
            ),
            dropout_p=self.config.attention_dropout_p,
            norm_order=self.transformer_normalize_order,
        )

    def build_projection_out(self) -> Linear:
        """Build final projection linear layer for attention pooling"""
        return Linear(
            input_dim=self.embedding_dim,
            output_dim=self.embedding_dim,
            bias=True,
        )

    def build_projection_in(self) -> Linear:
        """Build the input projection linear layer for mean-and-attention pooling"""
        return Linear(
            input_dim=self.config.model_dim,
            output_dim=self.embedding_dim,
            bias=True,
        )


def create_sonar_text_encoder_model(
    config: SonarTextEncoderConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> SonarTextTransformerEncoderModel:
    """Create an SonarTextEncoder model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return SonarTextEncoderBuilder(config, device=device, dtype=dtype).build_model()


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


sonar_text_decoder_archs = ConfigRegistry[SonarTextDecoderConfig]()

sonar_text_decoder_arch = sonar_text_decoder_archs.decorator


@sonar_text_decoder_arch("basic")
def decoder_basic() -> SonarTextDecoderConfig:
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


@sonar_text_decoder_arch("small")
def decoder_small(
    vocab_size=32005, depth=6, hidden_dim=1024 * 4
) -> SonarTextDecoderConfig:
    config = decoder_basic()
    config.vocab_info = VocabularyInfo(
        size=vocab_size, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=1
    )
    config.num_encoder_layers = depth
    config.num_decoder_layers = depth
    config.ffn_inner_dim = hidden_dim
    return config


@sonar_text_decoder_arch("toy")
def decoder_toy() -> SonarTextDecoderConfig:
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


class SonarTextDecoderBuilder:
    config: SonarTextDecoderConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: SonarTextDecoderConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config
        self.device = device
        self.dtype = dtype

        self.transformer_normalize_order = (
            TransformerNormOrder.PRE
            if self.config.normalize_before
            else TransformerNormOrder.POST
        )

    def build_decoder_frontend(self) -> TransformerFrontend:
        """
        decoder frontend is very similar to encoder one
        """
        embed = StandardEmbedding(
            num_embeddings=self.config.vocab_info.size,
            embedding_dim=self.config.model_dim,
            pad_idx=self.config.vocab_info.pad_idx,
            init_fn=init_scaled_embedding,
        )
        pos_encoder = SinusoidalPositionEncoder(
            encoding_dim=self.config.model_dim,
            max_seq_len=self.config.max_seq_len,
            _legacy_pad_idx=self.config.vocab_info.pad_idx,
        )
        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder,
            no_scale=self.config.no_scale_embedding,
            layer_norm=self.config.layernorm_embedding,
            dropout_p=self.config.emb_dropout_p,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_attention(kv_dim=self.config.model_dim)

        encoder_decoder_attn = self.build_attention(kv_dim=self.config.input_dim)

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn,
            ffn,
            dropout_p=self.config.attention_dropout_p,
            norm_order=TransformerNormOrder.PRE,
        )

    def build_attention(self, kv_dim=None) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        return StandardMultiheadAttention(
            self.config.model_dim,
            self.config.num_encoder_attn_heads,
            kv_dim=kv_dim or self.config.model_dim,
            sdpa=create_default_sdpa(attn_dropout_p=self.config.attention_dropout_p),
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            bias=True,
            inner_activation=getattr(torch.nn, self.config.activation_fn)(),
            inner_dropout_p=self.config.activation_dropout_p,
            norm_order=self.transformer_normalize_order,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""

        return StandardTransformerDecoder(
            [self.build_decoder_layer() for _ in range(self.config.num_decoder_layers)],
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_model(self) -> ConditionalTransformerDecoderModel:
        """Build a model."""
        decoder = self.build_decoder()
        decoder_frontend = self.build_decoder_frontend()
        final_proj = TiedProjection(weight=decoder_frontend.embed.weight, bias=None)

        model = ConditionalTransformerDecoderModel(
            decoder_frontend,
            decoder,
            final_proj,
            self.config.max_seq_len,
            self.config.vocab_info,
        )
        # using model.to(device) may accidentaly untie the parameters if the device is META, but to_device is safe
        to_device(model, device=self.device or CPU)
        return model.to(self.dtype)


def create_sonar_text_decoder_model(
    config: SonarTextDecoderConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> ConditionalTransformerDecoderModel:
    """Create an SonarTextDecoder model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return SonarTextDecoderBuilder(config, device=device, dtype=dtype).build_model()
