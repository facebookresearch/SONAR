# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq2.nn import BatchLayout
from fairseq2.runtime.config_registry import get_config
from fairseq2.runtime.dependency import get_dependency_resolver

from sonar.models.sonar_text import (
    SonarTextDecoderConfig,
    SonarTextDecoderFactory,
    SonarTextEncoderConfig,
    SonarTextEncoderFactory,
)


def test_low_dim_encoder():
    """Test that an encoder with a hidden dimension lower than the embedding dimension can be created and called."""
    resolver = get_dependency_resolver()
    cfg = get_config(resolver, SonarTextEncoderConfig, "basic")

    embed_dim = 256
    batch_size = 3

    cfg.model_dim = 32
    cfg.embedding_dim = embed_dim
    cfg.num_encoder_layers = 5
    cfg.num_decoder_layers = 2
    cfg.pooling = "attention"
    model = SonarTextEncoderFactory(cfg).create_model()

    tokens = torch.tensor([[0, 1, 2, 3, 4]] * batch_size)
    tokens_layout = BatchLayout.of(tokens)
    with torch.inference_mode():
        output = model(tokens, tokens_layout)
    print(output.sentence_embeddings.shape)
    assert output.sentence_embeddings.shape == (batch_size, embed_dim)


def test_low_dim_decoder():
    """Test that a decoder with a hidden dimension lower than the embedding dimension can be created and called."""
    resolver = get_dependency_resolver()

    cfg = get_config(resolver, SonarTextDecoderConfig, "toy")

    embed_dim = 256
    batch_size = 3

    cfg.model_dim = 32
    cfg.input_dim = embed_dim
    model = SonarTextDecoderFactory(cfg).create_model()

    embeds = torch.rand([batch_size, 1, embed_dim])
    prefix = torch.tensor([[0, 1, 2, 3, 4]] * batch_size)
    with torch.inference_mode():
        output = model(
            source_seqs=embeds,
            source_seqs_layout=BatchLayout.of(embeds),
            target_seqs=prefix,
            target_seqs_layout=BatchLayout.of(prefix),
        )

    assert output.shape == (batch_size, 5, cfg.vocab_info.size)
