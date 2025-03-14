# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq2.context import get_runtime_context
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch

from sonar.models.sonar_text import (
    SonarTextDecoderConfig,
    SonarTextDecoderFactory,
    SonarTextEncoderConfig,
    SonarTextEncoderFactory,
)


def test_low_dim_encoder():
    """Test that an encoder with a hidden dimension lower than the embedding dimension can be created and called."""
    context = get_runtime_context()

    config_registry = context.get_config_registry(SonarTextEncoderConfig)

    embed_dim = 256
    batch_size = 3

    cfg = config_registry.get("basic")
    cfg.model_dim = 32
    cfg.embedding_dim = embed_dim
    cfg.num_encoder_layers = 5
    cfg.num_decoder_layers = 2
    cfg.pooling = "attention"
    model = SonarTextEncoderFactory(cfg).create_model()

    tokens = torch.tensor([[0, 1, 2, 3, 4]] * batch_size)
    batch = SequenceBatch(
        seqs=tokens,
        padding_mask=None,
    )
    with torch.inference_mode():
        output = model(batch)
    print(output.sentence_embeddings.shape)
    assert output.sentence_embeddings.shape == (batch_size, embed_dim)


def test_low_dim_decoder():
    """Test that a decoder with a hidden dimension lower than the embedding dimension can be created and called."""
    context = get_runtime_context()

    config_registry = context.get_config_registry(SonarTextDecoderConfig)

    embed_dim = 256
    batch_size = 3

    cfg = config_registry.get("toy")
    cfg.model_dim = 32
    cfg.input_dim = embed_dim
    model = SonarTextDecoderFactory(cfg).create_model()

    embeds = torch.rand([batch_size, 1, embed_dim])
    prefix = torch.tensor([[0, 1, 2, 3, 4]] * batch_size)
    batch = Seq2SeqBatch(
        source_seqs=embeds,
        source_padding_mask=None,
        target_seqs=prefix,
        target_padding_mask=None,
    )
    with torch.inference_mode():
        output = model(batch)

    assert output.logits.shape == (batch_size, 5, cfg.vocab_info.size)
