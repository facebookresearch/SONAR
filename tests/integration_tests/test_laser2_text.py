# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import torch
from fairseq2.data import Collater, text
from torch.testing import assert_close

from sonar.models.laser2_text.loader import load_laser2_model, load_laser2_tokenizer

device = torch.device("cpu")

sentences = [
    "to be or not to be",
    "être ou ne pas être",
    "i want to go biking",
    "je veux faire du vélo",
]


def test_load_laser2_text() -> None:
    model = load_laser2_model(
        "laser2_text_encoder", device=device, progress=False
    ).eval()
    tokenizer = load_laser2_tokenizer("laser2_text_encoder", progress=False)
    encoder = tokenizer.create_encoder()

    with tempfile.NamedTemporaryFile(mode="w+t", newline="\n") as tmp:
        tmp.writelines("\n".join(sentences) + "\n")
        tmp.seek(0)
        pipeline = (
            text.read_text(tmp.name, rtrim=True, ltrim=True, memory_map=True)
            .map(encoder)
            .bucket(len(sentences), drop_remainder=True)
            .map(Collater(pad_idx=1))
            .and_return()
        )
        tokenized_sentences = next(iter(pipeline))

    embed_sentences = model(
        tokenized_sentences["seqs"], tokenized_sentences["seq_lens"]
    )
    embed_sentences_norm = torch.nn.functional.normalize(embed_sentences)
    actual_sim = torch.matmul(embed_sentences_norm, embed_sentences_norm.T)

    # copied from fairseq1 results
    expected_sim = torch.tensor(
        [
            [1.0000, 0.9614, 0.4412, 0.3923],
            [0.9614, 1.0000, 0.4110, 0.3935],
            [0.4412, 0.4110, 1.0000, 0.6960],
            [0.3923, 0.3935, 0.6960, 1.0000],
        ]
    )

    assert_close(actual_sim, expected_sim, rtol=1e-4, atol=1e-4)
