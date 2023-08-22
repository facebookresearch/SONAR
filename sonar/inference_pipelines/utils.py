# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.models.sequence import SequenceBatch
from fairseq2.typing import Device


def extract_sequence_batch(fbank: dict, device: Device) -> SequenceBatch:
    assert "seqs" in fbank.keys()
    assert "seq_lens" in fbank.keys()

    return SequenceBatch(
        seqs=fbank["seqs"].to(device=device),
        seq_lens=fbank["seq_lens"].to(device=device),
    )
