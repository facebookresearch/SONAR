# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.data import SequenceData
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import Device


def extract_sequence_batch(x: SequenceData, device: Device) -> SequenceBatch:
    seqs, padding_mask = get_seqs_and_padding_mask(x)

    if padding_mask is not None:
        padding_mask = padding_mask.to(device)

    return SequenceBatch(seqs.to(device), padding_mask)
