# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from pathlib import Path
from typing import Iterable, Optional, Union

from fairseq2.data import SequenceData
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import Device
from tqdm.auto import tqdm


def extract_sequence_batch(x: SequenceData, device: Device) -> SequenceBatch:
    seqs, padding_mask = get_seqs_and_padding_mask(x)

    if padding_mask is not None:
        padding_mask = padding_mask.to(device)

    return SequenceBatch(seqs.to(device), padding_mask)


def add_progress_bar(
    sequence: Iterable,
    inputs: Optional[Union[Iterable, str, Path]] = None,
    batch_size: int = 1,
    **kwargs,
) -> Iterable:
    """
    Wrap the input into a tqdm progress bar.
    Args:
        sequence (Iterable): the sequence to be wrapped
        inputs (Iterable, optional): the sequence to estimate the length of the inputs.
            Ignored if it is a string or Path, because probably it is just the filename.
        batch_size (int, optional): the multiplier to scale the input length. Defaults to 1.
        **kwargs: keyword arguments to pass to tqdm
    """
    total = None
    if inputs is None:
        inputs = sequence
    if hasattr(inputs, "__len__") and not isinstance(inputs, (str, Path)):
        total = math.ceil(len(inputs) / batch_size)  # type: ignore

    return tqdm(sequence, total=total, **kwargs)
