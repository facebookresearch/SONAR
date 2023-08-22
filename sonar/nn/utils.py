# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

from torch import Tensor

_neg_inf = float("-inf")


def compute_seq_length(seqs: Tensor, pad_idx: Union[int, float]) -> Tensor:
    """
    Computes sequence_lengths if there's some padding inputs otherwise returns None
    Stop at first pad_idx
    Args:
        seqs (Tensor): The sequences to mask. *Shape:* :math:`(bs, sq, *)`, where :math:`bs` is the
        the batch size, :math:`sq` is the max sequence length.
        pad_idx (int, float): padding index or padding value
    Returns:
        sequence_lengths *Shape:* :math:`(bs, *)`
    >>> import torch
    >>> seqs = torch.Tensor([[2, 3, 1, 2], [2, 3, 2, 2], [2, 1, 2, 1], [1, 2, 3, 1], [1, 1, 1, 1]])
    >>> compute_seq_length(seqs, 1)
    tensor([2, 4, 1, 0, 0])
    """
    return (seqs != pad_idx).int().cumprod(dim=1).sum(dim=1)
