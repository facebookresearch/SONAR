# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This class is added to keep functionality identical from fairseq2:v0.4.3. Upgrade at the time fairseq2:v0.5.
The loss calculation has been moved to the forward pass for more fusion potential adding a bit of performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.functional import log_softmax, nll_loss


@dataclass
class SequenceModelOutput:
    """Holds the output of a sequence model."""

    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S,T)`, where
    :math:`N` is the batch size, :math:`S` is the sequence length, and :math:`T`
    is the size of the vocabulary."""

    pad_idx: int | None
    """The index of the PAD symbols in the vocabulary."""

    def compute_loss(
        self,
        targets: Tensor,
        *,
        loss_mask: Tensor | None = None,
        ignore_prefix_size: int = 0,
        label_smoothing: float = 0.0,
    ) -> Tensor:
        """Compute the NLL (negative log-likelihood) loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S)`, where :math:`N` is the
            batch size and :math:`S` is the sequence length.
        :param loss_mask:
            The loss mask that specifies the elements in ``targets`` that should
            be used in the loss computation. All non-masked elements will be
            ignored. *Shape:* Same as ``targets``.
        :param ignore_prefix_size:
            The number of steps from the beginning of the sequence that should
            be ignored in the loss computation.
        :param label_smoothing:
            The amount of label smoothing to apply while computing the loss.

        :returns:
            A scalar tensor representing the summed NLL loss.
        """
        if ignore_prefix_size > 0:
            logits = self.logits[:, ignore_prefix_size:, :]
        else:
            logits = self.logits

        if ignore_prefix_size > 0:
            targets = targets[:, ignore_prefix_size:]

        # For numerical stability run in single precision.
        # (N, S, T)
        lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

        # sum: (), none: (N, S)
        loss = nll_loss(
            input=lprobs,
            target=targets,
            ignore_index=-100 if self.pad_idx is None else self.pad_idx,
            reduction="sum" if loss_mask is None else "none",
        )
        # TODO: support label_smoothing (nll_loss no longer supports it)

        if loss_mask is None:
            return loss

        if ignore_prefix_size > 0:
            loss_mask = loss_mask[:, ignore_prefix_size:]

        # ()
        return (loss * loss_mask).sum()
