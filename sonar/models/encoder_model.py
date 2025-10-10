# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass

from fairseq2.nn.batch_layout import BatchLayout
from torch import Tensor
from torch.nn import Module


@dataclass
class SonarEncoderOutput:
    """Dataclass for both speech and text SONAR encoder outputs"""

    encoded_seqs: Tensor
    """Holds the output of the encoder
    *Shape:* :math:`(N,S,M)`, where :math:`N` is the batch size,
    :math:`S` is the sequence length, and :math:`M` is the
    dimensionality of the model.
    """

    sentence_embeddings: Tensor
    """ Pooled representation, derived from encoded_seqs by pooling in dim=1
    *Shape:* :math:`(N,M)`, where :math:`N` is the batch size, and :math:`M` is the
    dimensionality of the model.
    """

    encoded_seqs_layout: BatchLayout
    """The batchlayout of the ``encoded_seqs``. Holds the information of sequence length,
    optional padding and whether the batch is packed.
    """


class SonarEncoderModel(ABC, Module):
    """Abstract class for both speech and text SONAR encoder models"""

    def __init__(self) -> None:
        super().__init__()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @abstractmethod
    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> SonarEncoderOutput: ...
