# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
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

    padding_mask: Optional[PaddingMask]
    """Optional, the floating padding mask over sequences (-inf means masked element)
    *Shape:* :math:`(N,S)`, where :math:`N` is the batch size,
    :math:`S` is the sequence length.
    """


class SonarEncoderModel(ABC, Module):
    """Abstract class for both speech and text SONAR encoder models"""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """

        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    @abstractmethod
    def forward(self, batch: SequenceBatch) -> SonarEncoderOutput:
        """
        :param batch:
            The batch of sequences to process.
        :returns:
            SonarEncoderOutput
        """
