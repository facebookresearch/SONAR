# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod

import torch
from fairseq2.device import Device
from fairseq2.models.transformer import TransformerDecoder, TransformerFrontend
from fairseq2.nn import Linear
from fairseq2.nn.batch_layout import BatchLayout
from torch import Tensor
from torch.nn import Module
from typing_extensions import override


class EncoderOutputPooler(Module):
    """Represents a pooler module to be called on encoder output."""

    @abstractmethod
    def __call__(
        self, encoder_output: Tensor, encoder_output_layout: BatchLayout | None
    ) -> Tensor:
        """Apply pooling on encoder_output

        :param encoder_output:
            The encoder output to use in encoder-decoder attention. *Shape:*
            :math:`(N,S_{enc},M)`, where :math:`N` is the batch size,
            :math:`S_{enc}` is the encoder output sequence length, and :math:`M`
            is the dimensionality of the model.

        :returns:
        The pooler output. *Shape:* :math:`(N,M)`, where :math:`N` is the
        batch size, and :math:`M` is the dimensionality of the model.
        """


class AttentionEncoderOutputPooler(EncoderOutputPooler):
    """Attention pooling applied using decoder architecture"""

    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    projection_out: Linear
    bos_idx: int

    def __init__(
        self,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        projection_out: Linear,
        bos_idx: int,
    ) -> None:
        super().__init__()

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.projection_out = projection_out
        self.bos_idx = bos_idx

    @override
    def __call__(
        self, encoder_output: Tensor, encoder_output_layout: BatchLayout | None
    ) -> Tensor:
        seqs = self._get_pooling_tokens(encoder_output.shape[0], encoder_output.device)
        seqs_layout = BatchLayout.of(seqs)
        if encoder_output_layout is None:
            encoder_output_layout = BatchLayout.of(encoder_output)

        seqs, seqs_layout = self.decoder_frontend(seqs, seqs_layout)

        decoder_out = self.decoder(
            seqs, seqs_layout, encoder_output, encoder_output_layout
        )

        return self.projection_out(decoder_out).squeeze(1)

    def _get_pooling_tokens(self, batch_size: int, device: Device) -> Tensor:
        """TODO Add clear comment on why we need this"""
        return torch.tensor(
            [self.bos_idx] * batch_size, dtype=torch.int64, device=device
        ).unsqueeze(1)
