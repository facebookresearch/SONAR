# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Optional, final

import torch
from fairseq2.models.transformer import TransformerEncoder, TransformerFrontend
from fairseq2.nn import LayerNorm
from fairseq2.nn.batch_layout import BatchLayout
from torch import Tensor
from typing_extensions import override

from sonar.models.encoder_model import SonarEncoderModel, SonarEncoderOutput
from sonar.nn.encoder_pooler import EncoderOutputPooler


class Pooling(Enum):
    MAX = 1
    MEAN = 2
    LAST = 3
    ATTENTION = 4


@final
class SonarTextTransformerEncoderModel(SonarEncoderModel):
    encoder_frontend: TransformerFrontend
    encoder: TransformerEncoder

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        max_source_seq_len: int,
        layer_norm: Optional[LayerNorm] = None,
        pooling: Pooling = Pooling.LAST,
        pooler: Optional[EncoderOutputPooler] = None,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder.
        :param max_source_seq_len:
            The maximum sequence length the encoder can ingest.
        :param layer_norm:
            optional LayerNorm that is applied on encoder output
        """
        super().__init__()
        #        if encoder_frontend.model_dim != encoder.model_dim:
        #            raise ValueError(
        #                f"`model_dim` of `encoder_frontend` and `model_dim` of `encoder` must be equal, but are {encoder_frontend.model_dim} and {encoder.model_dim} instead."
        #            )
        #        if (
        #            layer_norm is not None
        #            and layer_norm.normalized_shape[0] != encoder.model_dim
        #        ):
        #            raise ValueError(
        #                f"`model_dim` of `encoder` and `normalized_shape` of `layer_norm` must be equal, but are {encoder_frontend.model_dim} and {layer_norm.normalized_shape} instead."
        #            )
        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.layer_norm = layer_norm
        self.pooling = pooling
        self.pooler = pooler

    def pool(
        self, seqs: Tensor, seqs_layout: BatchLayout | None, pooling: Pooling
    ) -> Tensor:
        """Apply determininstic or trainable pooling"""
        if pooling == Pooling.ATTENTION:
            assert (
                self.pooler is not None
            ), "Cannot use trainable pooling without a pooler in the model"
            sentence_embedding = self.pooler(
                encoder_output=seqs, encoder_output_layout=seqs_layout
            )
        else:
            sentence_embedding = self.static_pooling(
                seqs=seqs, seqs_layout=seqs_layout, pooling=pooling
            )
        return sentence_embedding

    @staticmethod
    def static_pooling(
        seqs: Tensor, seqs_layout: BatchLayout | None, pooling: Pooling
    ) -> Tensor:
        """Deterministic pooling along sequence dimension to get a sentence representation.
        In the future, some SONAR text encoders may have a trainable pooler instead.
        Args:
            seqs (Tensor): bs x seq_len x model_dim (of float dtype)
            padding_mask (Tensor): bs x seq_len  (containing 0 and -inf)
            pooling (Pooling):

        Returns:
            Tensor: bs x model_dim
        """

        if pooling == Pooling.LAST:
            if seqs_layout is None or (seqs_layout and not seqs_layout.padded):
                sentence_embedding = seqs[:, -1]
            else:
                seq_lens = seqs_layout.seq_lens_pt

                sentence_embedding = seqs[
                    [torch.arange(seq_lens.shape[0]), (seq_lens - 1).clip_(0)]
                ]
        elif pooling == Pooling.MAX:
            seqs = SonarTextTransformerEncoderModel.replace_padded_values(
                seqs, seqs_layout, pad_value=-torch.inf
            )
            sentence_embedding = seqs.max(dim=1).values
        elif pooling == Pooling.MEAN:
            seqs = SonarTextTransformerEncoderModel.replace_padded_values(
                seqs, seqs_layout, pad_value=0
            )
            sentence_embedding = seqs.sum(dim=1)
            if seqs_layout is None or not seqs_layout.padded:
                weights = 1.0 / (seqs.size(1) + 1e-7)
                sentence_embedding = sentence_embedding * weights
            else:
                weights = 1.0 / (
                    seqs_layout.seq_lens_pt.to(sentence_embedding.dtype) + 1e-7
                )
                sentence_embedding = torch.einsum(
                    "i...,i->i...", sentence_embedding, weights
                )
        else:
            raise NotImplementedError(pooling)

        return sentence_embedding

    @staticmethod
    def replace_padded_values(
        seqs: Tensor,
        seqs_layout: BatchLayout | None,
        pad_value: int | float | Tensor = 0,
    ) -> Tensor:
        """Replace the padded values in ``seqs`` with `pad_value`.

        :param seqs:
            The sequences to mask. *Shape:* :math:`(N,S,*)` or :math:`(B,*)` for packed,
            where :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`*` is any number of sequence-specific dimensions including none.
        :param seqs_layout:
            The batch layout to apply. If None or not padded, returns seqs unchanged.
        :param pad_value:
            The value for padded positions.

        :returns:
            The input sequences with mask applied. *Shape:* Same as ``seqs``.
        """
        if seqs_layout is None or not seqs_layout.padded:
            return seqs

        # True for valid positions, False for padding
        mask = seqs_layout.position_indices >= 0

        # Handle broadcasting for higher-dimensional tensors
        for _ in range(seqs.ndim - mask.ndim):
            mask = mask.unsqueeze(-1)

        return seqs.where(mask, pad_value)

    @override
    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> SonarEncoderOutput:
        embed_seqs, embed_seqs_layout = self.encoder_frontend(seqs, seqs_layout)

        encoded_seqs = self.encoder(embed_seqs, embed_seqs_layout)
        # encoded_seqs_layout = BatchLayout.of(encoded_seqs)

        if self.layer_norm is not None:
            encoded_seqs = self.layer_norm(encoded_seqs)
        sentence_embeddings = self.pool(encoded_seqs, embed_seqs_layout, self.pooling)
        return SonarEncoderOutput(
            encoded_seqs=encoded_seqs,
            sentence_embeddings=sentence_embeddings,
            encoded_seqs_layout=embed_seqs_layout,
        )
