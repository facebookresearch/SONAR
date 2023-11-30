# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Optional, final

import torch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.transformer.frontend import TransformerFrontend
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask, apply_padding_mask
from fairseq2.nn.transformer import TransformerEncoder
from overrides import final as finaloverride
from torch import Tensor

from sonar.models.encoder_model import SonarEncoderModel, SonarEncoderOutput


class Pooling(Enum):
    MAX = 1
    MEAN = 2
    LAST = 3


@final
class SonarTextTransformerEncoderModel(SonarEncoderModel):
    encoder_frontend: TransformerFrontend
    encoder: TransformerEncoder

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        layer_norm: Optional[LayerNorm] = None,
        pooling: Pooling = Pooling.LAST,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder.
        :param layer_norm:
            optional LayerNorm that is applied on encoder output
        """
        super().__init__(encoder.model_dim)
        if encoder_frontend.model_dim != encoder.model_dim:
            raise ValueError(
                f"`model_dim` of `encoder_frontend` and `model_dim` of `encoder` must be equal, but are {encoder_frontend.model_dim} and {encoder.model_dim} instead."
            )
        if (
            layer_norm is not None
            and layer_norm.normalized_shape[0] != encoder.model_dim
        ):
            raise ValueError(
                f"`model_dim` of `encoder` and `normalized_shape` of `layer_norm` must be equal, but are {encoder_frontend.model_dim} and {layer_norm.normalized_shape} instead."
            )
        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.layer_norm = layer_norm
        self.pooling = pooling

    @staticmethod
    def sentence_embedding_pooling(
        seqs: Tensor, padding_mask: Optional[PaddingMask], pooling: Pooling
    ) -> Tensor:
        """Deterministic pooling along sequence dimension to get a sentence representation
        Args:
            seqs (Tensor): bs x seq_len x model_dim (of float dtype)
            padding_mask (Tensor): bs x seq_len  (containing 0 and -inf)
            pooling (Pooling):

        Returns:
            Tensor: bs x model_dim
        """
        if pooling == Pooling.LAST:
            if padding_mask is None:
                sentence_embedding = seqs[:, -1]
            else:
                seq_lens = padding_mask.seq_lens

                sentence_embedding = seqs[
                    [torch.arange(seq_lens.shape[0]), (seq_lens - 1).clip_(0)]
                ]
        elif pooling == Pooling.MAX:
            seqs = apply_padding_mask(seqs, padding_mask, pad_value=-torch.inf)
            sentence_embedding = seqs.max(dim=1).values
        elif pooling == Pooling.MEAN:
            seqs = apply_padding_mask(seqs, padding_mask, pad_value=0)
            sentence_embedding = seqs.sum(dim=1)
            if padding_mask is None:
                weights = 1.0 / (seqs.size(1) + 1e-7)
                sentence_embedding = sentence_embedding * weights
            else:
                weights = 1.0 / (padding_mask.seq_lens.float() + 1e-7)
                sentence_embedding = torch.einsum(
                    "i...,i->i...", sentence_embedding, weights
                )
        else:
            raise NotImplementedError(pooling)

        return sentence_embedding

    @finaloverride
    def forward(self, batch: SequenceBatch) -> SonarEncoderOutput:
        embed_seqs, padding_mask = self.encoder_frontend(batch.seqs, batch.padding_mask)

        encoded_seqs, _ = self.encoder(embed_seqs, padding_mask)

        if self.layer_norm is not None:
            encoded_seqs = self.layer_norm(encoded_seqs)
        sentence_embeddings = self.sentence_embedding_pooling(
            encoded_seqs, padding_mask, self.pooling
        )
        return SonarEncoderOutput(
            encoded_seqs=encoded_seqs,
            sentence_embeddings=sentence_embeddings,
            padding_mask=padding_mask,
        )
