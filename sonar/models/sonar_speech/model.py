# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import TransformerEncoder
from torch import Tensor
from torch.nn import Dropout

from sonar.models.encoder_model import SonarEncoderModel, SonarEncoderOutput
from sonar.nn.encoder_pooler import EncoderOutputPooler


class SonarSpeechEncoderModel(SonarEncoderModel):
    """Represents a SONAR speech encoder model as described in
    # TODO add correct paper cite :cite:t`URL`."""

    encoder_frontend: TransformerFrontend
    encoder: TransformerEncoder
    layer_norm: Optional[LayerNorm]
    final_dropout: Dropout
    encoder_pooler: EncoderOutputPooler

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        layer_norm: Optional[LayerNorm],
        final_dropout_p: float,
        encoder_pooler: EncoderOutputPooler,
    ) -> None:
        """
        :param encoder_frontend:
            The wav2vec2 encoder frontend.
        :param encoder:
            The wav2vec2 encoder model.
        :param layer_norm:
            Optional layer norm applied after wav2vec2 encoder.
        :param final_dropout_p:
            Dropout probability applied at the end of wav2vec2 encoder
        :param encoder_pooler:
            Encoder output pooler.
        """
        super().__init__(encoder.model_dim)

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.final_dropout = Dropout(final_dropout_p)
        self.layer_norm = layer_norm
        self.encoder_pooler = encoder_pooler

    def forward(self, batch: SequenceBatch) -> SonarEncoderOutput:
        seqs, padding_mask = self.encoder_frontend(batch.seqs, batch.padding_mask)
        encoder_output, encoder_padding_mask = self.encoder(seqs, padding_mask)

        # This is the workaround for the pre-LN issue of redundant LayerNorm.
        # We call here, to avoid fiddling with wav2vec2's model and config.
        if self.layer_norm is not None:
            encoder_output = self.layer_norm(encoder_output)

        encoder_output = self.final_dropout(encoder_output)
        encoder_output_pooled = self.encoder_pooler(
            encoder_output, encoder_padding_mask
        )

        return SonarEncoderOutput(
            encoded_seqs=encoder_output,
            sentence_embeddings=encoder_output_pooled,
            padding_mask=padding_mask,
        )

    def encode(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        sonar_output_encoder = self.encoder(seqs, padding_mask)
        return (
            sonar_output_encoder.sentence_embeddings.unsqueeze(1),
            None,
        )
