# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from fairseq2.models.transformer import TransformerEncoder, TransformerFrontend
from fairseq2.nn import BatchLayout, LayerNorm
from torch import Tensor
from torch.nn import Dropout
from typing_extensions import override

from sonar.models.encoder_model import SonarEncoderModel, SonarEncoderOutput
from sonar.nn.encoder_pooler import EncoderOutputPooler


class SonarSpeechEncoderModel(SonarEncoderModel):
    """Represents a SONAR speech encoder model as described in
    :cite:t`https://doi.org/10.48550/arXiv.2308.11466`."""

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
        super().__init__()

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.final_dropout = Dropout(final_dropout_p)
        self.layer_norm = layer_norm
        self.encoder_pooler = encoder_pooler

    @override
    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> SonarEncoderOutput:
        seqs, seqs_layout = self.encoder_frontend(seqs, seqs_layout)
        encoder_output = self.encoder(seqs, seqs_layout)

        # This is the workaround for the pre-LN issue of redundant LayerNorm.
        # We call here, to avoid fiddling with wav2vec2's model and config.
        if self.layer_norm is not None:
            encoder_output = self.layer_norm(encoder_output)

        encoder_output = self.final_dropout(encoder_output)
        encoder_output_pooled = self.encoder_pooler(encoder_output, seqs_layout)

        return SonarEncoderOutput(
            encoded_seqs=encoder_output,
            sentence_embeddings=encoder_output_pooled,
            encoded_seqs_layout=seqs_layout,
        )
