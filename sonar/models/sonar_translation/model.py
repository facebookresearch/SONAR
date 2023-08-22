# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceBatch, SequenceModelOutput
from fairseq2.models.transformer import TransformerDecoderModel
from torch import Tensor

from sonar.models import SonarEncoderModel


@final
class SonarEncoderDecoderModel(EncoderDecoderModel):
    """Sonar translation model.

    This is a generic model that can be used for speech any combination of speech,text
    translation by combining Speech/Text Encoder/Decoder components.
    """

    encoder: SonarEncoderModel
    decoder: TransformerDecoderModel

    def __init__(
        self,
        encoder: SonarEncoderModel,
        decoder: TransformerDecoderModel,
    ) -> None:
        super().__init__(model_dim=encoder.model_dim)
        if encoder.model_dim != decoder.model_dim:
            raise ValueError(
                f"`model_dim` of `encoder` and `model_dim` of `decoder` must be equal, but are {encoder.model_dim} and {decoder.model_dim} instead."
            )
        self.encoder = encoder
        self.decoder = decoder

    def encode(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        batch = SequenceBatch(seqs, seq_lens)
        sonar_output_encoder = self.encoder(batch)
        return (sonar_output_encoder.sentence_embeddings.unsqueeze(1), None)

    def decode(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor],
        state_bag=None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.decoder.decoder_frontend(seqs, seq_lens, state_bag)

        return self.decoder.decoder(  # type: ignore[no-any-return]
            seqs, padding_mask, encoder_output, encoder_padding_mask, state_bag
        )

    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[Tensor]
    ) -> SequenceModelOutput:
        return self.decoder.project(decoder_output, decoder_padding_mask)
