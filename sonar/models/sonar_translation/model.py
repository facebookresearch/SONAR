# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceBatch, SequenceModelOutput
from fairseq2.models.transformer import TransformerDecoderModel
from fairseq2.nn.padding import PaddingMask
from torch import Tensor

from sonar.models.encoder_model import SonarEncoderModel, SonarEncoderOutput


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
        super().__init__(encoder.model_dim, decoder.vocab_info)
        if encoder.model_dim != decoder.model_dim:
            raise ValueError(
                f"`model_dim` of `encoder` and `model_dim` of `decoder` must be equal, but are {encoder.model_dim} and {decoder.model_dim} instead."
            )
        self.encoder = encoder
        self.decoder = decoder

    def encode(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        batch = SequenceBatch(seqs, padding_mask)
        sonar_output_encoder = self.encoder(batch)
        return (sonar_output_encoder.sentence_embeddings.unsqueeze(1), None)

    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
        state_bag=None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs, padding_mask = self.decoder.decoder_frontend(
            seqs, padding_mask, state_bag=state_bag
        )

        return self.decoder.decoder(  # type: ignore[no-any-return]
            seqs,
            padding_mask,
            encoder_output,
            encoder_padding_mask,
            state_bag=state_bag,
        )

    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[PaddingMask]
    ) -> SequenceModelOutput:
        return self.decoder.project(decoder_output, decoder_padding_mask)


class DummyEncoderModel(SonarEncoderModel):
    """Abstract class for both speech and text SONAR encoder models which does not modify its inputs."""

    def forward(self, batch: SequenceBatch) -> SonarEncoderOutput:
        """
        :param batch:
            The batch of sequences to process.
        :returns:
            SonarEncoderOutput
        """
        return SonarEncoderOutput(
            encoded_seqs=batch.seqs,
            sentence_embeddings=batch.seqs,  # reduce in dim 1
            padding_mask=batch.padding_mask,
        )
