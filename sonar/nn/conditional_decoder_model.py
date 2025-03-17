# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
This module defines a conditional decoder model, which serves as a base for a SONAR text decoder
Fairseq2 does not have a suitable model, because:
    - fairseq2.models.transformer.model.TransformerModel imperatively includes a transformer encoder.
    - fairseq2.models.decoder.DecoderModel does not expect any additional inputs.
ConditionalTransformerDecoderModel inherits from EncoderDecoderModel, so it is a sibling class to TransformerModel.
"""

from typing import Optional, Tuple

from fairseq2.data import VocabularyInfo
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn import IncrementalStateBag, Projection
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import TransformerDecoder
from torch import Tensor


class ConditionalTransformerDecoderModel(EncoderDecoderModel):
    """Represents a Transformer-based decoder model conditional on the inputs from the encoder."""

    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection

    def __init__(
        self,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        max_target_seq_len: int,
        target_vocab_info: VocabularyInfo,
    ) -> None:
        """
        :param decoder_frontend:
            The decoder frontend.
        :param decoder:
            The decoder.
        :param final_proj:
            The projection to apply to decoder outputs.
        :param max_target_seq_len:
            The maximum length of sequences produced by the model.
        :param target_vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__(decoder.model_dim, max_target_seq_len, target_vocab_info)

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

    def encode(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """The encoding just returns the inputs as is."""
        return seqs, padding_mask

    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """Decoding is exactly the same as with fairseq2 TransformerModel"""
        seqs, padding_mask = self.decoder_frontend(
            seqs, padding_mask, state_bag=state_bag
        )

        return self.decoder(  # type: ignore[no-any-return]
            seqs,
            padding_mask,
            encoder_output,
            encoder_padding_mask,
            state_bag=state_bag,
        )

    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[PaddingMask]
    ) -> SequenceModelOutput:
        """Projection is exactly the same as with fairseq2 TransformerModel"""
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, pad_idx=self.target_vocab_info.pad_idx)
