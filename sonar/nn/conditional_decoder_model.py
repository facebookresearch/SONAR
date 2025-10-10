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
ConditionalTransformerDecoderModel inherits from Seq2SeqModel, so it is a sibling class to TransformerModel.

After fs2:v0.5 upgrade:
ConditionalTransformerDecoderModel inherited from EncoderDecoderModel (removed from fs2) and is replaced by Seq2Seq.
This is unconventional as a Seq2Seq model holds both encoder and decoder, while this is only the decoder implementation.
A custom solution might be required (similar to the encoder in sonar/models/encoder_model.py).
"""

from typing import Optional

import torch
from fairseq2.models.seq2seq import Seq2SeqModel
from fairseq2.models.transformer import TransformerDecoder, TransformerFrontend
from fairseq2.nn import BatchLayout, IncrementalStateBag, Projection
from torch import Tensor


class ConditionalTransformerDecoderModel(Seq2SeqModel):
    """Represents a Transformer-based decoder model conditional on the inputs from the encoder."""

    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection
    post_sentemb_proj: Projection | None

    def __init__(
        self,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        max_target_seq_len: int,
        normalize_emb: bool = False,
        post_sentemb_proj: Projection | None = None,
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
        :param normalize_emb:
            Whether to normalize the embedding before passing it to the decoder.
        :param post_sentemb_proj:
            The projection to apply to the sentence embedding.
        """
        super().__init__(max_source_seq_len=0, max_target_seq_len=max_target_seq_len)
        # NOTE: max_source_seq_len = 0 is a workaround due to Seq2Seq requiring *both* an encoder/decoder model and this is the wrong subclass
        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.post_sentemb_proj = post_sentemb_proj
        self.normalize_emb = normalize_emb
        self.final_proj = final_proj

    def encode(self, seqs: Tensor, seqs_layout: BatchLayout) -> Tensor:
        """The encoding just returns the inputs as is."""
        if self.normalize_emb:
            if seqs.dtype != torch.float32:
                original_dtype = seqs.dtype
                norm = torch.norm(seqs.float(), dim=-1, keepdim=True)
                norm = torch.clamp(norm, min=1e-6)
                seqs = seqs / norm.to(original_dtype)
            else:
                seqs = seqs / seqs.norm(dim=-1, keepdim=True)
        return seqs

    def decode(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """Decoding is exactly the same as with fairseq2 TransformerModel"""
        seqs, seqs_layout = self.decoder_frontend(
            seqs, seqs_layout, state_bag=state_bag
        )

        if self.post_sentemb_proj is not None:
            encoder_output = self.post_sentemb_proj(encoder_output)

        return self.decoder(  # type: ignore[no-any-return]
            seqs,
            seqs_layout,
            encoder_output,
            encoder_output_layout,
            state_bag=state_bag,
        )

    def project(self, decoder_output: Tensor) -> Tensor:
        """Projection is exactly the same as with fairseq2 TransformerModel"""
        return self.final_proj(decoder_output)

    def forward(  # type: ignore
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """Reference implementation from fs2:v0.4.3 EncoderDecoderModel using BatchLayout
        The decoder frontend is not used here, c.f. https://github.com/facebookresearch/fairseq2/blob/v0.4.3/src/fairseq2/models/encoder_decoder.py#L42

        Reasoning behind the API change from Seq2SeqBatch to more flat types to help torch.compile to trace
        """
        encoder_output = self.encode(source_seqs, source_seqs_layout)

        decoder_output = self.decode(
            target_seqs,
            target_seqs_layout,
            encoder_output,
            source_seqs_layout,  # encoder does not change padding if it existed and needs to be forwarded
        )

        return self.project(decoder_output)
