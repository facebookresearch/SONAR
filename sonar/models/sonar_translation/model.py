# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final

from fairseq2.models.seq2seq import Seq2SeqModel
from fairseq2.models.transformer.model import _TransformerModelState
from fairseq2.nn import BatchLayout, IncrementalStateBag
from torch import Tensor

from sonar.models.encoder_model import SonarEncoderModel, SonarEncoderOutput
from sonar.nn.conditional_decoder_model import ConditionalTransformerDecoderModel


@final
class SonarEncoderDecoderModel(Seq2SeqModel):
    """Sonar translation model supporting two distinct usage patterns:

    1. Sequence(Speech/Text)-to-Text (S2T): Real encoder transforms token sequences to sentence embeddings
    2. Embedding-to-Text (E2T): DummyEncoder passes pre-computed embeddings through

    Both patterns must produce identical encoder outputs for consistent decoder behavior.
    The encode() method ensures compatibility by reshaping 2D sentence embeddings
    [batch, embed_dim] to 3D [batch, 1, embed_dim] to match decoder expectations.

    Note (cirquit): This class is subclass of Seq2Seq but does not completely fit its mold, but is the closest we have to the previously
    deprecated EncoderDecoderModel implementation (<fs2:v0.5). This materializes in the strange super() call with the missing
    encoder.max_target_seq_len parameter, because ``SonarEncoderModel`` is defined in SONAR, while ``ConditionalTransformerDecoderModel``
    is a fairseq2 ``Seq2SeqModel``.
    """

    encoder: SonarEncoderModel
    decoder: ConditionalTransformerDecoderModel

    def __init__(
        self,
        encoder: SonarEncoderModel,
        decoder: ConditionalTransformerDecoderModel,
    ) -> None:
        super().__init__(0, decoder.max_target_seq_len)  # see note
        self.encoder = encoder
        self.decoder = decoder

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def encode(self, seqs: Tensor, seqs_layout: BatchLayout) -> Tensor:
        """Convert input sequences to decoder-ready embeddings.

        Transforms variable inputs (tokens or embeddings) to standardized output format:
        - Input: [batch, seq_len] tokens or [batch, embed_dim] embeddings
        - Output: [batch, 1, embed_dim] embeddings with sequence dimension

        The unsqueeze(1) operation creates a sequence dimension that the decoder
        interprets as a single timestep, ensuring consistent behavior across pipelines.
        """
        encoder_output = self.encoder(seqs, seqs_layout)
        return encoder_output.sentence_embeddings.unsqueeze(1)

    def decode(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        encoder_output: Tensor,
        encoder_output_layout: BatchLayout,
        state_bag=None,
    ) -> Tensor:
        return self.decoder.decode(  # type: ignore[no-any-return]
            seqs,
            seqs_layout,
            encoder_output,
            encoder_output_layout,
            state_bag=state_bag,
        )

    def project(self, decoder_output: Tensor) -> Tensor:
        return self.decoder.project(decoder_output)

    # TODO: figure out how typing should work with overload
    def forward(  # type: ignore
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
        target_seqs: Tensor,
        target_seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        # Incremental decoding needs to be handled on a model-level since fs2:v0.5
        if not self.training and state_bag is not None:
            state = state_bag.maybe_get_state(self, _TransformerModelState)
        else:
            state = None

        if state is None:
            encoder_output = self.encode(source_seqs, source_seqs_layout)
            encoder_output_layout = BatchLayout.of(encoder_output)

            if not self.training and state_bag is not None:
                state = _TransformerModelState(encoder_output, encoder_output_layout)

                state_bag.set_state(self, state)
        else:
            encoder_output = state.encoder_output

            encoder_output_layout = state.encoder_output_layout

        del source_seqs

        decoder_output = self.decode(
            target_seqs,
            target_seqs_layout,
            encoder_output,
            encoder_output_layout,
            state_bag=state_bag,
        )

        del target_seqs

        return self.project(decoder_output)


class DummyEncoderModel(SonarEncoderModel):
    """Passthrough encoder enabling architecture reuse for pre-computed embeddings.

    Allows SonarEncoderDecoderModel to handle embedding-to-text generation
    without architectural changes. Returns input embeddings unchanged, relying
    on the parent encode() method for proper shape formatting for `sentence_embeddings`.
    """

    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> SonarEncoderOutput:
        return SonarEncoderOutput(
            encoded_seqs=seqs,
            sentence_embeddings=seqs,  # see SonarEncoderDecoderModel note on the shape dimension expectation
            encoded_seqs_layout=seqs_layout,
        )
