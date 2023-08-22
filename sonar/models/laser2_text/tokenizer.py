# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, final

import torch
from fairseq2.data.text import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
    TextTokenDecoder,
    TextTokenEncoder,
)
from fairseq2.data.text.sentencepiece import vocabulary_from_sentencepiece
from fairseq2.data.typing import PathLike, StringLike
from fairseq2.typing import Device


@final
class Laser2Encoder:
    def __init__(self, spm_encoder: SentencePieceEncoder) -> None:
        super().__init__()
        self.spm_encoder: SentencePieceEncoder = spm_encoder

    def __call__(self, sentence: StringLike) -> torch.Tensor:
        out = self.spm_encoder(sentence)
        return torch.where(out >= 3, out + 4, out)


@final
class Laser2Tokenizer:
    """Represents the tokenizer used by S2T Transformer models."""

    model: SentencePieceModel

    def __init__(
        self,
        pathname: PathLike,
    ) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        """
        self.model = SentencePieceModel(pathname, ["<pad>"])
        self.vocab_info = vocabulary_from_sentencepiece(self.model)

    def create_encoder(
        self, device: Optional[Device] = None, pin_memory: bool = False
    ) -> Laser2Encoder:
        return Laser2Encoder(
            spm_encoder=SentencePieceEncoder(
                self.model,
                suffix_tokens=["</s>"],
                device=device,
                pin_memory=pin_memory,
            )
        )

    def create_decoder(self) -> TextTokenDecoder:
        return SentencePieceDecoder(self.model)
