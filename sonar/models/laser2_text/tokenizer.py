# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional, final

import torch
from fairseq2.data.text.tokenizers import (
    AbstractTextTokenizer,
    TextTokenDecoder,
    TextTokenEncoder,
)
from fairseq2.data.text.tokenizers.sentencepiece import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
    vocab_info_from_sentencepiece,
)
from fairseq2.typing import Device, override
from torch import Tensor
from typing_extensions import NoReturn


@final
class Laser2Encoder(TextTokenEncoder):
    def __init__(self, spm_encoder: SentencePieceEncoder) -> None:
        self.spm_encoder: SentencePieceEncoder = spm_encoder

    @override
    def __call__(self, sentence: str) -> torch.Tensor:
        out = self.spm_encoder(sentence)

        return torch.where(out >= 3, out + 4, out)

    @override
    def encode_as_tokens(self, text: str) -> NoReturn:
        raise RuntimeError("not implemented!")

    @property
    @override
    def prefix_indices(self) -> Optional[Tensor]:
        return self.spm_encoder.prefix_indices

    @property
    @override
    def suffix_indices(self) -> Optional[Tensor]:
        return self.spm_encoder.suffix_indices


@final
class Laser2Tokenizer(AbstractTextTokenizer):
    """Represents the tokenizer used by S2T Transformer models."""

    model: SentencePieceModel

    def __init__(self, path: Path) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        """
        self.model = SentencePieceModel(path, ["<pad>"])

        vocab_info = vocab_info_from_sentencepiece(self.model)

        super().__init__(vocab_info)

    @override
    def create_encoder(
        self,
        *,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> Laser2Encoder:
        return Laser2Encoder(
            spm_encoder=SentencePieceEncoder(
                self.model,
                suffix_tokens=["</s>"],
                device=device,
                pin_memory=pin_memory,
            )
        )

    @override
    def create_raw_encoder(
        self, *, device: Optional[Device] = None, pin_memory: bool = False
    ) -> TextTokenEncoder:
        return SentencePieceEncoder(self.model, device=device, pin_memory=pin_memory)

    @override
    def create_decoder(self) -> TextTokenDecoder:
        return SentencePieceDecoder(self.model)
