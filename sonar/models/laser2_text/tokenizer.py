# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Optional, final

import torch
from fairseq2.data.tokenizers import (
    TokenDecoder,
    TokenEncoder,
    Tokenizer,
    TokenizerModelError,
    VocabularyInfo,
)
from fairseq2.data.tokenizers.sentencepiece import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
    get_sentencepiece_vocabulary_info,
)
from fairseq2.device import Device
from fairseq2.error import OperationalError
from torch import Tensor
from typing_extensions import NoReturn


@final
class Laser2Encoder(TokenEncoder):
    def __init__(self, spm_encoder: SentencePieceEncoder) -> None:
        self.spm_encoder: SentencePieceEncoder = spm_encoder

    def __call__(self, sentence: str) -> torch.Tensor:
        out = self.spm_encoder(sentence)

        return torch.where(out >= 3, out + 4, out)

    def encode_as_tokens(self, text: str) -> NoReturn:
        raise RuntimeError("not implemented!")

    @property
    def prefix_indices(self) -> Optional[Tensor]:
        return self.spm_encoder.prefix_indices

    @property
    def suffix_indices(self) -> Optional[Tensor]:
        return self.spm_encoder.suffix_indices


@final
class Laser2Tokenizer(Tokenizer):
    """Represents the tokenizer used by S2T Transformer models."""

    model: SentencePieceModel
    _vocab_info: VocabularyInfo
    # breaking styleguide to implement the vocab_info abstract property interface

    def __init__(self, path: Path) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        """
        self.model = SentencePieceModel(path, ["<pad>"])
        self._vocab_info = get_sentencepiece_vocabulary_info(self.model)

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

    def create_raw_encoder(
        self, *, device: Optional[Device] = None, pin_memory: bool = False
    ) -> TokenEncoder:
        return SentencePieceEncoder(self.model, device=device, pin_memory=pin_memory)

    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        return SentencePieceDecoder(self.model)

    @property
    def vocab_info(self) -> VocabularyInfo:
        return self._vocab_info


def _load_laser2_tokenizer(path: Path, config: None = None) -> Tokenizer:
    try:
        model = Laser2Tokenizer(path)
    except OSError as ex:
        raise OperationalError(
            f"A system error has occurred while reading the '{path}' tokenizer model. See the nested exception for details."
        ) from ex
    except RuntimeError as ex:
        raise TokenizerModelError(
            path,
            f"The '{path}' tokenizer model cannot be loaded. See the nested exception for details.",  # fmt: skip
        ) from ex

    return model
