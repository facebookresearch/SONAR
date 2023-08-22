# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List, Sequence, Union

import torch
from fairseq2.data import Collater
from fairseq2.data.cstring import CString
from fairseq2.data.data_pipeline import read_sequence
from fairseq2.data.text import TextTokenizer, read_text
from fairseq2.generation import TextTranslator
from fairseq2.models.transformer import TransformerDecoderModel
from fairseq2.typing import Device

from sonar.inference_pipelines.utils import extract_sequence_batch
from sonar.models import SonarEncoderModel, SonarEncoderOutput
from sonar.models.sonar_text import (
    load_sonar_text_decoder_model,
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)
from sonar.models.sonar_translation import SonarEncoderDecoderModel

CPU_DEVICE = torch.device("cpu")


class TextToTextModelPipeline(torch.nn.Module):
    model: SonarEncoderDecoderModel
    tokenizer: TextTokenizer

    def __init__(
        self,
        encoder: Union[str, SonarEncoderModel],
        decoder: Union[str, TransformerDecoderModel],
        tokenizer: Union[str, TextTokenizer],
        device: Device = CPU_DEVICE,
    ) -> None:
        """
        Args:
            encoder (Union[str, SonarEncoderModel]): either cart name or model object
            decoder (Union[str, TransformerDecoderModel]): either cart name or model object
            tokenizer (Union[str, TextTokenizer]): either cart name or tokenizer object
            device (device, optional): . Defaults to CPU_DEVICE.
        """
        super().__init__()
        if isinstance(encoder, str):
            encoder = load_sonar_text_encoder_model(
                encoder, device=device, progress=False
            )
        if isinstance(decoder, str):
            decoder = load_sonar_text_decoder_model(
                decoder, device=device, progress=False
            )
        if isinstance(tokenizer, str):
            tokenizer = load_sonar_tokenizer(tokenizer, progress=False)

        self.tokenizer = tokenizer
        self.model = SonarEncoderDecoderModel(encoder, decoder).to(device).eval()

    @torch.inference_mode()
    def predict(
        self,
        input: Union[Path, Sequence[str]],
        source_lang: str,
        target_lang: str,
        batch_size: int = 5,
    ) -> List[str]:
        translator = TextTranslator(
            model=self.model,
            tokenizer=self.tokenizer,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        pipeline = (
            (
                read_text(input)
                if isinstance(input, (str, Path))
                else read_sequence(input)
            )
            .bucket(batch_size)
            .map(translator)
            .and_return()
        )

        results: List[List[CString]] = list(iter(pipeline))
        return [str(x) for y in results for x in y]


class TextToEmbeddingModelPipeline(torch.nn.Module):
    model: SonarEncoderModel
    tokenizer: TextTokenizer

    def __init__(
        self,
        encoder: Union[str, SonarEncoderModel],
        tokenizer: Union[str, TextTokenizer],
        device: Device = CPU_DEVICE,
    ) -> None:
        """
        Args:
            encoder (Union[str, SonarEncoderModel]): either cart name or model object
            tokenizer (Union[str, TextTokenizer]): either cart name or tokenizer object
            device (device, optional): . Defaults to CPU_DEVICE.
        """
        super().__init__()
        if isinstance(encoder, str):
            encoder = load_sonar_text_encoder_model(
                encoder, device=device, progress=False
            )
        if isinstance(tokenizer, str):
            tokenizer = load_sonar_tokenizer(tokenizer, progress=False)

        self.tokenizer = tokenizer
        self.model = encoder.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def predict(
        self, input: Union[Path, Sequence[str]], source_lang: str, batch_size: int = 5
    ) -> torch.Tensor:
        tokenizer_encoder = self.tokenizer.create_encoder(lang=source_lang)
        pipeline = (
            (
                read_text(input)
                if isinstance(input, (str, Path))
                else read_sequence(input)
            )
            .map(tokenizer_encoder)
            .bucket(batch_size)
            .map(Collater(self.tokenizer.vocab_info.pad_idx))
            .map(lambda x: extract_sequence_batch(x, self.device))
            .map(self.model)
            .and_return()
        )

        results: List[SonarEncoderOutput] = list(iter(pipeline))

        sentence_embeddings = torch.cat([x.sentence_embeddings for x in results], dim=0)
        return sentence_embeddings
