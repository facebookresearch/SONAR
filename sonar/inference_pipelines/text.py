# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import torch
from fairseq2.data import Collater, StringLike
from fairseq2.data.cstring import CString
from fairseq2.data.data_pipeline import read_sequence
from fairseq2.data.text import TextTokenizer, read_text
from fairseq2.generation import (
    BeamSearchSeq2SeqGenerator,
    SequenceToTextConverter,
    TextTranslator,
)
from fairseq2.models.transformer import TransformerDecoderModel
from fairseq2.typing import Device

from sonar.inference_pipelines.utils import add_progress_bar, extract_sequence_batch
from sonar.models.encoder_model import SonarEncoderModel, SonarEncoderOutput
from sonar.models.sonar_text import (
    load_sonar_text_decoder_model,
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)
from sonar.models.sonar_translation import SonarEncoderDecoderModel
from sonar.models.sonar_translation.model import DummyEncoderModel

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
        progress_bar: bool = False,
        **generator_kwargs,
    ) -> List[str]:
        generator = BeamSearchSeq2SeqGenerator(self.model, **generator_kwargs)
        translator = TextTranslator(
            generator,
            tokenizer=self.tokenizer,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        def _do_translate(src_texts: List[StringLike]) -> List[StringLike]:
            texts, _ = translator.batch_translate(src_texts)
            return texts

        pipeline: Iterable = (
            (
                read_text(input)
                if isinstance(input, (str, Path))
                else read_sequence(input)
            )
            .bucket(batch_size)
            .map(_do_translate)
            .and_return()
        )
        if progress_bar:
            pipeline = add_progress_bar(pipeline, inputs=input, batch_size=batch_size)

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
        self,
        input: Union[Path, Sequence[str]],
        source_lang: str,
        batch_size: int = 5,
        max_seq_len: Optional[int] = None,
        progress_bar: bool = False,
        target_device: Optional[Device] = None,
    ) -> torch.Tensor:
        """
        Transform the input texts (from a list of strings or from a text file) into a matrix of their embeddings.
        The texts are truncated to `max_seq_len` tokens,
        or, if it is not specified, to the maximum that the model supports.
        """
        tokenizer_encoder = self.tokenizer.create_encoder(lang=source_lang)
        model_max_len = self.model.encoder_frontend.pos_encoder.max_seq_len
        if max_seq_len is None:
            max_seq_len = model_max_len
        elif max_seq_len > model_max_len:
            raise ValueError(
                f"max_seq_len cannot be larger than max_seq_len of the encoder model: {model_max_len}"
            )

        n_truncated = 0

        def truncate(x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] > max_seq_len:
                nonlocal n_truncated
                n_truncated += 1
            return x[:max_seq_len]

        pipeline: Iterable = (
            (
                read_text(input)
                if isinstance(input, (str, Path))
                else read_sequence(input)
            )
            .map(tokenizer_encoder)
            .map(truncate)
            .bucket(batch_size)
            .map(Collater(self.tokenizer.vocab_info.pad_idx))
            .map(lambda x: extract_sequence_batch(x, self.device))
            .map(self.model)
            .map(lambda x: x.sentence_embeddings.to(target_device or self.device))
            .and_return()
        )
        if progress_bar:
            pipeline = add_progress_bar(pipeline, inputs=input, batch_size=batch_size)
        results: List[torch.Tensor] = list(iter(pipeline))

        if n_truncated:
            warnings.warn(
                f"For {n_truncated} input tensors for SONAR text encoder, "
                f"the length was truncated to {max_seq_len} elements."
            )

        sentence_embeddings = torch.cat(results, dim=0)
        return sentence_embeddings


class EmbeddingToTextModelPipeline(torch.nn.Module):
    model: SonarEncoderDecoderModel
    tokenizer: TextTokenizer

    def __init__(
        self,
        decoder: Union[str, TransformerDecoderModel],
        tokenizer: Union[str, TextTokenizer],
        device: Device = CPU_DEVICE,
    ) -> None:
        """
        Args:
            decoder (Union[str, TransformerDecoderModel]): either card name or model object
            tokenizer (Union[str, TextTokenizer]): either cart name or tokenizer object
            device (device, optional): . Defaults to CPU_DEVICE.
        """
        super().__init__()
        if isinstance(decoder, str):
            decoder = load_sonar_text_decoder_model(
                decoder, device=device, progress=False
            )
        if isinstance(tokenizer, str):
            tokenizer = load_sonar_tokenizer(tokenizer, progress=False)

        encoder = DummyEncoderModel(decoder.model_dim)

        self.device = device
        self.tokenizer = tokenizer

        self.model = SonarEncoderDecoderModel(encoder, decoder).to(device).eval()

    @torch.inference_mode()
    def predict(
        self,
        inputs: torch.Tensor,
        target_lang: str,
        batch_size: int = 5,
        progress_bar: bool = False,
        **generator_kwargs,
    ) -> List[str]:
        generator = BeamSearchSeq2SeqGenerator(self.model, **generator_kwargs)

        converter = SequenceToTextConverter(
            generator,
            self.tokenizer,
            task="translation",
            target_lang=target_lang,
        )

        def _do_translate(src_tensors: List[torch.Tensor]) -> List[StringLike]:
            texts, _ = converter.batch_convert(
                torch.stack(src_tensors).to(self.device), None
            )
            return texts

        pipeline: Iterable = (
            read_sequence(list(inputs))
            .bucket(batch_size)
            .map(_do_translate)
            .and_return()
        )
        if progress_bar:
            pipeline = add_progress_bar(pipeline, inputs=inputs, batch_size=batch_size)

        results: List[List[CString]] = list(iter(pipeline))
        return [str(x) for y in results for x in y]
