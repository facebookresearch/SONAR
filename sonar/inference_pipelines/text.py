# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union, cast

import torch
from fairseq2.data.data_pipeline import Collater, read_sequence
from fairseq2.data.text import read_text
from fairseq2.data.tokenizers import Tokenizer, load_tokenizer
from fairseq2.data_type import DataType
from fairseq2.device import CPU, Device
from fairseq2.generation import Seq2SeqGenerator
from fairseq2.generation.beam_search.generator import BeamSearchSeq2SeqGenerator
from fairseq2.generation.sampling import Sampler, SamplingSeq2SeqGenerator
from fairseq2.models import load_model
from fairseq2.nn import BatchLayout

from sonar.inference_pipelines.utils import (
    SequenceToTextConverter,
    TextTranslator,
    add_progress_bar,
    extract_sequence_batch,
)
from sonar.models.encoder_model import SonarEncoderModel
from sonar.models.sonar_translation import SonarEncoderDecoderModel
from sonar.models.sonar_translation.model import DummyEncoderModel
from sonar.nn.conditional_decoder_model import ConditionalTransformerDecoderModel


class precision_context:
    dtype_to_precision: Dict[torch.dtype, str] = {
        torch.bfloat16: "medium",
        torch.float16: "medium",
        torch.float32: "high",
        torch.float64: "highest",
    }

    def __init__(self, dtype: torch.dtype):
        self.precision = self.dtype_to_precision.get(dtype, "high")

    def __enter__(self):
        self.original_precision = torch.get_float32_matmul_precision()

        if self.precision:
            torch.set_float32_matmul_precision(self.precision)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_float32_matmul_precision(self.original_precision)


class TextToTextModelPipeline(torch.nn.Module):
    model: SonarEncoderDecoderModel
    tokenizer: Tokenizer

    def __init__(
        self,
        encoder: Union[str, SonarEncoderModel],
        decoder: Union[str, ConditionalTransformerDecoderModel],
        tokenizer: Optional[
            Union[str, Tokenizer]
        ] = None,  # did not remove this to avoid breaking existing code
        encoder_tokenizer: Optional[Union[str, Tokenizer]] = None,
        decoder_tokenizer: Optional[Union[str, Tokenizer]] = None,
        device: Device = CPU,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        Args:
            encoder (Union[str, SonarEncoderModel]): either card name or model object
            decoder (Union[str, ConditionalTransformerDecoderModel]): either card name or model object
            tokenizer (Union[str, Tokenizer], optional): either card name or tokenizer object. Defaults to None.
            encoder_tokenizer (Union[str, Tokenizer], optional): either card name or tokenizer object. Defaults to None.
            decoder_tokenizer (Union[str, Tokenizer], optional): either card name or tokenizer object. Defaults to None.
            device (Device, optional): Defaults to CPU.
            dtype (DataType, optional): The data type of the model parameters and buffers.
        """
        super().__init__()
        if isinstance(encoder, str):
            encoder = load_model(encoder, device=device, dtype=dtype)  # type: ignore
        if isinstance(decoder, str):
            decoder = load_model(decoder, device=device, dtype=dtype)  # type: ignore
        if isinstance(tokenizer, str):
            tokenizer = load_tokenizer(tokenizer)

        assert tokenizer is not None or (
            encoder_tokenizer is not None and decoder_tokenizer is not None
        ), "Either tokenizer or both encoder_tokenizer and decoder_tokenizer must be provided"

        if tokenizer is not None:
            if isinstance(tokenizer, str):
                tokenizer = load_tokenizer(tokenizer)
            self.encoder_tokenizer = tokenizer
            self.decoder_tokenizer = tokenizer
        else:
            if isinstance(encoder_tokenizer, str):
                encoder_tokenizer = load_tokenizer(encoder_tokenizer)
            if isinstance(decoder_tokenizer, str):
                decoder_tokenizer = load_tokenizer(decoder_tokenizer)
            assert (
                encoder_tokenizer is not None and decoder_tokenizer is not None
            ), "we need both encoder_tokenizer and decoder_tokenizer if tokenizer is not provided"  # noqa
            self.encoder_tokenizer = encoder_tokenizer
            self.decoder_tokenizer = decoder_tokenizer

        self.model = SonarEncoderDecoderModel(encoder, decoder).eval()  # type: ignore

    @torch.inference_mode()
    def predict(
        self,
        input: Union[Path, Sequence[str]],
        source_lang: str,
        target_lang: str,
        source_mode: str = "source",
        target_mode: str = "target",
        batch_size: int = 5,
        progress_bar: bool = False,
        **generator_kwargs,
    ) -> List[str]:
        # truncate the max seq len to avoid model to fail
        generator_kwargs = generator_kwargs or {}

        model_max_seq_len = cast(
            int | None,
            (
                self.model.decoder.decoder_frontend.pos_encoder.max_seq_len  # type: ignore
                if self.model.decoder.decoder_frontend.pos_encoder is not None
                else self.model.decoder.decoder.layers[
                    0
                ].self_attn.pos_encoder.max_seq_len
            ),
        )
        if model_max_seq_len is None:
            model_max_seq_len = generator_kwargs.get("max_seq_len", model_max_seq_len)

        generator_kwargs["max_seq_len"] = min(
            model_max_seq_len,
            generator_kwargs.get("max_seq_len", model_max_seq_len),  # type: ignore
        )
        generator = BeamSearchSeq2SeqGenerator(
            self.model, self.decoder_tokenizer.vocab_info, **generator_kwargs
        )
        translator = TextTranslator(
            generator,
            encoder_tokenizer=self.encoder_tokenizer,  # type: ignore
            decoder_tokenizer=self.decoder_tokenizer,  # type: ignore
            source_lang=source_lang,
            target_lang=target_lang,
            source_mode=source_mode,
            target_mode=target_mode,
        )

        def _do_translate(src_texts: List[str]) -> List[str]:
            texts, _ = translator.batch_translate(src_texts)
            return texts

        pipeline: Iterable = (
            (
                read_text(Path(input))
                if isinstance(input, (str, Path))
                else read_sequence(input)
            )
            .bucket(batch_size)
            .map(_do_translate)
            .and_return()
        )
        if progress_bar:
            pipeline = add_progress_bar(pipeline, inputs=input, batch_size=batch_size)

        with precision_context(self.model.dtype):
            results: List[List[str]] = list(iter(pipeline))

        return [x for y in results for x in y]


class TextToEmbeddingModelPipeline(torch.nn.Module):
    model: SonarEncoderModel
    tokenizer: Tokenizer

    def __init__(
        self,
        encoder: Union[str, SonarEncoderModel],
        tokenizer: Union[str, Tokenizer],
        device: Device = CPU,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        Args:
            encoder (Union[str, SonarEncoderModel]): either card name or model object
            tokenizer (Union[str, Tokenizer]): either card name or tokenizer object
            device (device, optional): Defaults to CPU.
            dtype (DataType, optional): The data type of the model parameters and buffers.
        """
        super().__init__()
        if isinstance(encoder, str):
            encoder = load_model(encoder, device=device, dtype=dtype)  # type: ignore
        if isinstance(tokenizer, str):
            tokenizer = load_tokenizer(tokenizer)

        self.tokenizer = tokenizer

        self.model = encoder.eval()  # type: ignore
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def predict(
        self,
        input: Union[Path, Sequence[str]],
        source_lang: str | None = None,
        batch_size: Optional[int] = 5,
        batch_max_tokens: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        progress_bar: bool = False,
        target_device: Optional[Device] = None,
    ) -> torch.Tensor:
        """
        Transform the input texts (from a list of strings or from a text file) into a matrix of their embeddings.
        The texts are truncated to `max_seq_len` tokens,
        or, if it is not specified, to the maximum that the model supports.
        """
        if batch_max_tokens is None and batch_size is None:
            raise ValueError(
                "at least one of `batch_size` or `batch_max_tokens` should be provided"
            )
        if batch_max_tokens is not None and batch_max_tokens <= 0:
            raise ValueError("`batch_max_tokens` should be strictly positive")

        if batch_size is not None and batch_size <= 0:
            raise ValueError("`batch_size` should be strictly positive")

        tokenizer_encoder = self.tokenizer.create_encoder(
            lang=source_lang, device=self.device
        )
        model_max_len = cast(
            int | None,
            (
                self.model.encoder_frontend.pos_encoder.max_seq_len  # type: ignore
                if self.model.encoder_frontend.pos_encoder is not None  # type: ignore
                else self.model.encoder.layers[0].self_attn.pos_encoder.max_seq_len  # type: ignore
            ),
        )
        if max_seq_len is None:
            max_seq_len = model_max_len
        if max_seq_len is not None and model_max_len is not None:
            if max_seq_len > model_max_len:
                raise ValueError(
                    f"max_seq_len cannot be larger than max_seq_len of the encoder model: {model_max_len}"
                )

        n_truncated = 0

        def truncate(x: torch.Tensor) -> torch.Tensor:
            if max_seq_len is None:
                return x
            if x.shape[0] > max_seq_len:
                nonlocal n_truncated
                n_truncated += 1
            return x[:max_seq_len]

        if isinstance(input, (str, Path)):
            pipeline_builder = read_text(Path(input))
            sorting_index = None
        else:
            # so it should a list
            sorting_index = torch.argsort(torch.tensor(list(map(len, input))))
            pipeline_builder = read_sequence(list(sorting_index.cpu())).map(
                input.__getitem__
            )

        pipeline: Iterable = (
            pipeline_builder.map(tokenizer_encoder)
            .map(truncate)
            .dynamic_bucket(
                batch_max_tokens or 2**31,
                len,
                min_num_examples=1,
                max_num_examples=batch_size or 20_000,
                drop_remainder=False,
            )
            .map(Collater(self.tokenizer.vocab_info.pad_idx))
            .map(lambda x: extract_sequence_batch(x, self.device))
            .prefetch(2)
            .map(lambda x: self.model(*x))
            .map(lambda x: x.sentence_embeddings.to(target_device or self.device))
            .and_return()
        )
        if progress_bar:
            pipeline = add_progress_bar(
                pipeline,
                inputs=input,
                batch_size=batch_size if batch_max_tokens is None else None,
            )

        with precision_context(self.model.dtype):
            results: List[torch.Tensor] = list(iter(pipeline))

        if n_truncated:
            warnings.warn(
                f"For {n_truncated} input tensors for SONAR text encoder, "
                f"the length was truncated to {max_seq_len} elements."
            )

        sentence_embeddings = torch.cat(results, dim=0)

        if sorting_index is not None:
            reversed_index = torch.argsort(sorting_index)
            sentence_embeddings = sentence_embeddings[reversed_index]
        return sentence_embeddings


class EmbeddingToTextModelPipeline(torch.nn.Module):
    model: SonarEncoderDecoderModel
    tokenizer: Tokenizer

    def __init__(
        self,
        decoder: Union[str, ConditionalTransformerDecoderModel],
        tokenizer: Union[str, Tokenizer],
        device: Device = CPU,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        Args:
            decoder (Union[str, ConditionalTransformerDecoderModel]): either card name or model object
            tokenizer (Union[str, Tokenizer]): either card name or tokenizer object
            device (device, optional): Defaults to CPU.
            dtype (DataType, optional): The data type of the model parameters and buffers.
        """
        super().__init__()
        if isinstance(decoder, str):
            decoder = load_model(decoder, device=device, dtype=dtype)  # type: ignore
        if isinstance(tokenizer, str):
            tokenizer = load_tokenizer(tokenizer)

        encoder = DummyEncoderModel()  # type: ignore

        self.device = device
        self.tokenizer = tokenizer
        self.model = SonarEncoderDecoderModel(encoder, decoder).eval()  # type: ignore

    @torch.inference_mode()
    def predict(
        self,
        inputs: torch.Tensor,
        target_lang: str,
        target_mode: str = "target",
        batch_size: int = 5,
        progress_bar: bool = False,
        sampler: Optional[Sampler] = None,
        **generator_kwargs,
    ) -> List[str]:
        if sampler is not None:
            generator: Seq2SeqGenerator = SamplingSeq2SeqGenerator(
                self.model, self.tokenizer.vocab_info, sampler, **generator_kwargs
            )
        else:
            generator = BeamSearchSeq2SeqGenerator(
                self.model, self.tokenizer.vocab_info, **generator_kwargs
            )

        converter = SequenceToTextConverter(
            generator,
            self.tokenizer,
            task="translation",
            target_lang=target_lang,
            mode=target_mode,
        )

        def _do_translate(src_tensors: List[torch.Tensor]) -> List[str]:
            seqs = torch.stack(src_tensors).to(self.device)
            seqs_layout = BatchLayout.of(seqs)
            texts, _ = converter.batch_convert(seqs, seqs_layout)
            return texts

        pipeline: Iterable = (
            read_sequence(list(inputs))
            .bucket(batch_size)
            .map(_do_translate)
            .and_return()
        )
        if progress_bar:
            pipeline = add_progress_bar(pipeline, inputs=inputs, batch_size=batch_size)
        with precision_context(self.model.dtype):
            results: List[List[str]] = list(iter(pipeline))

        return [x for y in results for x in y]
