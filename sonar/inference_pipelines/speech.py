# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union, cast

import torch
from fairseq2.data import (
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    FileMapper,
    StringLike,
)
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.data.data_pipeline import read_sequence
from fairseq2.data.text import StrSplitter, TextTokenizer, read_text
from fairseq2.generation import BeamSearchSeq2SeqGenerator, SequenceToTextConverter
from fairseq2.memory import MemoryBlock
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.transformer import TransformerDecoderModel
from fairseq2.typing import DataType, Device

from sonar.inference_pipelines.utils import add_progress_bar, extract_sequence_batch
from sonar.models.encoder_model import SonarEncoderModel
from sonar.models.sonar_speech.loader import load_sonar_speech_model
from sonar.models.sonar_speech.model import SonarSpeechEncoderModel
from sonar.models.sonar_text import load_sonar_text_decoder_model, load_sonar_tokenizer
from sonar.models.sonar_translation.model import SonarEncoderDecoderModel

CPU_DEVICE = torch.device("cpu")


@dataclass
class SpeechInferenceParams:
    data_file: Path
    """The pathname of the test TSV data file."""

    audio_root_dir: Path
    """The pathname of the directory under which audio files are stored."""

    # TODO: This can be computed if we provide column name instead.
    audio_path_index: int
    """Column index of audio path in given TSV data file."""

    batch_size: int
    """The batch size for model input."""

    fbank_dtype: DataType = torch.float32

    target_lang: Optional[str] = None
    """The target translation language."""

    pad_idx: int = 0
    """Padding idx to use after applying fbank"""

    device: Device = CPU_DEVICE
    """The device on which to run inference."""

    # TODO: This will be soon auto-tuned. Right now hand-tuned for devfair.
    n_parallel: int = 4
    """Number of parallel calls when running the pipeline."""

    n_prefetched_batches: int = 4
    """Number of prefetched batches"""


# TODO: Have a common interface with Text and Speech
class SpeechInferencePipeline(ABC):
    @abstractmethod
    def prebuild_pipeline(self, context: SpeechInferenceParams) -> DataPipelineBuilder:
        """Build Data Pipeline for inference and return builder.
        We choose to return builder for an easier aggregation/composition of pipelines.

        :params:
            context: inference configuration

        :returns:
            DataPipelineBuilder
        """

    def build_pipeline(self, context: SpeechInferenceParams) -> DataPipeline:
        return self.prebuild_pipeline(context).and_return()


class AudioToFbankDataPipelineBuilder(SpeechInferencePipeline):
    """Represents an audio to fbank data pipeline"""

    def prebuild_pipeline(self, context: SpeechInferenceParams) -> DataPipelineBuilder:
        # string splitter that splits on '\t' take audio_path_index only and put result
        # in dict at key "audio"
        split_tsv = StrSplitter(names=["audio"], indices=[context.audio_path_index])

        # Start building the pipeline
        pipeline_builder = (
            # Open TSV, skip the header line, split into fields, and return audio field
            read_text(context.data_file, rtrim=True)
            .skip(1)
            .map(split_tsv)
        )

        # Memory map audio files and cache up to 10 files.
        map_file = FileMapper(root_dir=context.audio_root_dir, cached_fd_count=10)

        pipeline_builder.map(
            map_file, selector="audio", num_parallel_calls=context.n_parallel
        )

        # Decode mmap'ed audio using libsndfile and convert them from waveform to fbank.
        decode_audio = AudioDecoder(dtype=context.fbank_dtype)

        convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            device=context.device,
            dtype=context.fbank_dtype,
        )

        pipeline_builder.map(
            [decode_audio, convert_to_fbank],
            # To access FileMapper results
            selector="audio.data",
            num_parallel_calls=context.n_parallel,
        )

        # Batch every `context.batch_size` line
        pipeline_builder.bucket(bucket_size=context.batch_size)

        collate = Collater(pad_value=context.pad_idx, pad_to_multiple=2)

        pipeline_builder.map(collate, num_parallel_calls=context.n_parallel)

        # Prefetch up to `context.n_prefetched_batches`  batches in background.
        pipeline_builder.prefetch(context.n_prefetched_batches)

        # Return the data pipeline builder.
        return pipeline_builder


class SpeechToEmbeddingPipeline(SpeechInferencePipeline):
    """Represents a speech to embedding data pipeline
    Example of usage
    >>> speech_embedding_dp_builder = SpeechToEmbeddingPipeline.load_from_name("sonar_speech_encoder_eng")
    >>> speech_ctx = SpeechInferenceParams(
            data_file=".../test_fleurs_fra-eng.tsv",
            audio_root_dir=".../audio_zips",
            audio_path_index=2,
            target_lang='fra_Latn',
            batch_size=4,
            pad_idx=0,
            device=device,
            fbank_dtype=torch.float32,
            n_parallel=4
        )
    >>> speech_embedding_dp = speech_embedding_dp_builder.build_pipeline(speech_ctx)
    >>> with torch.inference_mode(): speech_emb = next(iter(speech_embedding_dp))
    >>> speech_emb

    """

    audio_to_fbank_dp_builder: AudioToFbankDataPipelineBuilder = (
        AudioToFbankDataPipelineBuilder()
    )
    model: SonarSpeechEncoderModel

    def __init__(self, model: SonarSpeechEncoderModel) -> None:
        self.model = model.eval()

    @classmethod
    def load_model_from_name(cls, encoder_name: str) -> "SpeechToEmbeddingPipeline":
        encoder = load_sonar_speech_model(
            encoder_name, device=CPU_DEVICE, progress=False
        )
        return cls(model=encoder)

    def prebuild_pipeline(self, context: SpeechInferenceParams) -> DataPipelineBuilder:
        self.model = self.model.to(context.device)
        return (
            self.audio_to_fbank_dp_builder.prebuild_pipeline(context)
            .map(
                lambda fbank: extract_sequence_batch(fbank, context.device),
                selector="audio.data.fbank",
            )
            .map(self.run_inference, selector="audio.data")
        )

    @torch.inference_mode()
    def run_inference(self, data: dict) -> dict:
        # TODO assert all(data['sample_rate'] == 16000.0)
        return self.model(data["fbank"])


class SpeechToTextPipeline(SpeechInferencePipeline):
    """Represents a speech to text translation pipeline.

    Example of usage:
    >>> speech_ctx = SpeechInferenceParams(
            data_file=".../test_fleurs_fra-eng.tsv",
            audio_root_dir=".../audio_zips",
            audio_path_index=2,
            target_lang='fra_Latn',
            batch_size=4,
            pad_idx=0,
            device=device,
            fbank_dtype=torch.float32,
            n_parallel=4
        )
    >>> speech_to_text_dp_builder = SpeechToTextPipeline.load_from_name(encoder_name="sonar_speech_encoder_eng",
                                                                        decoder_name="text_sonar_basic_decoder")
    >>> speech_to_text_dp = speech_to_text_dp_builder.build_pipeline(speech_ctx)
    >>> with torch.inference_mode(): speech_text_translation = next(iter(speech_to_text_dp))
    >>> speech_text_translation
    """

    audio_to_fbank_dp_builder: AudioToFbankDataPipelineBuilder = (
        AudioToFbankDataPipelineBuilder()
    )
    model: SonarEncoderDecoderModel
    tokenizer: TextTokenizer

    def __init__(
        self, model: SonarEncoderDecoderModel, tokenizer: TextTokenizer
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer

    @classmethod
    def load_model_from_name(
        cls, encoder_name: str, decoder_name: str
    ) -> "SpeechToTextPipeline":
        tokenizer = load_sonar_tokenizer(decoder_name, progress=False)
        encoder = load_sonar_speech_model(
            encoder_name, device=CPU_DEVICE, progress=False
        )
        decoder = load_sonar_text_decoder_model(
            decoder_name, device=CPU_DEVICE, progress=False
        )
        model = SonarEncoderDecoderModel(encoder, decoder).eval()
        return cls(model=model, tokenizer=tokenizer)

    def prebuild_pipeline(self, context: SpeechInferenceParams) -> DataPipelineBuilder:
        assert context.target_lang is not None
        generator = BeamSearchSeq2SeqGenerator(self.model.to(context.device))
        converter = SequenceToTextConverter(
            generator,
            self.tokenizer,
            task="translation",
            target_lang=context.target_lang,
        )

        def _do_generate(data: dict) -> List[StringLike]:
            batch = cast(SequenceBatch, data["fbank"])
            texts, _ = converter.batch_convert(batch.seqs, batch.padding_mask)
            return texts

        return (
            self.audio_to_fbank_dp_builder.prebuild_pipeline(context)
            .map(
                lambda fbank: extract_sequence_batch(fbank, context.device),
                selector="audio.data.fbank",
            )
            .map(_do_generate, selector="audio.data")
        )


class SpeechModelPipelineInterface(torch.nn.Module):
    device: Device

    def __init__(self, fbank_dtype: DataType) -> None:
        super().__init__()
        self.convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            device=self.device,
            dtype=fbank_dtype,
        )
        self._fbank_dtype = fbank_dtype

    @property
    @lru_cache(maxsize=10)
    def audio_decoder(self):
        return AudioDecoder(dtype=self._fbank_dtype)

    def _decode_audio(self, inp: Union[str, torch.Tensor]) -> dict:
        if isinstance(inp, torch.Tensor):
            return {
                "waveform": inp.transpose(1, 0),
                "sample_rate": 16000.0,
                "format": -1,
            }
        else:
            with Path(str(inp)).open("rb") as fb:
                block = MemoryBlock(fb.read())
            return self.audio_decoder(block)  # type: ignore


class SpeechToTextModelPipeline(SpeechModelPipelineInterface):
    model: SonarEncoderDecoderModel
    tokenizer: TextTokenizer

    def __init__(
        self,
        encoder: Union[str, SonarEncoderModel],
        decoder: Union[str, TransformerDecoderModel],
        tokenizer: Union[str, TextTokenizer],
        device: Device = CPU_DEVICE,
        fbank_dtype: DataType = torch.float32,
    ) -> None:
        """
        Args:
            encoder (Union[str, SonarEncoderModel]): either cart name or model object
            decoder (Union[str, TransformerDecoderModel]): either cart name or model object
            tokenizer (Union[str, TextTokenizer]): either cart name or tokenizer object
            device (device, optional): . Defaults to CPU_DEVICE.
            fbank_dtype (DataType, optional):. Defaults to torch.float32.
        """
        self.device = device
        super().__init__(fbank_dtype)
        if isinstance(encoder, str):
            encoder = load_sonar_speech_model(encoder, device=device, progress=False)
        if isinstance(decoder, str):
            decoder = load_sonar_text_decoder_model(
                decoder, device=device, progress=False
            )
        if isinstance(tokenizer, str):
            tokenizer = load_sonar_tokenizer(tokenizer, progress=False)

        self.tokenizer = tokenizer
        self.model = SonarEncoderDecoderModel(encoder, decoder).to(device).eval()

        # Only quantize the model in CUDA to bypass the error "LayerNormKernelImpl" not implemented for 'Half'
        # in some CUDAs and torch versions
        if fbank_dtype == torch.float16 and device.type == "cuda":
            self.model = self.model.half()

    @torch.inference_mode()
    def predict(
        self,
        input: Union[Sequence[str], Sequence[torch.Tensor]],
        target_lang: str,
        batch_size: int = 3,
        n_parallel: int = 1,
        pad_idx: int = 0,
        n_prefetched_batches: int = 2,
        progress_bar: int = False,
        **generator_kwargs,
    ) -> List[str]:
        generator = BeamSearchSeq2SeqGenerator(
            self.model.to(self.device), **generator_kwargs
        )
        converter = SequenceToTextConverter(
            generator,
            self.tokenizer,
            task="translation",
            target_lang=target_lang,
        )

        def _do_generate(data: dict) -> List[StringLike]:
            batch = cast(SequenceBatch, data["fbank"])
            texts, _ = converter.batch_convert(batch.seqs, batch.padding_mask)
            return texts

        pipeline: Iterable = (
            read_sequence(input)
            .map(self._decode_audio)
            .map(self.convert_to_fbank, num_parallel_calls=n_parallel)
            .bucket(bucket_size=batch_size)
            .map(
                Collater(pad_value=pad_idx, pad_to_multiple=2),
                num_parallel_calls=n_parallel,
            )
            .prefetch(n_prefetched_batches)
            .map(
                lambda fbank: extract_sequence_batch(fbank, self.device),
                selector="fbank",
            )
            .map(_do_generate)
            .and_return()
        )
        if progress_bar:
            pipeline = add_progress_bar(pipeline, inputs=input, batch_size=batch_size)

        results: List[List[StringLike]] = list(iter(pipeline))
        return [str(x) for y in results for x in y]


class SpeechToEmbeddingModelPipeline(SpeechModelPipelineInterface):
    model: SonarEncoderModel
    tokenizer: TextTokenizer

    def __init__(
        self,
        encoder: Union[str, SonarEncoderModel],
        device: Device = CPU_DEVICE,
        fbank_dtype: DataType = torch.float32,
    ) -> None:
        """
        Args:
            encoder (Union[str, SonarEncoderModel]): either cart name or model object
            device (device, optional): . Defaults to CPU_DEVICE.
            fbank_dtype (DataType, optional):. Defaults to torch.float32.
        """
        self.device = device
        super().__init__(fbank_dtype)

        if isinstance(encoder, str):
            encoder = load_sonar_speech_model(encoder, device=device, progress=False)
        self.model = encoder.to(device).eval()

        # Only quantize the model in CUDA to bypass the error "LayerNormKernelImpl" not implemented for 'Half'
        # in some CUDAs and torch versions
        if fbank_dtype == torch.float16 and device.type == "cuda":
            self.model = self.model.half()

    def build_predict_pipeline(
        self,
        input_pipeline,
        batch_size: int = 3,
        n_parallel: int = 1,
        pad_idx: int = 0,
        n_prefetched_batches: int = 2,
    ) -> DataPipelineBuilder:
        pipeline = (
            input_pipeline.map(self._decode_audio)
            .map(self.convert_to_fbank, num_parallel_calls=n_parallel)
            .bucket(bucket_size=batch_size)
            .map(
                Collater(pad_value=pad_idx, pad_to_multiple=2),
                num_parallel_calls=n_parallel,
            )
            .prefetch(n_prefetched_batches)
            .map(
                lambda fbank: extract_sequence_batch(fbank, self.device),
                selector="fbank",
            )
            .map(lambda data: self.model(data["fbank"]).sentence_embeddings)
        )

        return pipeline

    @torch.inference_mode()
    def predict(
        self,
        input: Union[Sequence[str], Sequence[torch.Tensor]],
        batch_size: int = 3,
        n_parallel: int = 1,
        pad_idx: int = 0,
        n_prefetched_batches: int = 2,
        progress_bar: bool = False,
    ) -> torch.Tensor:
        pipeline: Iterable = self.build_predict_pipeline(
            read_sequence(input), batch_size, n_parallel, pad_idx, n_prefetched_batches
        ).and_return()
        if progress_bar:
            pipeline = add_progress_bar(pipeline, inputs=input, batch_size=batch_size)
        results = list(iter(pipeline))
        sentence_embeddings = torch.cat(results, dim=0)
        return sentence_embeddings
