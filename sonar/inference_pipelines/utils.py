# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union, final

from fairseq2.data.data_pipeline import SequenceData
from fairseq2.data.tokenizers import TokenDecoder, TokenEncoder, Tokenizer
from fairseq2.device import Device
from fairseq2.error import InternalError
from fairseq2.generation import Seq2SeqGenerator, SequenceGeneratorOutput
from fairseq2.nn import BatchLayout
from fairseq2.nn.utils.module import maybe_infer_device
from fairseq2.nn.utils.padding import pad_seqs
from torch import Tensor
from tqdm.auto import tqdm


def extract_sequence_batch(
    x: SequenceData, device: Device
) -> Tuple[Tensor, BatchLayout]:
    """
    Naive conversion from `SequenceData` to `SequenceBatch` without padding or packing.
    Moving `x` to device for backward compatibility of this function definition.

    This was a call to deprecated `get_seqs_and_padding_mask` in fs2:v0.4.6.

    Args:
        x (SequenceData): holding sequences and their lengths
        device (Device): the computing device (cuda, cpu, etc.)
    Returns:
        SequenceBatch: rewrapped `x` and moved to `device`
    """
    seqs, seq_lens = x["seqs"].to(device), x["seq_lens"]
    return seqs, BatchLayout.of(seqs, seq_lens)


def add_progress_bar(
    sequence: Iterable,
    inputs: Optional[Union[Iterable, str, Path]] = None,
    batch_size: Optional[int] = 1,
    **kwargs,
) -> Iterable:
    """
    Wrap the input into a tqdm progress bar.
    Args:
        sequence (Iterable): the sequence to be wrapped
        inputs (Iterable, optional): the sequence to estimate the length of the inputs.
            Ignored if it is a string or Path, because probably it is just the filename.
        batch_size (int, optional): the multiplier to scale the input length. Defaults to 1.
        **kwargs: keyword arguments to pass to tqdm
    """
    total = None
    if inputs is None:
        inputs = sequence
    if batch_size is not None:
        if hasattr(inputs, "__len__") and not isinstance(inputs, (str, Path)):
            total = math.ceil(len(inputs) / batch_size)  # type: ignore

    return tqdm(sequence, total=total, **kwargs)


@final
class SequenceToTextConverter:
    """Converts source sequences to text."""

    # cirquit: This is a carbon copy of fs2:v0.5 with additional `mode`
    # parameter passed to the tokenizer encoder. Should be upstreamed.
    _generator: Seq2SeqGenerator
    _target_prefix_seq: Tensor
    _text_decoder: TokenDecoder

    def __init__(
        self,
        generator: Seq2SeqGenerator,
        tokenizer: Tokenizer,
        task: str,
        target_lang: str | None = None,
        mode: str = "target",
        skip_special_tokens: bool = True,
    ) -> None:
        """
        :param generator:
            The sequence-to-sequence generator.
        :param tokenizer:
            The text tokenizer.
        :param task:
            The conversion task (e.g. translation, transcription).
        :param target_lang:
            The target language for conversion.
        :param mode:
            The mode in which to generate token indices. Typically, translation
            tasks use ``mode`` to distinguish between different modes such as
            'source' or 'target'.
        :param skip_special_tokens:
            Whether the tokenizer decoder skips outputting special tokens like <EOS>.
        """
        self._generator = generator

        try:
            device = maybe_infer_device(generator.model)
        except ValueError as ex:
            raise ValueError(
                "The device of `generator.model` is not valid. See the nested exception for details."
            ) from ex

        target_text_encoder = tokenizer.create_encoder(
            task=task, lang=target_lang, mode=mode, device=device
        )

        # (S)
        target_prefix_seq = target_text_encoder.prefix_indices
        if target_prefix_seq is None:
            raise ValueError(
                "`tokenizer` must specify a prefix sequence for the target language."
            )

        self._target_prefix_seq = target_prefix_seq
        self._text_decoder = tokenizer.create_decoder(
            skip_special_tokens=skip_special_tokens
        )

    def __call__(self, source_seqs: Tensor) -> tuple[str, SequenceGeneratorOutput]:
        """
        :param source_seqs:
            The source sequence. *Shape:* :math:`(S,*)`, where :math:`S` is the
            sequence length and :math:`*` is any number of sequence-specific
            dimensions including none.

        :returns:
            - The converted text.
            - The output of the underlying sequence-to-sequence generator.
        """
        source_seqs = source_seqs.unsqueeze(0)
        source_seqs_layout = BatchLayout.of(source_seqs)
        texts, generator_output = self._do_convert(source_seqs, source_seqs_layout)

        return texts[0], generator_output

    def batch_convert(
        self, source_seqs: Tensor, source_seqs_layout: BatchLayout
    ) -> tuple[list[str], SequenceGeneratorOutput]:
        """
        :param source_seqs:
            The source sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`*` is
            any number of sequence-specific dimensions including none.

        :returns:
            - The converted texts.
            - The output of the underlying sequence-to-sequence generator.
        """
        if len(source_seqs) == 0:
            raise ValueError(
                "`source_seqs` must contain at least one element, but is empty instead."
            )

        return self._do_convert(source_seqs, source_seqs_layout)

    def _do_convert(
        self,
        source_seqs: Tensor,
        source_seqs_layout: BatchLayout,
    ) -> tuple[list[str], SequenceGeneratorOutput]:
        """A subclass should call this method for actual text conversion.

        :param source_seqs:
            The source sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`*` is
            any number of sequence-specific dimensions including none.

        :returns:
            - The converted texts.
            - The output of the underlying sequence-to-sequence generator.
        """
        batch_size = source_seqs.size(0)

        # (S) -> (N, S)
        target_prefix_seqs = self._target_prefix_seq.expand(batch_size, -1)
        target_prefix_layout = BatchLayout.of(target_prefix_seqs)

        generator_output = self._generator(
            source_seqs, source_seqs_layout, target_prefix_seqs, target_prefix_layout
        )

        texts: list[str] = []

        for idx, hypotheses in enumerate(generator_output.hypotheses):
            if len(hypotheses) == 0:
                raise InternalError(
                    f"The sequence generator returned no hypothesis at index {idx}."
                )

            texts.append(self._text_decoder(hypotheses[0].seq))

        return texts, generator_output


@final
class TextTranslator:
    """Translates text from one language to another."""

    # TODO: cirquit - this is a carbon copy of fs2:v0.5 TextTranslator except for
    # - SequenceToTextConverter.skip_special_tokens=True and the source_mode
    # - call to pad_seqs with self._pad_ixd which comes from the encoder_tokenizer.vocab_info

    _converter: SequenceToTextConverter
    _pad_idx: int
    _source_text_encoder: TokenEncoder
    _max_source_len: int | None

    def __init__(
        self,
        generator: Seq2SeqGenerator,
        encoder_tokenizer: Tokenizer,
        decoder_tokenizer: Tokenizer,
        source_lang: str | None = None,
        target_lang: str | None = None,
        source_mode: str = "source",  # this was also added
        target_mode: str = "target",
        *,
        max_source_len: int | None = None,
        skip_special_tokens: bool = True,
    ) -> None:
        """
        :param generator:
            The sequence-to-sequence generator.
        :param tokenizer:
            The text tokenizer.
        :param source_lang:
            The source language.
        :param target_lang:
            The target language.
        :param max_source_len:
            The maximum number of tokens above which the source sequence gets
            truncated.
        :param skip_special_tokens:
            Whether the tokenizer decoder skips outputting special tokens like <EOS>.
        """
        task = "translation"

        self._converter = SequenceToTextConverter(
            generator=generator,
            tokenizer=decoder_tokenizer,
            task=task,
            target_lang=target_lang,
            mode=target_mode,
            skip_special_tokens=skip_special_tokens,
        )

        pad_idx = encoder_tokenizer.vocab_info.pad_idx
        if pad_idx is None:
            raise ValueError(
                "``vocab_info` of `tokenizer` must have a PAD symbol defined."
            )

        self._pad_idx = pad_idx

        try:
            device = maybe_infer_device(generator.model)
        except ValueError as ex:
            raise ValueError(
                "The device of `generator.model` is not valid. See the nested exception for details."
            ) from ex

        self._source_text_encoder = encoder_tokenizer.create_encoder(
            task="translation", lang=source_lang, mode=source_mode, device=device
        )

        if max_source_len is not None and max_source_len <= 0:
            raise ValueError(
                f"`max_source_len` must be greater than or equal to 1, but is {max_source_len} instead."
            )

        self._max_source_len = max_source_len

    def __call__(self, source_text: str) -> tuple[str, SequenceGeneratorOutput]:
        """
        :param source_text:
            The text in the source language.

        :returns:
            - The translated text.
            - The output of the underlying sequence-to-sequence generator.
        """
        source_seq = self._source_text_encoder(source_text)

        if self._max_source_len:
            source_seq = source_seq[: self._max_source_len]

        return self._converter(source_seq)

    def batch_translate(
        self, source_texts: Sequence[str]
    ) -> tuple[list[str], SequenceGeneratorOutput]:
        """
        :param source_texts:
            The texts in the source language.

        :returns:
            - The translated texts.
            - The output of the underlying sequence-to-sequence generator.
        """
        if len(source_texts) == 0:
            raise ValueError(
                "`source_texts` must contain at least one element, but is empty instead."
            )

        source_seq_list = [self._source_text_encoder(t) for t in source_texts]

        if self._max_source_len:
            source_seq_list = [seq[: self._max_source_len] for seq in source_seq_list]

        source_seqs, source_seqs_layout = pad_seqs(
            source_seq_list, pad_value=self._pad_idx
        )

        return self._converter.batch_convert(source_seqs, source_seqs_layout)
