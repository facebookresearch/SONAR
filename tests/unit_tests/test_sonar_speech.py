# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest
import torch
import torchaudio  # type: ignore[import]

from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline


@pytest.fixture(scope="module")
def encoder():
    # request.param: whether the encoder is based on a quantized model (fp16) or not
    device = torch.device("cpu")

    return SpeechToEmbeddingModelPipeline(
        encoder="sonar_speech_encoder_eng",
        device=device,
        fbank_dtype=torch.float32,
    )


def test_speech_embedding_with_zeros_input(encoder):
    audio = torch.zeros(1, 175920)
    embedding = encoder.predict([audio])
    assert embedding.shape == torch.Size([1, 1024])


def test_speech_embedding_with_waveform_input(encoder):
    fake_audio = torch.rand(1, 175920)
    embedding = encoder.predict([fake_audio])
    assert embedding.shape == torch.Size([1, 1024])


# Parsing audio within sonar does not support fp16 audio decoding yet
def test_speech_embedding_pipeline_with_audio_files(tmp_path: Path, encoder):
    print(torchaudio.list_audio_backends())
    fake_audio = torch.rand(1, 175920)
    audio_file = tmp_path / "audio.wav"
    torchaudio.save(audio_file, fake_audio, 16000)
    embedding = encoder.predict([str(audio_file.resolve())])
    assert embedding.shape == torch.Size([1, 1024])
