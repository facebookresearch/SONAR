# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import wavfile

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
    fake_audio = torch.rand(1, 175920)
    audio_file = tmp_path / "audio.wav"
    # Use scipy for WAV file writing (torchaudio I/O removed in 2.9+)
    audio_numpy = fake_audio.squeeze().numpy()
    wavfile.write(str(audio_file), 16000, (audio_numpy * 32767).astype(np.int16))
    embedding = encoder.predict([str(audio_file.resolve())])
    assert embedding.shape == torch.Size([1, 1024])
