# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch
import torchaudio  # type: ignore
from torch.testing import assert_close  # type: ignore

from sonar.inference_pipelines.speech import (
    SpeechToEmbeddingModelPipeline,
    SpeechToTextModelPipeline,
)

DATA_DIR = Path(__file__).parent.joinpath("data")
AUDIO_Paths = [
    str(DATA_DIR.joinpath("audio_files/audio_1.wav")),
    str(DATA_DIR.joinpath("audio_files/audio_2.wav")),
]

WAV, sr = torchaudio.load(AUDIO_Paths[0])
assert sr == 16000, "Sample rate should be 16kHz"


def test_speech_to_embedding_model_pipeline():
    s2vec_model = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_eng")
    out2 = s2vec_model.predict(AUDIO_Paths)
    out1 = s2vec_model.predict([WAV])
    assert out1.grad is None
    assert out2.grad is None
    assert_close(out1[0], out2[0], rtol=1e-05, atol=1e-05)
    assert_close(
        out1.numpy().dot(out2.numpy().T),
        torch.Tensor([[0.0429819, 0.00286825]]).numpy(),
        rtol=1e-05,
        atol=1e-05,
    )


def test_speech_to_text_model_pipeline():
    s2t_model = SpeechToTextModelPipeline(
        encoder="sonar_speech_encoder_eng",
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_decoder",
    )

    expected = [
        "Television reports show white smoke coming from the plant.",
        "These couples may choose to make an adoption plan for their baby.",
    ]

    actual = s2t_model.predict([WAV], target_lang="eng_Latn")
    assert expected[0] == actual[0]

    # passing multiple wav files
    actual = s2t_model.predict(AUDIO_Paths, target_lang="eng_Latn")
    assert expected == actual
