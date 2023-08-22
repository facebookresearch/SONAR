# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
from pathlib import Path

import pytest
import torch

from sonar.models.sonar_speech.loader import load_sonar_speech_model


@pytest.mark.skip(reason="loading all models could take a long time")
def test_load_sonar_speech_model() -> None:
    list_of_files = glob.glob("../../sonar/store/cards/sonar_speech_encoder_*.yaml")
    assert len(list_of_files) == 37
    for file in list_of_files:
        try:
            load_sonar_speech_model(Path(file).stem, device=torch.device("cpu"))
        except Exception as model_loading_exception:
            raise ValueError(
                f"Failed to load model {Path(file).stem}", str(model_loading_exception)
            )
