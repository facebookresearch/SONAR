# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from sonar.models.sonar_speech.config import (
    SonarSpeechEncoderConfig as SonarSpeechEncoderConfig,
)
from sonar.models.sonar_speech.config import (
    register_sonar_speech_encoder_configs as register_sonar_speech_encoder_configs,
)
from sonar.models.sonar_speech.factory import (
    SonarSpeechEncoderFactory as SonarSpeechEncoderFactory,
)
from sonar.models.sonar_speech.handler import (
    SonarSpeechEncoderHandler as SonarSpeechEncoderHandler,
)
from sonar.models.sonar_speech.model import (
    SonarSpeechEncoderModel as SonarSpeechEncoderModel,
)

# isort: split

from fairseq2.models import ModelHubAccessor

get_sonar_speech_encoder_hub = ModelHubAccessor(
    SonarSpeechEncoderModel, SonarSpeechEncoderConfig
)
