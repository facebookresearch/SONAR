# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from sonar.models.sonar_speech.checkpoint import (
    _convert_sonar_speech_checkpoint as _convert_sonar_speech_checkpoint,
)
from sonar.models.sonar_speech.config import SONAR_SPEECH_FAMILY as SONAR_SPEECH_FAMILY
from sonar.models.sonar_speech.config import (
    SonarSpeechEncoderConfig as SonarSpeechEncoderConfig,
)
from sonar.models.sonar_speech.config import (
    _register_sonar_speech_encoder_configs as _register_sonar_speech_encoder_configs,
)
from sonar.models.sonar_speech.factory import (
    SonarSpeechEncoderFactory as SonarSpeechEncoderFactory,
)
from sonar.models.sonar_speech.factory import (
    _create_sonar_speech_encoder_model as _create_sonar_speech_encoder_model,
)

# isort: split

from fairseq2.models import ModelHubAccessor

from sonar.models.sonar_speech.model import (
    SonarSpeechEncoderModel as SonarSpeechEncoderModel,
)

get_sonar_speech_encoder_hub = ModelHubAccessor(
    SONAR_SPEECH_FAMILY, SonarSpeechEncoderModel, SonarSpeechEncoderConfig
)
