# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.models import ModelHubAccessor

from sonar.models.sonar_text.checkpoint import (
    _convert_sonar_text_decoder_checkpoint as _convert_sonar_text_decoder_checkpoint,
)
from sonar.models.sonar_text.checkpoint import (
    _convert_sonar_text_encoder_checkpoint as _convert_sonar_text_encoder_checkpoint,
)
from sonar.models.sonar_text.config import (
    SONAR_TEXT_DECODER_FAMILY as SONAR_TEXT_DECODER_FAMILY,
)
from sonar.models.sonar_text.config import (
    SONAR_TEXT_ENCODER_FAMILY as SONAR_TEXT_ENCODER_FAMILY,
)
from sonar.models.sonar_text.config import (
    SonarTextDecoderConfig as SonarTextDecoderConfig,
)
from sonar.models.sonar_text.config import (
    SonarTextEncoderConfig as SonarTextEncoderConfig,
)
from sonar.models.sonar_text.config import (
    _register_sonar_text_decoder_configs as _register_sonar_text_decoder_configs,
)
from sonar.models.sonar_text.config import (
    _register_sonar_text_encoder_configs as _register_sonar_text_encoder_configs,
)
from sonar.models.sonar_text.factory import (
    SonarTextDecoderFactory as SonarTextDecoderFactory,
)
from sonar.models.sonar_text.factory import (
    SonarTextEncoderFactory as SonarTextEncoderFactory,
)
from sonar.models.sonar_text.factory import (
    _create_sonar_text_decoder_model as _create_sonar_text_decoder_model,
)
from sonar.models.sonar_text.factory import (
    _create_sonar_text_encoder_model as _create_sonar_text_encoder_model,
)
from sonar.models.sonar_text.model import (
    SonarTextTransformerEncoderModel as SonarTextTransformerEncoderModel,
)
from sonar.nn.conditional_decoder_model import ConditionalTransformerDecoderModel

get_sonar_text_encoder_hub = ModelHubAccessor(
    SONAR_TEXT_ENCODER_FAMILY, SonarTextTransformerEncoderModel, SonarTextEncoderConfig
)

get_sonar_text_decoder_hub = ModelHubAccessor(
    SONAR_TEXT_DECODER_FAMILY,
    ConditionalTransformerDecoderModel,
    SonarTextDecoderConfig,
)
