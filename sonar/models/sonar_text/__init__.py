# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from sonar.models.sonar_text.config import (
    SonarTextDecoderConfig as SonarTextDecoderConfig,
)
from sonar.models.sonar_text.config import (
    SonarTextEncoderConfig as SonarTextEncoderConfig,
)
from sonar.models.sonar_text.config import (
    register_sonar_text_decoder_configs as register_sonar_text_decoder_configs,
)
from sonar.models.sonar_text.config import (
    register_sonar_text_encoder_configs as register_sonar_text_encoder_configs,
)
from sonar.models.sonar_text.factory import (
    SonarTextDecoderFactory as SonarTextDecoderFactory,
)
from sonar.models.sonar_text.factory import (
    SonarTextEncoderFactory as SonarTextEncoderFactory,
)
from sonar.models.sonar_text.handler import (
    SonarTextDecoderHandler as SonarTextDecoderHandler,
)
from sonar.models.sonar_text.handler import (
    SonarTextEncoderHandler as SonarTextEncoderHandler,
)
from sonar.models.sonar_text.model import (
    SonarTextTransformerEncoderModel as SonarTextTransformerEncoderModel,
)

# isort: split

from fairseq2.models import ModelHubAccessor

from sonar.nn.conditional_decoder_model import ConditionalTransformerDecoderModel

get_sonar_text_encoder_hub = ModelHubAccessor(
    SonarTextTransformerEncoderModel, SonarTextEncoderConfig
)


get_sonar_text_decoder_hub = ModelHubAccessor(
    ConditionalTransformerDecoderModel, SonarTextDecoderConfig
)
