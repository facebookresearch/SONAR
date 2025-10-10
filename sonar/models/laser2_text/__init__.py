# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from types import NoneType

from sonar.models.laser2_text.checkpoint import (
    _convert_laser2_checkpoint as _convert_laser2_checkpoint,
)
from sonar.models.laser2_text.config import LASER2_FAMILY as LASER2_FAMILY
from sonar.models.laser2_text.config import Laser2Config as Laser2Config
from sonar.models.laser2_text.config import (
    _register_laser2_configs as _register_laser2_configs,
)
from sonar.models.laser2_text.factory import (
    _create_laser2_model as _create_laser2_model,
)
from sonar.models.laser2_text.tokenizer import Laser2Tokenizer as Laser2Tokenizer
from sonar.models.laser2_text.tokenizer import (
    _load_laser2_tokenizer as _load_laser2_tokenizer,
)

# isort: split

from fairseq2.data.tokenizers import TokenizerHubAccessor
from fairseq2.models import ModelHubAccessor

from sonar.nn.laser_lstm_encoder import LaserLstmEncoder

get_laser2_model_hub = ModelHubAccessor(LASER2_FAMILY, LaserLstmEncoder, Laser2Config)

get_laser2_tokenizer_hub = TokenizerHubAccessor(
    LASER2_FAMILY, Laser2Tokenizer, NoneType
)
