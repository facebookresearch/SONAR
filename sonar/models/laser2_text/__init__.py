# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sonar.models.laser2_text.config import Laser2Config as Laser2Config
from sonar.models.laser2_text.config import (
    register_laser2_configs as register_laser2_configs,
)
from sonar.models.laser2_text.handler import Laser2ModelHandler as Laser2ModelHandler
from sonar.models.laser2_text.handler import (
    Laser2TokenizerHandler as Laser2TokenizerHandler,
)
from sonar.models.laser2_text.tokenizer import Laser2Tokenizer as Laser2Tokenizer

# isort: split

from fairseq2.models import ModelHubAccessor

from sonar.nn.laser_lstm_encoder import LaserLstmEncoder

get_laser2_model_hub = ModelHubAccessor(LaserLstmEncoder, Laser2Config)
