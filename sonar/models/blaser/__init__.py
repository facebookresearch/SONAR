# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sonar.models.blaser.checkpoint import (
    _convert_blaser_checkpoint as _convert_blaser_checkpoint,
)
from sonar.models.blaser.config import BLASER_FAMILY as BLASER_FAMILY
from sonar.models.blaser.config import BlaserConfig as BlaserConfig
from sonar.models.blaser.config import (
    _register_blaser_configs as _register_blaser_configs,
)
from sonar.models.blaser.factory import _create_blaser_model as _create_blaser_model
from sonar.models.blaser.model import BlaserModel as BlaserModel

# isort: split

from fairseq2.models import ModelHubAccessor

get_blaser_model_hub = ModelHubAccessor(BLASER_FAMILY, BlaserModel, BlaserConfig)
