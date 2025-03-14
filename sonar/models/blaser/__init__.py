# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sonar.models.blaser.config import BlaserConfig as BlaserConfig
from sonar.models.blaser.config import (
    register_blaser_configs as register_blaser_configs,
)
from sonar.models.blaser.factory import create_blaser_model as create_blaser_model
from sonar.models.blaser.handler import BlaserModelHandler as BlaserModelHandler
from sonar.models.blaser.model import BlaserModel as BlaserModel

# isort: split

from fairseq2.models import ModelHubAccessor

get_blaser_model_hub = ModelHubAccessor(BlaserModel, BlaserConfig)
