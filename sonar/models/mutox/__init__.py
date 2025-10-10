# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sonar.models.mutox.checkpoint import (
    _convert_mutox_checkpoint as _convert_mutox_checkpoint,
)
from sonar.models.mutox.config import MUTOX_FAMILY as MUTOX_FAMILY
from sonar.models.mutox.config import MutoxConfig as MutoxConfig
from sonar.models.mutox.config import _register_mutox_configs as _register_mutox_configs
from sonar.models.mutox.factory import _create_mutox_model as _create_mutox_model

# isort: split

from fairseq2.models import ModelHubAccessor

from sonar.models.mutox.model import MutoxClassifier as MutoxClassifier

get_mutox_model_hub = ModelHubAccessor(MUTOX_FAMILY, MutoxClassifier, MutoxConfig)
