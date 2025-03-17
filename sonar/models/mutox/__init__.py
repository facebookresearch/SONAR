# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sonar.models.mutox.config import MutoxConfig, register_mutox_configs
from sonar.models.mutox.factory import create_mutox_model
from sonar.models.mutox.handler import MutoxModelHandler
from sonar.models.mutox.model import MutoxClassifier

# isort: split

from fairseq2.models import ModelHubAccessor

get_mutox_model_hub = ModelHubAccessor(MutoxClassifier, MutoxConfig)
