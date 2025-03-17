# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict

from torch import nn

from sonar.models.mutox.config import MutoxConfig
from sonar.models.mutox.model import MutoxClassifier


def create_mutox_model(config: MutoxConfig) -> MutoxClassifier:
    # TODO: refactor the model and the config to make this more flexible
    model_h1 = nn.Sequential(
        nn.Dropout(0.01),
        nn.Linear(config.input_size, 512),
    )

    model_h2 = nn.Sequential(
        nn.ReLU(),
        nn.Linear(512, 128),
    )

    model_h3 = nn.Sequential(
        nn.ReLU(),
        nn.Linear(128, 1),
    )

    model_all = nn.Sequential(
        model_h1,
        model_h2,
        model_h3,
    )
    classifier = MutoxClassifier(model_all)
    return classifier
