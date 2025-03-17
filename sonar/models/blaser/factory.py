# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict

from sonar.models.blaser.config import BlaserConfig
from sonar.models.blaser.model import BlaserModel


def create_blaser_model(config: BlaserConfig) -> BlaserModel:
    return BlaserModel(**asdict(config))
