# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import fairseq2

from sonar.models.blaser import get_blaser_model_hub
from sonar.models.blaser.model import BlaserModel


def load_blaser_model(model_name: str) -> BlaserModel:
    """
    This file exists purely for backward compatibility of the package interface!
    Normally, the user is encouraged to call `setup_fairseq2` and `get_blaser_model_hub` on their own.
    """
    fairseq2.setup_fairseq2()
    model_hub = get_blaser_model_hub()
    model = model_hub.load(model_name)
    return model
