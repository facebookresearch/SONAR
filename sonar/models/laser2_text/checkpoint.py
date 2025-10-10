# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast

from sonar.models.laser2_text.config import Laser2Config


def _convert_laser2_checkpoint(
    state_dict: dict[str, object], config: Laser2Config
) -> dict[str, object]:
    # fairseq2 does not use a top-level "model" keyword anymore (v0.5+)
    try:
        state_dict = cast(dict[str, object], state_dict["model"])
    except KeyError:
        pass

    return state_dict
