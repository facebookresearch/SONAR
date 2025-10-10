# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast

from sonar.models.mutox.config import MutoxConfig


def _convert_mutox_checkpoint(
    state_dict: dict[str, object], config: MutoxConfig
) -> dict[str, object]:
    # fairseq2 does not use a top-level "model" keyword anymore (v0.5+)
    try:
        state_dict = cast(dict[str, object], state_dict["model"])
    except KeyError:
        pass

    new_dict = {}
    for key in state_dict:
        if key.startswith("model_all."):
            new_dict[key] = state_dict[key]
    return new_dict
