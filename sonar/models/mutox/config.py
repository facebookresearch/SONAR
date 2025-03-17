# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from fairseq2.context import RuntimeContext


@dataclass
class MutoxConfig:
    """Holds the configuration of a Mutox Classifier model."""

    # size of the input embedding supported by this model
    input_size: int


def register_mutox_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(MutoxConfig)

    arch = registry.decorator

    @arch("mutox")
    def _base_mutox() -> MutoxConfig:
        return MutoxConfig(
            input_size=1024,
        )
