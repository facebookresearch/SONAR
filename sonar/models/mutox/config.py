# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Final

from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer

MUTOX_FAMILY: Final = "mutox_classifier"


@dataclass
class MutoxConfig:
    """Holds the configuration of a Mutox Classifier model."""

    # size of the input embedding supported by this model
    input_size: int


def _register_mutox_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, MutoxConfig)

    @arch("mutox")
    def _base_mutox() -> MutoxConfig:
        return MutoxConfig(
            input_size=1024,
        )
