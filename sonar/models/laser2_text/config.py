# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Final

from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer

LASER2_FAMILY: Final = "lstm"


@dataclass
class Laser2Config:
    vocabulary_size: int
    pad_idx: int
    model_dim: int = 320
    hidden_size: int = 512
    num_layers: int = 1
    bidirectional: bool = False
    padding_value: float = 0.0


def _register_laser2_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, Laser2Config)

    @arch("laser2")
    def laser2() -> Laser2Config:
        return Laser2Config(
            vocabulary_size=50004,
            pad_idx=1,
            model_dim=320,
            hidden_size=512,
            num_layers=5,
            bidirectional=True,
            padding_value=0.0,
        )
