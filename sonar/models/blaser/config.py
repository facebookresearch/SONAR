# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List

from fairseq2.context import RuntimeContext

from sonar.models.blaser.model import ACTIVATIONS, BLASER_INPUT_FORMS


@dataclass
class BlaserConfig:
    input_form: str = "COMET"
    norm_emb: bool = True
    embedding_dim: int = 1024
    output_dim: int = 1
    hidden_dims: List[int] = field(default_factory=lambda: [3072, 1536])
    dropout: float = 0.1
    activation: str = "TANH"
    output_act: bool = False

    def __post__init__(self):
        """Validate the config"""
        if self.input_form not in BLASER_INPUT_FORMS:
            raise ValueError(
                f"Input form '{self.input_form}' is invalid; should be one of {list(BLASER_INPUT_FORMS)}."
            )
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"Activation '{self.activation}' is invalid; should be one of {list(ACTIVATIONS.keys())}."
            )


def register_blaser_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(BlaserConfig)

    arch = registry.decorator

    @arch("basic_ref")
    def basic_ref() -> BlaserConfig:
        return BlaserConfig(
            embedding_dim=1024,
            output_dim=1,
            norm_emb=True,
            input_form="COMET",
            dropout=0.1,
            hidden_dims=[3072, 1536],
            activation="TANH",
            output_act=False,
        )

    @arch("basic_qe")
    def basic_qe() -> BlaserConfig:
        return BlaserConfig(
            embedding_dim=1024,
            output_dim=1,
            norm_emb=True,
            input_form="QE",
            dropout=0.1,
            hidden_dims=[3072, 1536],
            activation="TANH",
            output_act=False,
        )
