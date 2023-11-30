# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.typing import DataType, Device

from sonar.models.blaser.model import ACTIVATIONS, BLASER_INPUT_FORMS, BlaserModel


@dataclass
class BlaserConfig:
    """Holds the configuration of a BLASER model."""

    input_form: str = "COMET"
    norm_emb: bool = True
    embedding_dim: int = 1024
    output_dim: int = 1
    hidden_dims: List = field(default_factory=lambda: [3072, 1536])
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


blaser_archs = ArchitectureRegistry[BlaserConfig]("blaser")

blaser_arch = blaser_archs.decorator


@blaser_arch("basic_ref")
def _arch_blaser_basic_ref() -> BlaserConfig:
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


@blaser_arch("basic_qe")
def _arch_blaser_basic_qe() -> BlaserConfig:
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


class BlaserBuilder:
    config: BlaserConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: BlaserConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config
        self.device = device
        self.dtype = dtype

    def build_model(self) -> BlaserModel:
        """Build a model."""
        model = BlaserModel(**asdict(self.config))
        return model.to(device=self.device, dtype=self.dtype)


def create_blaser_model(
    config: BlaserConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> BlaserModel:
    """Create an Blaser model.
    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return BlaserBuilder(config, device=device, dtype=dtype).build_model()
