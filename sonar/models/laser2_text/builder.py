# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.typing import DataType, Device

from sonar.nn.laser_lstm_encoder import LaserLstmEncoder


@dataclass
class Laser2Config:
    """Holds the configuration of an LSTM model."""

    vocabulary_size: int
    pad_idx: int
    model_dim: int = 320
    hidden_size: int = 512
    num_layers: int = 1
    bidirectional: bool = False
    padding_value: float = 0.0


laser2_archs = ArchitectureRegistry[Laser2Config]("lstm")

laser2_arch = laser2_archs.decorator


@laser2_arch("laser2")
def _laser2() -> Laser2Config:
    return Laser2Config(
        vocabulary_size=50004,
        pad_idx=1,
        model_dim=320,
        hidden_size=512,
        num_layers=5,
        bidirectional=True,
        padding_value=0.0,
    )


class Laser2Builder:
    config: Laser2Config
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: Laser2Config,
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

    def build_model(self) -> LaserLstmEncoder:
        """Build a model."""
        model = LaserLstmEncoder(
            num_embeddings=self.config.vocabulary_size,
            padding_idx=self.config.pad_idx,
            embed_dim=self.config.model_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            bidirectional=self.config.bidirectional,
            padding_value=self.config.padding_value,
        )
        return model.to(device=self.device, dtype=self.dtype)


def create_laser2_model(
    config: Laser2Config,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> LaserLstmEncoder:
    """Create an Laser2 model.
    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return Laser2Builder(config, device=device, dtype=dtype).build_model()
