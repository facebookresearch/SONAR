# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from sonar.models.laser2_text.config import Laser2Config
from sonar.nn import LaserLstmEncoder


def _create_laser2_model(config: Laser2Config) -> LaserLstmEncoder:
    return LaserLstmEncoder(
        num_embeddings=config.vocabulary_size,
        padding_idx=config.pad_idx,
        embed_dim=config.model_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        bidirectional=config.bidirectional,
        padding_value=config.padding_value,
    )
