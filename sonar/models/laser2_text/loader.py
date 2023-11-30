# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.assets import asset_store, download_manager
from fairseq2.models.utils import ConfigLoader, ModelLoader, TokenizerLoader

from sonar.models.laser2_text.builder import (
    Laser2Config,
    create_laser2_model,
    laser2_archs,
)
from sonar.models.laser2_text.tokenizer import Laser2Tokenizer
from sonar.nn.laser_lstm_encoder import LaserLstmEncoder

load_laser2_config = ConfigLoader[Laser2Config](asset_store, laser2_archs)

load_laser2_model = ModelLoader[LaserLstmEncoder, Laser2Config](
    asset_store,
    download_manager,
    load_laser2_config,
    create_laser2_model,
    restrict_checkpoints=False,
)

load_laser2_tokenizer = TokenizerLoader[Laser2Tokenizer](
    asset_store, download_manager, Laser2Tokenizer
)
