# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping

from fairseq2.assets import asset_store, download_manager
from fairseq2.models.utils import ConfigLoader, ModelLoader

from sonar.models.blaser.builder import BlaserConfig, blaser_archs, create_blaser_model
from sonar.models.blaser.model import BlaserModel


def convert_blaser_checkpoint(
    checkpoint: Mapping[str, Any], config: BlaserConfig
) -> Mapping[str, Any]:
    # Return directly if found fairseq2 attribute in state dict
    if "model" in checkpoint.keys():
        return checkpoint
    # Othewise (the old checkpoint format), move the whole state dict to the "model" section
    return {"model": checkpoint}


load_blaser_config = ConfigLoader[BlaserConfig](asset_store, blaser_archs)

load_blaser_model = ModelLoader[BlaserModel, BlaserConfig](
    asset_store,
    download_manager,
    load_blaser_config,
    create_blaser_model,
    convert_blaser_checkpoint,
    restrict_checkpoints=False,
)
