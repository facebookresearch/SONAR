# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping, final

from fairseq2.assets import download_manager
from fairseq2.models.utils.model_loader import ModelLoader

from sonar.models.blaser.builder import BlaserConfig, blaser_archs, create_blaser_model
from sonar.models.blaser.model import BlaserModel
from sonar.store import asset_store


@final
class BlaserLoader(ModelLoader[BlaserModel, BlaserConfig]):
    """Loads Blaser models"""

    def _upgrade_checkpoint(
        self, checkpoint: Mapping[str, Any], config: BlaserConfig
    ) -> Mapping[str, Any]:
        # Return directly if found fairseq2 attribute in state dict
        if "model" in checkpoint.keys():
            return checkpoint
        # Othewise (the old checkpoint format), move the whole state dict to the "model" section
        return {"model": checkpoint}


load_blaser_model = BlaserLoader(
    asset_store,
    download_manager,
    create_blaser_model,
    blaser_archs,
)
