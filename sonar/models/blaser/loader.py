# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from fairseq2.models.config_loader import StandardModelConfigLoader
from fairseq2.models.loader import StandardModelLoader, load_model

from sonar.models.blaser.builder import BlaserConfig, blaser_archs, create_blaser_model


def convert_blaser_checkpoint(
    checkpoint: Dict[str, Any], config: BlaserConfig
) -> Dict[str, Any]:
    # Return directly if found fairseq2 attribute in state dict
    if "model" in checkpoint.keys():
        return checkpoint
    # Othewise (the old checkpoint format), move the whole state dict to the "model" section
    return {"model": checkpoint}


load_blaser_config = StandardModelConfigLoader(
    family="blaser", config_kls=BlaserConfig, arch_configs=blaser_archs
)

load_blaser_model = StandardModelLoader(
    config_loader=load_blaser_config,
    factory=create_blaser_model,
    checkpoint_converter=convert_blaser_checkpoint,
    restrict_checkpoints=False,
)

load_model.register("blaser", load_blaser_model)
