# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from fairseq2.models.config_loader import StandardModelConfigLoader
from fairseq2.models.loader import StandardModelLoader, load_model

from .builder import create_mutox_model
from .classifier import MutoxConfig, mutox_archs

__import__("sonar")  # Import only to update asset_store


@mutox_archs.decorator("mutox")
def _base_mutox() -> MutoxConfig:
    return MutoxConfig(
        input_size=1024,
    )


def convert_mutox_checkpoint(
    checkpoint: Dict[str, Any], config: MutoxConfig
) -> Dict[str, Any]:
    new_dict = {}
    for key in checkpoint:
        if key.startswith("model_all."):
            new_dict[key] = checkpoint[key]
    return {"model": new_dict}


load_mutox_config = StandardModelConfigLoader(
    family="mutox", config_kls=MutoxConfig, arch_configs=mutox_archs
)

load_mutox_model = StandardModelLoader(
    config_loader=load_mutox_config,
    factory=create_mutox_model,
    checkpoint_converter=convert_mutox_checkpoint,
    restrict_checkpoints=False,
)

load_model.register("mutox_classifier", load_mutox_model)
