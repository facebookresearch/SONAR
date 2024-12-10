# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import final

from fairseq2.assets import AssetCard
from fairseq2.data.text import AbstractTextTokenizerLoader
from fairseq2.models.config_loader import StandardModelConfigLoader
from fairseq2.models.loader import StandardModelLoader, load_model
from fairseq2.typing import override

from sonar.models.laser2_text.builder import (
    Laser2Config,
    create_laser2_model,
    laser2_archs,
)
from sonar.models.laser2_text.tokenizer import Laser2Tokenizer

load_laser2_config = StandardModelConfigLoader(
    family="lstm", config_kls=Laser2Config, arch_configs=laser2_archs
)

load_laser2_model = StandardModelLoader(
    config_loader=load_laser2_config,
    factory=create_laser2_model,
    restrict_checkpoints=False,
)

load_model.register("lstm", load_laser2_model)


@final
class Laser2TokenizerLoader(AbstractTextTokenizerLoader[Laser2Tokenizer]):
    @override
    def _load(self, path: Path, card: AssetCard) -> Laser2Tokenizer:
        return Laser2Tokenizer(path)


load_laser2_tokenizer = Laser2TokenizerLoader()
