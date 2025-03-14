# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, final

from fairseq2.models import AbstractModelHandler
from torch.nn import Module
from typing_extensions import override

from sonar.models.blaser.config import BlaserConfig
from sonar.models.blaser.factory import create_blaser_model
from sonar.models.blaser.model import BlaserModel


@final
class BlaserModelHandler(AbstractModelHandler):
    @override
    @property
    def family(self) -> str:
        return "blaser"

    @override
    @property
    def kls(self) -> type[Module]:
        return BlaserModel

    @override
    def _create_model(self, config: object) -> Module:
        config = cast(BlaserConfig, config)

        return create_blaser_model(config)

    @override
    def _convert_checkpoint(
        self, checkpoint: dict[str, object], config: object
    ) -> dict[str, object]:
        # Return directly if found fairseq2 attribute in state dict
        if "model" in checkpoint:
            return checkpoint

        # Othewise (the old checkpoint format), move the whole state dict to the "model" section
        return {"model": checkpoint}
