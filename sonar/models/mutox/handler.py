# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, final

from fairseq2.models import AbstractModelHandler
from torch.nn import Module
from typing_extensions import override

from sonar.models.mutox.config import MutoxConfig
from sonar.models.mutox.factory import create_mutox_model
from sonar.models.mutox.model import MutoxClassifier


@final
class MutoxModelHandler(AbstractModelHandler):
    @override
    @property
    def family(self) -> str:
        return "mutox_classifier"

    @override
    @property
    def kls(self) -> type[Module]:
        return MutoxClassifier

    @override
    def _create_model(self, config: object) -> Module:
        config = cast(MutoxConfig, config)

        return create_mutox_model(config)

    @override
    def _convert_checkpoint(
        self, checkpoint: dict[str, object], config: object
    ) -> dict[str, object]:
        new_dict = {}
        for key in checkpoint:
            if key.startswith("model_all."):
                new_dict[key] = checkpoint[key]
        return {"model": new_dict}
