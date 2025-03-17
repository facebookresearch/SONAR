# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import cast, final

from fairseq2.assets import AssetCard
from fairseq2.data.text.tokenizers import AbstractTextTokenizerHandler, TextTokenizer
from fairseq2.models import AbstractModelHandler
from torch.nn import Module
from typing_extensions import override

from sonar.models.laser2_text.config import Laser2Config
from sonar.models.laser2_text.tokenizer import Laser2Tokenizer
from sonar.nn.laser_lstm_encoder import LaserLstmEncoder


@final
class Laser2ModelHandler(AbstractModelHandler):
    @override
    @property
    def family(self) -> str:
        return "lstm"

    @override
    @property
    def kls(self) -> type[Module]:
        return LaserLstmEncoder

    @override
    def _create_model(self, config: object) -> Module:
        config = cast(Laser2Config, config)

        return LaserLstmEncoder(
            num_embeddings=config.vocabulary_size,
            padding_idx=config.pad_idx,
            embed_dim=config.model_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            padding_value=config.padding_value,
        )


@final
class Laser2TokenizerHandler(AbstractTextTokenizerHandler):
    @override
    @property
    def family(self) -> str:
        return "lstm"

    @override
    def _load_tokenizer(self, path: Path, card: AssetCard) -> TextTokenizer:
        return Laser2Tokenizer(path)
