# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from fairseq2.assets import download_manager
from fairseq2.models.utils.model_loader import ModelLoader

from sonar.models.laser2_text.builder import (
    Laser2Config,
    create_laser2_model,
    laser2_archs,
)
from sonar.models.laser2_text.tokenizer import Laser2Tokenizer
from sonar.nn.laser_lstm_encoder import LaserLstmEncoder
from sonar.store import asset_store

load_laser2_model = ModelLoader[LaserLstmEncoder, Laser2Config](
    asset_store, download_manager, create_laser2_model, laser2_archs
)


from fairseq2.assets import AssetDownloadManager, AssetStore, download_manager


class Laser2TokenizerLoader:
    """Loads tokenizers of Laser2 models."""

    def __init__(
        self,
        asset_store: AssetStore = asset_store,
        download_manager: AssetDownloadManager = download_manager,
    ) -> None:
        """
        :param asset_store:
            The asset store to retrieve the model information.
        :param download_manager:
            The download manager to use.
        """
        self.asset_store = asset_store
        self.download_manager = download_manager

    def __call__(
        self, model_name: str, force: bool = False, progress: bool = False
    ) -> Laser2Tokenizer:
        """
        :param name:
            The name of the model.
        :param force:
            If ``True``, downloads the tokenizer even if it is already in cache.
        :param progress:
            If ``True``, displays a progress bar to stderr.
        """
        card = self.asset_store.retrieve_card(model_name)
        uri = card.field("tokenizer").as_uri()
        pathname = self.download_manager.download_tokenizer(
            uri, card.name, force=force, progress=progress
        )
        return Laser2Tokenizer(pathname)


load_laser2_tokenizer = Laser2TokenizerLoader()
