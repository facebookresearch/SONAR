# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SONAR provides a set of speech and text encoders for multilingual, multimodal semantic embedding.

"""

from pathlib import Path

from fairseq2.assets import FileAssetMetadataProvider, asset_store

__version__ = "0.2.1"


def _update_asset_store() -> None:
    cards_dir = Path(__file__).parent.joinpath("cards")

    # Make sure that the default fairseq2 asset store can resolve cards under
    # the directory <sonar>/cards.
    asset_store.metadata_providers.append(FileAssetMetadataProvider(cards_dir))


_update_asset_store()
