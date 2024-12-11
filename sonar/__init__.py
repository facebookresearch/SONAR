# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SONAR provides a set of speech and text encoders for multilingual, multimodal semantic embedding."""

from fairseq2.assets import default_asset_store

__version__ = "0.3.1"


def setup_fairseq2() -> None:
    # Make sure that the default fairseq2 asset store can resolve cards under
    # the directory <sonar>/cards.
    default_asset_store.add_package_metadata_provider("sonar.cards")


setup_fairseq2()
