# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path

import torch
from fairseq2.assets import (
    AssetCard,
    StandardAssetStore,
    get_asset_store,
    load_in_memory_asset_metadata,
)
from fairseq2.runtime.config_registry import get_config
from fairseq2.runtime.dependency import get_dependency_resolver

from sonar.models.sonar_text import (
    SONAR_TEXT_DECODER_FAMILY,
    SonarTextDecoderConfig,
    SonarTextDecoderFactory,
    get_sonar_text_decoder_hub,
)


def create_model_card(
    asset_store: StandardAssetStore,
    checkpoint_path: Path,
    model_type: str,
    model_arch: str,
    model_name: str = "on_the_fly_model",
) -> AssetCard:
    model_card_info: dict[str, object] = {
        "name": model_name,
        "model_type": model_type,
        "model_family": model_type,
        "model_arch": model_arch,
        "checkpoint": "file://" + checkpoint_path.as_posix(),
    }
    metadata_provider = load_in_memory_asset_metadata("memory", [model_card_info])
    asset_store._metadata_providers.append(metadata_provider)
    return asset_store.retrieve_card(model_name)


def test_tied_weight():
    """Testing that the decoder input and ouput embeddings are tied after creating the model and after loading"""

    resolver = get_dependency_resolver()
    config = get_config(resolver, SonarTextDecoderConfig, "toy")

    model = SonarTextDecoderFactory(config).create_model()
    assert model.decoder_frontend.embed.weight is model.final_proj.weight  # type: ignore

    # counting the parameters
    total_params = sum(p.numel() for p in model.parameters())
    frontend_params = sum(p.numel() for p in model.decoder_frontend.parameters())
    transformer_body_params = sum(p.numel() for p in model.decoder.parameters())
    final_proj_params = sum(p.numel() for p in model.final_proj.parameters())

    assert final_proj_params == frontend_params
    assert total_params == frontend_params + transformer_body_params

    # save the model to disk, to check that the weight tying still works after loading it back
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "checkpoint.pt"
        torch.save({"model": model.state_dict()}, filename)

        # now load the model using a standard loader, based on a card
        card = create_model_card(
            get_asset_store(),  # type: ignore
            checkpoint_path=filename,
            model_type=SONAR_TEXT_DECODER_FAMILY,
            model_arch="toy",
        )
        print(card)
        decoder_hub = get_sonar_text_decoder_hub()
        model_new = decoder_hub.load_model(card)

        # test that the newly loaded model has the same weight tying as the original one
        total_params_new = sum(p.numel() for p in model_new.parameters())
        assert total_params_new == total_params
        assert model_new.decoder_frontend.embed.weight is model_new.final_proj.weight  # type: ignore
