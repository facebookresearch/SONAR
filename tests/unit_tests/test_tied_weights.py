# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path

import torch
from fairseq2.assets import InProcAssetMetadataProvider, default_asset_store
from fairseq2.assets.card import AssetCard

from sonar.models.sonar_text.builder import SonarTextDecoderBuilder, decoder_toy
from sonar.models.sonar_text.loader import load_sonar_text_decoder_model


def create_model_card(
    checkpoint_path: Path,
    model_type: str,
    model_arch: str,
    model_name: str = "on_the_fly_model",
) -> AssetCard:
    model_card_info = {
        "name": model_name,
        "model_type": model_type,
        "model_family": model_type,
        "model_arch": model_arch,
        "checkpoint": "file://" + checkpoint_path.as_posix(),
    }
    default_asset_store.metadata_providers.append(
        InProcAssetMetadataProvider([model_card_info])
    )
    return default_asset_store.retrieve_card(model_name)


def test_tied_weight():
    """Testing that the decoder input and ouput embeddings are tied after creating the model and after loading"""
    cfg = decoder_toy()
    model = SonarTextDecoderBuilder(cfg).build_model()
    assert model.decoder_frontend.embed.weight is model.final_proj.weight

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
            checkpoint_path=filename,
            model_type="transformer_decoder",
            model_arch="toy",
        )
        model_new = load_sonar_text_decoder_model(card)

        # test that the newly loaded model has the same weight tying as the original one
        total_params_new = sum(p.numel() for p in model_new.parameters())
        assert total_params_new == total_params
        assert model_new.decoder_frontend.embed.weight is model_new.final_proj.weight
