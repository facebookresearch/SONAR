# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.testing import assert_close

from sonar.models.blaser.builder import BlaserConfig, create_blaser_model


@pytest.mark.parametrize("embedding_dim", [32, 1024])
@pytest.mark.parametrize("batch_size", [1, 10])
def test_blaser_qe(embedding_dim, batch_size):
    """Testing that a BLASER-QE model can be created and runs"""
    config = BlaserConfig(input_form="QE", embedding_dim=embedding_dim)
    blaser = create_blaser_model(config).eval()
    embedding = torch.zeros([batch_size, embedding_dim])

    # test that the forward method produces an expected shape
    result = blaser(src=embedding, mt=embedding, ref=embedding)
    assert result.shape == (batch_size, 1)

    # test that reference does not matter
    result2 = blaser(src=embedding, mt=embedding)
    assert_close(result, result2)


@pytest.mark.parametrize("embedding_dim", [32, 1024])
@pytest.mark.parametrize("batch_size", [1, 10])
def test_blaser_ref(embedding_dim, batch_size):
    """Testing that a model can be created and that forward returns a right shape"""
    config = BlaserConfig(input_form="COMET", embedding_dim=embedding_dim)
    blaser = create_blaser_model(config)
    embedding = torch.zeros([batch_size, embedding_dim])

    # test that the forward method produces an expected shape
    result = blaser(src=embedding, mt=embedding, ref=embedding)
    assert result.shape == (batch_size, 1)

    # test that reference is required
    with pytest.raises(ValueError):
        result = blaser(src=embedding, mt=embedding)


@pytest.mark.parametrize("input_form", ["COMET", "QE"])
@pytest.mark.parametrize("embedding_dim", [32, 1024])
def test_input_form(input_form, embedding_dim):
    """Testing that BLASER inputs are processed correctlyb"""
    config = BlaserConfig(input_form=input_form, embedding_dim=embedding_dim)
    blaser = create_blaser_model(config)
    # the input vectors are arbitrary; we are checking only how they are concatenated
    src = torch.arange(0, embedding_dim).unsqueeze(0) / embedding_dim
    mt = torch.cos(src)
    ref = torch.exp(src)

    features = blaser.featurize_input(src=src, mt=mt, ref=ref)
    assert blaser.mlp[1].in_features == features.shape[1]

    if input_form == "COMET":
        expected = [
            ref,
            mt,
            src * mt,
            ref * mt,
            torch.absolute(mt - src),
            torch.absolute(mt - ref),
        ]
    else:
        expected = [src, mt, src * mt, torch.absolute(mt - src)]
    assert_close(features, torch.cat(expected, dim=-1))
