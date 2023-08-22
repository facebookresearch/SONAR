# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.testing import assert_close

from sonar.models.blaser.loader import load_blaser_model


def test_blaser2_ref():
    """Compare predictions of a specific reference-based model with hardcoded expected values"""
    blaser = load_blaser_model("blaser_2_0_ref")
    blaser.eval()
    emb = torch.zeros([1, 1024]) + 1 / 32
    pred = blaser(src=emb, mt=emb, ref=emb).item()
    assert_close(pred, 5.255207538604736)

    pred = blaser(src=emb, mt=emb, ref=-emb).item()
    assert_close(pred, 2.309619665145874)

    pred = blaser(src=emb, mt=-emb, ref=emb).item()
    assert_close(pred, -2.178907632827759)


def test_blaser2_qe():
    """Compare predictions of a specific referenceless model with hardcoded expected values"""
    blaser = load_blaser_model("blaser_2_0_qe")
    blaser.eval()
    emb = torch.zeros([1, 1024]) + 1 / 32
    pred = blaser(src=emb, mt=emb).item()
    assert_close(pred, 4.981893062591553)

    pred = blaser(src=emb, mt=-emb).item()
    assert_close(pred, -0.8291061520576477)
