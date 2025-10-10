# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq2.nn.batch_layout import BatchLayout
from torch.testing import assert_close  # type: ignore

from sonar.models.sonar_text.model import Pooling, SonarTextTransformerEncoderModel

pooling_method = SonarTextTransformerEncoderModel.static_pooling


def test_pooling_max() -> None:
    # padding_mask = PaddingMask(torch.tensor([2, 1]), batch_seq_len=3)
    seqs = torch.Tensor(
        [[[7, 2], [3, 4], [10, 20]], [[-1, -2], [100, 1000], [-10, -20]]]
    )
    seqs_layout = BatchLayout.of(seqs, seq_lens=[2, 1])
    expected = torch.Tensor([[7.0, 4.0], [-1.0, -2.0]])
    actual = pooling_method(seqs, seqs_layout, Pooling.MAX)
    assert_close(expected, actual)

    actual_extra = pooling_method(seqs.unsqueeze(3), seqs_layout, Pooling.MAX)
    assert_close(expected.unsqueeze(2), actual_extra)


def test_pooling_mean() -> None:
    # padding_mask = PaddingMask(torch.tensor([2, 1]), batch_seq_len=3)
    seqs = torch.Tensor(
        [[[7, 2], [3, 4], [10, 20]], [[-1, -2], [100, 1000], [-10, -20]]]
    )
    seqs_layout = BatchLayout.of(seqs, seq_lens=[2, 1])
    expected = torch.Tensor([[5.0, 3.0], [-1.0, -2.0]])
    actual = pooling_method(seqs, seqs_layout, Pooling.MEAN)
    assert_close(expected, actual)

    actual_extra = pooling_method(seqs.unsqueeze(3), seqs_layout, Pooling.MEAN)
    assert_close(expected.unsqueeze(2), actual_extra)


def test_pooling_last() -> None:
    # padding_mask = PaddingMask(torch.tensor([2, 1]), batch_seq_len=3)
    seqs = torch.Tensor(
        [[[7, 2], [3, 4], [10, 20]], [[-1, -2], [100, 1000], [-10, -20]]]
    )
    seqs_layout = BatchLayout.of(seqs, seq_lens=[2, 1])
    expected = torch.Tensor([[3.0, 4.0], [-1.0, -2.0]])
    actual = pooling_method(seqs, seqs_layout, Pooling.LAST)
    assert_close(expected, actual)

    actual_extra = pooling_method(seqs.unsqueeze(3), seqs_layout, Pooling.LAST)
    assert_close(expected.unsqueeze(2), actual_extra)


def test_pooling_mean_none_padding() -> None:
    seqs = torch.Tensor([[[7, 2], [3, 2], [2, 20]], [[-1, -3], [-4, 2], [-7, -2]]])

    expected1 = torch.Tensor([[2, 20], [-7, -2]])
    actual1 = pooling_method(seqs, None, Pooling.LAST)
    assert_close(expected1, actual1)

    expected2 = torch.Tensor([[7, 20], [-1, 2]])
    actual2 = pooling_method(seqs, None, Pooling.MAX)
    assert_close(expected2, actual2)

    expected3 = torch.Tensor([[4, 8], [-4, -1]])
    actual3 = pooling_method(seqs, None, Pooling.MEAN)
    assert_close(expected3, actual3)
