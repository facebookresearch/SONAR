# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.testing import assert_close  # type: ignore

from sonar.nn.utils import compute_seq_length


def test_compute_seq_length_edge_cases() -> None:
    seqs: torch.Tensor = torch.Tensor(
        [[2, 3, 1, 2], [4, 3, 2, 2], [-1, 1, 2, 1], [1, 2, 3, 1], [1, 1, 1, 1]]
    )
    actual1: torch.Tensor = compute_seq_length(seqs, 1)  # type: ignore
    expected1: torch.Tensor = torch.Tensor([2, 4, 1, 0, 0]).long()
    assert_close(actual1, expected1)

    actual2: torch.Tensor = compute_seq_length(seqs, 2)  # type: ignore
    expected2: torch.Tensor = torch.Tensor([0, 2, 2, 1, 4]).long()
    assert_close(actual2, expected2)

    inf_seqs = torch.where(seqs == 1, -torch.inf, 0)
    actual3: torch.Tensor = compute_seq_length(inf_seqs, -torch.inf)  # type: ignore
    expected3: torch.Tensor = torch.Tensor([2, 4, 1, 0, 0]).long()
    assert_close(actual3, expected3)
