# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class MutoxClassifier(nn.Module):
    def __init__(
        self,
        model_all,
    ):
        super().__init__()
        self.model_all = model_all

    def forward(self, inputs: torch.Tensor, output_prob: bool = False) -> torch.Tensor:
        outputs = self.model_all(inputs)

        if output_prob:
            outputs = torch.sigmoid(outputs)

        return outputs
