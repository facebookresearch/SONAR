# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# References:
#
# SynCap: https://github.com/e-bug/syncap
# --------------------------------------------------------

from typing import List

import torch
import torch.nn as nn

"""
GradNorm (https://arxiv.org/abs/1711.02257) wrapper for joint contrastive (CLIP/FLIP-style) & MAE training

Implementation adapted from https://github.com/e-bug/syncap/blob/master/code/train.py
"""


class GradNorm(nn.Module):
    def __init__(
        self, alpha: float, initial_contrastive_loss: torch.Tensor, initial_mae_loss: torch.Tensor
    ):
        """
        Creates a GradNorm wrapper

        The initial losses need to be pre-computed
        """

        super().__init__()

        self.alpha = alpha
        self.contrastive_weight = nn.Parameter(torch.ones(1, dtype=torch.float))
        self.mae_weight = nn.Parameter(torch.ones(1, dtype=torch.float))

        self.register_buffer("initial_contrastive_loss", initial_contrastive_loss)
        self.register_buffer("initial_mae_loss", initial_mae_loss)

        self.criterion = nn.L1Loss()

    def forward(
        self,
        contrastive_loss: torch.Tensor,
        mae_loss: torch.Tensor,
        shared_params: List[nn.Parameter],
    ) -> torch.Tensor:
        """
        Computes the GradNorm loss based on the two losses (contrastive and MAE) and their shared parameters
        """

        # Get the gradients of the shared layers and calculate their l2-norm
        # Contrastive
        G1R = torch.autograd.grad(
            contrastive_loss,
            shared_params,
            retain_graph=True,
            create_graph=True,
        )
        G1R_flattened = torch.cat([g.view(-1) for g in G1R if g is not None])
        G1 = torch.norm(self.contrastive_weight * G1R_flattened.data, 2).unsqueeze(0)

        # MAE
        G2R = torch.autograd.grad(mae_loss, shared_params)
        G2R_flattened = torch.cat([g.view(-1) for g in G2R])
        G2 = torch.norm(self.mae_weight * G2R_flattened.data, 2).unsqueeze(0)

        # Calculate the average gradient norm across all tasks
        G_avg = torch.mean(G1 + G2)

        # Calculate relative losses
        lhat1 = torch.div(contrastive_loss, self.initial_contrastive_loss)
        lhat2 = torch.div(mae_loss, self.initial_mae_loss)
        lhat_avg = torch.mean(lhat1 + lhat2)

        # Calculate relative inverse training rates
        inv_rate1 = torch.div(lhat1, lhat_avg)
        inv_rate2 = torch.div(lhat2, lhat_avg)

        # Calculate the gradient norm target for this batch
        C1 = G_avg * (inv_rate1**self.alpha)
        C2 = G_avg * (inv_rate2**self.alpha)

        # Calculate the gradnorm loss
        gradnorm_loss = self.criterion(G1, C1.data.unsqueeze(0)) + self.criterion(
            G2, C2.data.unsqueeze(0)
        )

        return gradnorm_loss

    def rescale_weights(self) -> None:
        """
        Renormalizes loss weights and ensures they are positive
        """
        coef = 2 / (torch.abs(self.contrastive_weight) + torch.abs(self.mae_weight))
        self.contrastive_weight.data = coef * torch.abs(self.contrastive_weight.data)
        self.mae_weight.data = coef * torch.abs(self.mae_weight.data)
