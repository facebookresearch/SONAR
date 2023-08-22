# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

BLASER_INPUT_FORMS = {
    "COMET",
    "QE",
}


ACTIVATIONS = {
    "TANH": nn.Tanh,
    "RELU": nn.ReLU,
}


class BlaserModel(nn.Module):
    """
    A multilayer perceptron over concatenated embeddings of source, translation and optionally reference,
    and their pointwise products and differences.
    """

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float,
        activation: str,
        input_form: str,
        norm_emb: bool,
        output_act: bool,
    ):
        super(BlaserModel, self).__init__()
        self.input_form = input_form
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.norm_emb = norm_emb
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.output_act = output_act

        if input_form == "COMET":
            embedding_dim *= 6
        elif input_form == "QE":
            embedding_dim *= 4
        else:
            raise Exception(f"Unrecognized input format: {input_form}")

        if activation not in ACTIVATIONS:
            raise Exception(f"Unrecognized activation: {activation}")

        modules: List[nn.Module] = []
        if len(self.hidden_dims) > 0:
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
            nprev = embedding_dim
            for hidden_size in self.hidden_dims:
                if hidden_size > 0:
                    modules.append(nn.Linear(nprev, hidden_size))
                    nprev = hidden_size
                    modules.append(ACTIVATIONS[activation]())
                    if dropout > 0:
                        modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(nprev, output_dim))
            if output_act:
                modules.append(nn.Tanh())
        else:
            modules.append(nn.Linear(embedding_dim, output_dim))
        self.mlp = nn.Sequential(*modules)

    def forward(self, src: Tensor, mt: Tensor, ref: Optional[Tensor] = None) -> Tensor:
        proc = self.featurize_input(
            src=self._norm_vec(src),  # type: ignore
            mt=self._norm_vec(mt),  # type: ignore
            ref=self._norm_vec(ref),
        )
        return self.mlp(proc)

    def _norm_vec(self, emb: Optional[Tensor]) -> Optional[Tensor]:
        if self.norm_emb and emb is not None:
            return F.normalize(emb)
        else:
            return emb

    def featurize_input(
        self, src: Tensor, mt: Tensor, ref: Optional[Tensor] = None
    ) -> Tensor:
        if self.input_form == "COMET":
            if ref is None:
                raise ValueError(
                    "With the COMET input form of BLASER, a reference embedding must be provided."
                )
            processed = torch.cat(
                [
                    ref,
                    mt,
                    src * mt,
                    ref * mt,
                    torch.absolute(mt - src),
                    torch.absolute(mt - ref),
                ],
                dim=-1,
            )
        elif self.input_form == "QE":
            processed = torch.cat(
                [
                    src,
                    mt,
                    src * mt,
                    torch.absolute(mt - src),
                ],
                dim=-1,
            )
        return processed
