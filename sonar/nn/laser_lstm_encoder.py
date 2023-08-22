# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class LaserLstmEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        padding_idx: int,
        embed_dim: int = 320,
        hidden_size: int = 512,
        num_layers: int = 1,
        bidirectional: bool = False,
        padding_value: float = 0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(
            num_embeddings, embed_dim, padding_idx=self.padding_idx
        )

        self.lstm: nn.Module = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    @staticmethod
    def _sort_by_length(
        seqs: Tensor, seq_lens: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        order_indices = torch.argsort(-seq_lens)
        return (
            seqs[order_indices],
            seq_lens[order_indices],
            torch.argsort(order_indices),
        )

    def forward(self, seqs: Tensor, seq_lens: Tensor) -> Tensor:
        src_tokens, lengths, order_indices = self._sort_by_length(seqs, seq_lens)
        bsz, max_seqlen = src_tokens.size()

        # embed tokens
        embed_tokens = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        embed_tokens = embed_tokens.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(embed_tokens, lengths)

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = embed_tokens.data.new(*state_size).zero_()
        c0 = embed_tokens.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        embed_tokens, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value
        )
        assert list(embed_tokens.size()) == [max_seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs: Tensor) -> Tensor:
                return torch.cat(
                    [
                        torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(
                            1, bsz, self.output_units
                        )
                        for i in range(self.num_layers)
                    ],
                    dim=0,
                )

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            embed_tokens = (
                embed_tokens.float()
                .masked_fill_(padding_mask, float("-inf"))
                .type_as(embed_tokens)
            )

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = embed_tokens.max(dim=0)[0]
        sentemb = sentemb[order_indices]
        return torch.as_tensor(sentemb)  # cast back to Tensor for mypy
