# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, cast

import torch
from fairseq2.models.utils.checkpoint import convert_fairseq_state_dict

from sonar.models.sonar_text.config import (
    SonarTextDecoderConfig,
    SonarTextEncoderConfig,
)


def _convert_sonar_text_encoder_checkpoint(
    state_dict: Dict[str, Any], config: SonarTextEncoderConfig
) -> Dict[str, Any]:
    # fairseq2 does not use a top-level "model" keyword anymore (v0.5+)
    try:
        state_dict = cast(dict[str, object], state_dict["model"])
    except KeyError:
        pass

    # Return directly if found fairseq2 attribute in state dict
    if "encoder_frontend.embed.weight" in state_dict.keys():
        return state_dict

    state_dict = state_dict["state_dict"]

    try:
        del state_dict["version"]
        del state_dict["embed_positions._float_tensor"]
    except:
        pass
    # del state_dict["decoder.version"]

    out_checkpoint = {"model": state_dict}

    key_map = {
        r"layers\.([0-9]+)\.self_attn\.q_proj\.": r"encoder.layers.\1.self_attn.q_proj.",
        r"layers\.([0-9]+)\.self_attn\.v_proj\.": r"encoder.layers.\1.self_attn.v_proj.",
        r"layers\.([0-9]+)\.self_attn\.k_proj\.": r"encoder.layers.\1.self_attn.k_proj.",
        r"layers\.([0-9]+)\.self_attn\.out_proj\.": r"encoder.layers.\1.self_attn.output_proj.",
        r"layers\.([0-9]+)\.self_attn_layer_norm\.": r"encoder.layers.\1.self_attn_layer_norm.",
        r"layers\.([0-9]+)\.fc1\.": r"encoder.layers.\1.ffn.inner_proj.",
        r"layers\.([0-9]+)\.fc2\.": r"encoder.layers.\1.ffn.output_proj.",
        r"layers\.([0-9]+)\.final_layer_norm\.": r"encoder.layers.\1.ffn_layer_norm.",
        r"embed_tokens\.": r"encoder_frontend.embed.",
        # fmt: on
    }

    out_checkpoint = convert_fairseq_state_dict(out_checkpoint, key_map)  # type: ignore

    embeds = state_dict["embed_tokens"].weight
    # # The embedding positions of the control tokens do not match the
    # # SentencePiece model of the tokenizer.
    with torch.inference_mode():
        # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
        embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]
    out_checkpoint["encoder_frontend.embed.weight"] = embeds

    return out_checkpoint


def _convert_sonar_text_decoder_checkpoint(
    state_dict: dict[str, Any], config: SonarTextDecoderConfig
) -> dict[str, Any]:
    # fairseq2 does not use a top-level "model" keyword anymore (v0.5+)
    try:
        state_dict = cast(dict[str, object], state_dict["model"])
    except KeyError:
        pass

    # Return directly if found fairseq2 attribute in state dict
    if "decoder_frontend.embed.weight" in state_dict.keys():
        return state_dict

    # assuming pre fs2:v0.5 formatting with top-level "model" key
    state_dict = state_dict["state_dict"]
    try:
        del state_dict["version"]
        del state_dict["embed_positions._float_tensor"]
    except:
        pass

    out_checkpoint = state_dict

    key_map = {
        r"layers\.([0-9]+)\.self_attn\.k_proj\.": r"decoder.layers.\1.self_attn.k_proj.",
        r"layers\.([0-9]+)\.self_attn\.v_proj\.": r"decoder.layers.\1.self_attn.v_proj.",
        r"layers\.([0-9]+)\.self_attn\.q_proj\.": r"decoder.layers.\1.self_attn.q_proj.",
        r"layers\.([0-9]+)\.self_attn.out_proj\.": r"decoder.layers.\1.self_attn.output_proj.",
        r"layers\.([0-9]+)\.self_attn_layer_norm\.": r"decoder.layers.\1.self_attn_layer_norm.",
        r"layers\.([0-9]+).ffn\.inner_proj\.": r"decoder.layers.\1.ffn.inner_proj.",
        r"layers\.([0-9]+).ffn\.output_proj\.": r"decoder.layers.\1.ffn.output_proj.",
        r"layers\.([0-9]+)\.ffn_layer_norm\.": r"decoder.layers.\1.ffn_layer_norm.",
        r"layers\.([0-9]+).encoder_attn\.k_proj\.": r"decoder.layers.\1.encoder_decoder_attn.k_proj.",
        r"layers\.([0-9]+).encoder_attn\.v_proj\.": r"decoder.layers.\1.encoder_decoder_attn.v_proj.",
        r"layers\.([0-9]+).encoder_attn\.q_proj\.": r"decoder.layers.\1.encoder_decoder_attn.q_proj.",
        r"layers\.([0-9]+).encoder_attn\.out_proj\.": r"decoder.layers.\1.encoder_decoder_attn.output_proj.",
        r"layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"decoder.layers.\1.encoder_decoder_attn_layer_norm.",
        r"layers\.([0-9]+)\.fc1\.": r"decoder.layers.\1.ffn.inner_proj.",
        r"layers\.([0-9]+)\.fc2\.": r"decoder.layers.\1.ffn.output_proj.",
        r"layers\.([0-9]+)\.final_layer_norm\.": r"decoder.layers.\1.ffn_layer_norm.",
        r"output_projection.": r"final_proj.",
        r"embed_tokens.": r"decoder_frontend.embed.",
        r"layer_norm.": r"decoder.layer_norm.",
    }

    out_checkpoint = convert_fairseq_state_dict(out_checkpoint, key_map)

    out_checkpoint = cast(dict[str, Any], out_checkpoint)

    embeds = out_checkpoint["decoder_frontend.embed.weight"]
    # # The embedding positions of the control tokens do not match the
    # # SentencePiece model of the tokenizer.
    with torch.inference_mode():
        # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
        embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]
    out_checkpoint["decoder_frontend.embed.weight"] = embeds
    return out_checkpoint
