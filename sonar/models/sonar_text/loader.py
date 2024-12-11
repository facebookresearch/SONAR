# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch
from fairseq2.models.config_loader import StandardModelConfigLoader
from fairseq2.models.loader import StandardModelLoader, load_model
from fairseq2.models.nllb import load_nllb_tokenizer
from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint

from sonar.models.sonar_text.builder import (
    SonarTextDecoderConfig,
    SonarTextEncoderConfig,
    create_sonar_text_decoder_model,
    create_sonar_text_encoder_model,
    sonar_text_decoder_archs,
    sonar_text_encoder_archs,
)


def convert_sonar_text_encoder_checkpoint(
    checkpoint: Dict[str, Any], config: SonarTextEncoderConfig
) -> Dict[str, Any]:
    # Return directly if found fairseq2 attribute in state dict
    if (
        "model" in checkpoint.keys()
        and "encoder_frontend.embed.weight" in checkpoint["model"].keys()
    ):
        return checkpoint

    state_dict = checkpoint["state_dict"]

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

    out_checkpoint = convert_fairseq_checkpoint(out_checkpoint, key_map)

    embeds = checkpoint["embed_tokens"].weight
    # # The embedding positions of the control tokens do not match the
    # # SentencePiece model of the tokenizer.
    with torch.inference_mode():
        # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
        embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]
    out_checkpoint["encoder_frontend.embed.weight"] = embeds

    return out_checkpoint


load_sonar_text_encoder_config = StandardModelConfigLoader(
    family="transformer_encoder",
    config_kls=SonarTextEncoderConfig,
    arch_configs=sonar_text_encoder_archs,
)

load_sonar_text_encoder_model = StandardModelLoader(
    config_loader=load_sonar_text_encoder_config,
    factory=create_sonar_text_encoder_model,
    checkpoint_converter=convert_sonar_text_encoder_checkpoint,
    restrict_checkpoints=False,
)

load_model.register("transformer_encoder", load_sonar_text_encoder_model)


def convert_sonar_text_decoder_checkpoint(
    checkpoint: Dict[str, Any], config: SonarTextDecoderConfig
) -> Dict[str, Any]:
    # Return directly if found fairseq2 attribute in state dict
    if (
        "model" in checkpoint.keys()
        and "decoder_frontend.embed.weight" in checkpoint["model"].keys()
    ):
        return checkpoint

    state_dict = checkpoint["state_dict"]
    try:
        del state_dict["version"]
        del state_dict["embed_positions._float_tensor"]
    except:
        pass

    out_checkpoint = {"model": state_dict}

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

    out_checkpoint = convert_fairseq_checkpoint(out_checkpoint, key_map)

    embeds = out_checkpoint["model"]["decoder_frontend.embed.weight"]
    # # The embedding positions of the control tokens do not match the
    # # SentencePiece model of the tokenizer.
    with torch.inference_mode():
        # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
        embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]
    out_checkpoint["model"]["decoder_frontend.embed.weight"] = embeds
    return out_checkpoint


load_sonar_text_decoder_config = StandardModelConfigLoader(
    family="transformer_decoder",
    config_kls=SonarTextDecoderConfig,
    arch_configs=sonar_text_decoder_archs,
)

load_sonar_text_decoder_model = StandardModelLoader(
    config_loader=load_sonar_text_decoder_config,
    factory=create_sonar_text_decoder_model,
    checkpoint_converter=convert_sonar_text_decoder_checkpoint,
    restrict_checkpoints=False,
)

load_model.register("transformer_decoder", load_sonar_text_decoder_model)

load_sonar_tokenizer = load_nllb_tokenizer
