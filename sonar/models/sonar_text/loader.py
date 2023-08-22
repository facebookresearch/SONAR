# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, final

import torch
from fairseq2.assets import download_manager
from fairseq2.models.nllb.loader import NllbTokenizerLoader
from fairseq2.models.transformer import TransformerDecoderModel
from fairseq2.models.utils.checkpoint_loader import upgrade_fairseq_checkpoint
from fairseq2.models.utils.model_loader import ModelLoader
from overrides import override as finaloverride

from sonar.models.sonar_text.builder import (
    SonarTextDecoderConfig,
    SonarTextEncoderConfig,
    create_sonar_text_decoder_model,
    create_sonar_text_encoder_model,
    sonar_text_decoder_archs,
    sonar_text_encoder_archs,
)
from sonar.models.sonar_text.model import SonarTextTransformerEncoderModel
from sonar.store import asset_store

load_sonar_tokenizer = NllbTokenizerLoader(asset_store, download_manager)


@final
class SonarTextEncoderLoader(
    ModelLoader[SonarTextTransformerEncoderModel, SonarTextEncoderConfig]
):
    """Loads SonarEncoder models."""

    @finaloverride
    def _upgrade_checkpoint(
        self, checkpoint: Mapping[str, Any], config: SonarTextEncoderConfig
    ) -> Mapping[str, Any]:
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
        out_checkpoint = upgrade_fairseq_checkpoint(
            out_checkpoint, self._fairseq_key_map()
        )
        embeds = checkpoint["embed_tokens"].weight
        # # The embedding positions of the control tokens do not match the
        # # SentencePiece model of the tokenizer.
        with torch.inference_mode():
            # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
            embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]
        out_checkpoint["encoder_frontend.embed.weight"] = embeds

        return out_checkpoint

    @staticmethod
    def _fairseq_key_map() -> Dict[str, str]:
        return {
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


load_sonar_text_encoder_model = SonarTextEncoderLoader(
    asset_store,
    download_manager,
    create_sonar_text_encoder_model,
    sonar_text_encoder_archs,
)


@final
class SonarTextDecoderLoader(
    ModelLoader[TransformerDecoderModel, SonarTextDecoderConfig]
):
    """Loads SonarEncoder models."""

    @finaloverride
    def _upgrade_checkpoint(
        self, checkpoint: Mapping[str, Any], config: SonarTextDecoderConfig
    ) -> Mapping[str, Any]:
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
        out_checkpoint = upgrade_fairseq_checkpoint(
            out_checkpoint, self._fairseq_key_map()
        )
        embeds = out_checkpoint["model"]["decoder_frontend.embed.weight"]
        # # The embedding positions of the control tokens do not match the
        # # SentencePiece model of the tokenizer.
        with torch.inference_mode():
            # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
            embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]
        out_checkpoint["model"]["decoder_frontend.embed.weight"] = embeds
        return out_checkpoint

    @staticmethod
    def _fairseq_key_map() -> Dict[str, str]:
        return {
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


load_sonar_text_decoder_model = SonarTextDecoderLoader(
    asset_store,
    download_manager,
    create_sonar_text_decoder_model,
    sonar_text_decoder_archs,
)
