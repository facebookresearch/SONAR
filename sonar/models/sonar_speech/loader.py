# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping

from fairseq2.assets import asset_store, download_manager
from fairseq2.models.utils import ConfigLoader, ModelLoader
from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint

from sonar.models.sonar_speech.builder import (
    SonarSpeechEncoderConfig,
    create_sonar_speech_encoder_model,
    sonar_speech_archs,
)
from sonar.models.sonar_speech.model import SonarSpeechEncoderModel


def convert_sonar_speech_checkpoint(
    checkpoint: Mapping[str, Any], config: SonarSpeechEncoderConfig
) -> Mapping[str, Any]:
    state_dict = checkpoint["model"]

    # Check if we have a fairseq2 checkpoint.
    if "encoder_frontend.model_dim_proj" in state_dict:
        return checkpoint

    if "encoder.w2v_model.mask_emb" in state_dict:
        del state_dict["encoder.w2v_model.mask_emb"]

    key_map = {
        # fmt: off
        # encoder
        r"^encoder.w2v_model.layer_norm\.":                                              r"encoder_frontend.post_extract_layer_norm.",
        r"^encoder.w2v_model.post_extract_proj\.":                                       r"encoder_frontend.model_dim_proj.",
        r"^encoder.w2v_model.encoder\.pos_conv\.0\.":                                    r"encoder_frontend.pos_encoder.conv.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"encoder.layers.\1.conv.batch_norm.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"encoder.layers.\1.conv.depthwise_conv.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"encoder.layers.\1.conv_layer_norm.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"encoder.layers.\1.conv.pointwise_conv1.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"encoder.layers.\1.conv.pointwise_conv2.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"encoder.layers.\1.ffn\2_layer_norm.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"encoder.layers.\1.ffn\2.inner_proj.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"encoder.layers.\1.ffn\2.output_proj.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":         r"encoder.layers.\1.self_attn_layer_norm.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.self_attn\.linear_q\.":          r"encoder.layers.\1.self_attn.q_proj.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.self_attn\.linear_k\.":          r"encoder.layers.\1.self_attn.k_proj.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.self_attn\.linear_v\.":          r"encoder.layers.\1.self_attn.v_proj.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.self_attn\.linear_out\.":        r"encoder.layers.\1.self_attn.output_proj.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.self_attn\.linear_pos\.":        r"encoder.layers.\1.self_attn.sdpa.r_proj.",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_u":          r"encoder.layers.\1.self_attn.sdpa.u_bias",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_v":          r"encoder.layers.\1.self_attn.sdpa.v_bias",
        r"^encoder.w2v_model.encoder\.layers\.([0-9]+)\.final_layer_norm\.":             r"encoder.layers.\1.layer_norm.",
        r"^encoder.w2v_model.encoder\.layer_norm\.":                                     r"encoder.layer_norm.",

        r"^decoder\.embed_tokens\.":                              r"encoder_pooler.decoder_frontend.embed.",
        r"^decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":    r"encoder_pooler.decoder.layers.\1.self_attn_layer_norm.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"encoder_pooler.decoder.layers.\1.self_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.":               r"encoder_pooler.decoder.layers.\1.self_attn.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"encoder_pooler.decoder.layers.\1.encoder_decoder_attn_layer_norm.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"encoder_pooler.decoder.layers.\1.encoder_decoder_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.":            r"encoder_pooler.decoder.layers.\1.encoder_decoder_attn.",
        r"^decoder\.layers\.([0-9]+)\.fc1\.":                     r"encoder_pooler.decoder.layers.\1.ffn.inner_proj.",
        r"^decoder\.layers\.([0-9]+)\.fc2\.":                     r"encoder_pooler.decoder.layers.\1.ffn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"encoder_pooler.decoder.layers.\1.ffn_layer_norm.",

        r"^decoder\.embed_out":                                   r"encoder_pooler.projection_out.weight",
        # fmt: on
    }

    # In normal circumstances, we should never encounter a `LayerNorm` when
    # `use_conformer` is `True`. Unfortunately, the w2v-BERT pretraining in
    # fairseq was accidentally run with a pre-LN encoder, and ended up with
    # a redundant `LayerNorm` right after the Conformer blocks. We mitigate
    # that issue here by moving that `LayerNorm` to the sonar block.
    if config.w2v2_encoder_config.use_conformer:
        key_map.update({r"^encoder.w2v_model.encoder\.layer_norm\.": r"layer_norm."})

    return convert_fairseq_checkpoint(checkpoint, key_map)


load_sonar_speech_config = ConfigLoader[SonarSpeechEncoderConfig](
    asset_store, sonar_speech_archs
)

load_sonar_speech_model = ModelLoader[
    SonarSpeechEncoderModel, SonarSpeechEncoderConfig
](
    asset_store,
    download_manager,
    load_sonar_speech_config,
    create_sonar_speech_encoder_model,
    convert_sonar_speech_checkpoint,
)
