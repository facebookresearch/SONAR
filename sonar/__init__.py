# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SONAR provides a set of speech and text encoders for multilingual, multimodal semantic embedding.
"""

from fairseq2 import setup_fairseq2
from fairseq2.config_registry import ConfigProvider
from fairseq2.context import RuntimeContext
from fairseq2.data.text.tokenizers import TextTokenizerHandler
from fairseq2.models import ModelHandler
from fairseq2.setup import register_package_metadata_provider
from fairseq2.utils.file import TorchTensorLoader

from sonar.models.blaser import (
    BlaserConfig,
    BlaserModelHandler,
    register_blaser_configs,
)
from sonar.models.laser2_text import (
    Laser2Config,
    Laser2ModelHandler,
    Laser2TokenizerHandler,
    register_laser2_configs,
)
from sonar.models.mutox import MutoxConfig, MutoxModelHandler, register_mutox_configs
from sonar.models.sonar_speech import (
    SonarSpeechEncoderConfig,
    SonarSpeechEncoderHandler,
    register_sonar_speech_encoder_configs,
)
from sonar.models.sonar_text import (
    SonarTextDecoderConfig,
    SonarTextDecoderHandler,
    SonarTextEncoderConfig,
    SonarTextEncoderHandler,
    register_sonar_text_decoder_configs,
    register_sonar_text_encoder_configs,
)

__version__ = "0.4.0"


def setup_fairseq2_extension(context: RuntimeContext) -> None:
    # Make sure that the default fairseq2 asset store can resolve cards under
    # the directory <sonar>/cards.
    register_package_metadata_provider(context, "sonar.cards")

    _register_models(context)

    _register_text_tokenizers(context)


def _register_models(context: RuntimeContext) -> None:
    asset_download_manager = context.asset_download_manager

    tensor_loader = TorchTensorLoader(context.file_system, restrict=False)

    registry = context.get_registry(ModelHandler)

    handler: ModelHandler

    configs: ConfigProvider[object]

    # Blaser
    configs = context.get_config_registry(BlaserConfig)

    default_arch = "basic_ref"

    handler = BlaserModelHandler(
        configs, default_arch, asset_download_manager, tensor_loader
    )

    registry.register(handler.family, handler)

    register_blaser_configs(context)

    # Laser2
    configs = context.get_config_registry(Laser2Config)

    default_arch = "laser2"

    handler = Laser2ModelHandler(
        configs, default_arch, asset_download_manager, tensor_loader
    )

    registry.register(handler.family, handler)

    register_laser2_configs(context)

    # mutox
    configs = context.get_config_registry(MutoxConfig)
    default_arch = "mutox"
    handler = MutoxModelHandler(
        configs, default_arch, asset_download_manager, tensor_loader
    )
    registry.register(handler.family, handler)
    register_mutox_configs(context)

    # SONAR Speech Encoder
    configs = context.get_config_registry(SonarSpeechEncoderConfig)

    default_arch = "english"

    handler = SonarSpeechEncoderHandler(
        configs, default_arch, asset_download_manager, tensor_loader
    )

    registry.register(handler.family, handler)

    register_sonar_speech_encoder_configs(context)

    # SONAR Text Encoder
    configs = context.get_config_registry(SonarTextEncoderConfig)

    default_arch = "basic"

    handler = SonarTextEncoderHandler(
        configs, default_arch, asset_download_manager, tensor_loader
    )

    registry.register(handler.family, handler)

    register_sonar_text_encoder_configs(context)

    # SONAR Text Decoder
    configs = context.get_config_registry(SonarTextDecoderConfig)

    default_arch = "basic"

    handler = SonarTextDecoderHandler(
        configs, default_arch, asset_download_manager, tensor_loader
    )

    registry.register(handler.family, handler)

    register_sonar_text_decoder_configs(context)


def _register_text_tokenizers(context: RuntimeContext) -> None:
    registry = context.get_registry(TextTokenizerHandler)

    # Laser2
    handler = Laser2TokenizerHandler(context.asset_download_manager)

    registry.register(handler.family, handler)
