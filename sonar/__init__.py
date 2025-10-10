# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SONAR provides a set of speech and text encoders for multilingual, multimodal semantic embedding.
"""

from types import NoneType

from fairseq2.composition.assets import register_package_assets
from fairseq2.composition.models import register_model_family
from fairseq2.composition.tokenizers import register_tokenizer_family
from fairseq2.runtime.dependency import DependencyContainer

from sonar.models.blaser import (
    BLASER_FAMILY,
    BlaserConfig,
    BlaserModel,
    _convert_blaser_checkpoint,
    _create_blaser_model,
    _register_blaser_configs,
)
from sonar.models.laser2_text import (
    LASER2_FAMILY,
    Laser2Config,
    LaserLstmEncoder,
    _convert_laser2_checkpoint,
    _create_laser2_model,
    _load_laser2_tokenizer,
    _register_laser2_configs,
)
from sonar.models.laser2_text.tokenizer import Laser2Tokenizer
from sonar.models.mutox import (
    MUTOX_FAMILY,
    MutoxClassifier,
    MutoxConfig,
    _convert_mutox_checkpoint,
    _create_mutox_model,
    _register_mutox_configs,
)
from sonar.models.sonar_speech import (
    SONAR_SPEECH_FAMILY,
    SonarSpeechEncoderConfig,
    SonarSpeechEncoderModel,
    _convert_sonar_speech_checkpoint,
    _create_sonar_speech_encoder_model,
    _register_sonar_speech_encoder_configs,
)
from sonar.models.sonar_text import (
    SONAR_TEXT_DECODER_FAMILY,
    SONAR_TEXT_ENCODER_FAMILY,
    ConditionalTransformerDecoderModel,
    SonarTextDecoderConfig,
    SonarTextEncoderConfig,
    SonarTextTransformerEncoderModel,
    _convert_sonar_text_decoder_checkpoint,
    _convert_sonar_text_encoder_checkpoint,
    _create_sonar_text_decoder_model,
    _create_sonar_text_encoder_model,
    _register_sonar_text_decoder_configs,
    _register_sonar_text_encoder_configs,
)

__version__ = "0.5.0"


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    # Make sure that the default fairseq2 asset store can resolve cards under
    # the directory <sonar>/cards.
    register_package_assets(container, "sonar.cards")

    _register_models(container)

    _register_text_tokenizers(container)


def _register_models(container: DependencyContainer) -> None:
    # Blaser
    register_model_family(
        container,
        BLASER_FAMILY,
        kls=BlaserModel,
        config_kls=BlaserConfig,
        factory=_create_blaser_model,
        state_dict_converter=_convert_blaser_checkpoint,
    )

    _register_blaser_configs(container)

    # Laser2
    register_model_family(
        container,
        LASER2_FAMILY,
        kls=LaserLstmEncoder,
        config_kls=Laser2Config,
        factory=_create_laser2_model,
        state_dict_converter=_convert_laser2_checkpoint,
    )

    _register_laser2_configs(container)

    # mutox
    register_model_family(
        container,
        MUTOX_FAMILY,
        kls=MutoxClassifier,
        config_kls=MutoxConfig,
        factory=_create_mutox_model,
        state_dict_converter=_convert_mutox_checkpoint,
    )

    _register_mutox_configs(container)

    # SONAR Speech Encoder
    register_model_family(
        container,
        SONAR_SPEECH_FAMILY,
        kls=SonarSpeechEncoderModel,
        config_kls=SonarSpeechEncoderConfig,
        factory=_create_sonar_speech_encoder_model,
        state_dict_converter=_convert_sonar_speech_checkpoint,
    )

    _register_sonar_speech_encoder_configs(container)

    # SONAR Text Encoder
    register_model_family(
        container,
        SONAR_TEXT_ENCODER_FAMILY,
        kls=SonarTextTransformerEncoderModel,
        config_kls=SonarTextEncoderConfig,
        factory=_create_sonar_text_encoder_model,
        state_dict_converter=_convert_sonar_text_encoder_checkpoint,
    )

    _register_sonar_text_encoder_configs(container)

    # SONAR Text Decoder
    register_model_family(
        container,
        SONAR_TEXT_DECODER_FAMILY,
        kls=ConditionalTransformerDecoderModel,
        config_kls=SonarTextDecoderConfig,
        state_dict_converter=_convert_sonar_text_decoder_checkpoint,
        factory=_create_sonar_text_decoder_model,
    )

    _register_sonar_text_decoder_configs(container)


def _register_text_tokenizers(container: DependencyContainer) -> None:
    # Laser2
    register_tokenizer_family(
        container,
        LASER2_FAMILY,
        kls=Laser2Tokenizer,
        config_kls=NoneType,
        loader=_load_laser2_tokenizer,
    )
