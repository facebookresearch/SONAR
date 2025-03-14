# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Union

import torch
from fairseq2.data import DataPipelineBuilder
from fairseq2.typing import Device

from sonar.inference_pipelines.speech import (
    AudioToFbankDataPipelineBuilder,
    SpeechInferenceParams,
    SpeechInferencePipeline,
)
from sonar.inference_pipelines.utils import extract_sequence_batch
from sonar.models.encoder_model import SonarEncoderModel
from sonar.models.mutox import MutoxClassifier, get_mutox_model_hub
from sonar.models.sonar_speech import get_sonar_speech_encoder_hub

CPU_DEVICE = torch.device("cpu")


class MutoxSpeechClassifierPipeline(SpeechInferencePipeline):
    model: SonarEncoderModel

    def __init__(
        self,
        mutox_classifier: Union[str, MutoxClassifier],
        encoder: Union[str, SonarEncoderModel],
        device: Device = CPU_DEVICE,
    ) -> None:
        if isinstance(encoder, str):
            self.model = self.load_model_from_name(
                "sonar_mutox", encoder, device=device
            )  # type: ignore
        else:
            self.model = encoder

        super().__init__()

        self.model.to(device).eval()

        if isinstance(mutox_classifier, str):
            self.mutox_classifier = get_mutox_model_hub().load(
                mutox_classifier,
                device=device,
            )
        else:
            self.mutox_classifier = mutox_classifier

        self.mutox_classifier.to(device).eval()

    @classmethod
    def load_model_from_name(
        cls,
        mutox_classifier_name: str,
        encoder_name: str,
        device: Device = CPU_DEVICE,
    ) -> "MutoxSpeechClassifierPipeline":
        encoder_hub = get_sonar_speech_encoder_hub()
        encoder = encoder_hub.load(encoder_name, device=device)
        mutox_classifier = get_mutox_model_hub().load(
            mutox_classifier_name,
            device=device,
        )
        return cls(mutox_classifier=mutox_classifier, encoder=encoder, device=device)

    def prebuild_pipeline(self, context: SpeechInferenceParams) -> DataPipelineBuilder:
        audio_to_fbank_dp_builder = AudioToFbankDataPipelineBuilder()
        pipeline_builder = (
            audio_to_fbank_dp_builder.prebuild_pipeline(context)
            .map(
                lambda fbank: extract_sequence_batch(fbank, context.device),
                selector="audio.data.fbank",
            )
            .map(self.run_inference, selector="audio.data")
        )
        return pipeline_builder.map(self._run_classifier, selector="audio.data")

    @torch.inference_mode()
    def run_inference(self, fbank: torch.Tensor) -> dict:
        """Runs the encoder model on the extracted FBANK features."""
        return {"sentence_embeddings": self.model(fbank)}

    @torch.inference_mode()
    def _run_classifier(self, data: dict):
        sentence_embeddings = data.get("sentence_embeddings")
        if sentence_embeddings is None:
            raise ValueError("Missing sentence embeddings in the data.")
        return self.mutox_classifier(sentence_embeddings)
