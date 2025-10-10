# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch
from fairseq2.models.nllb.hub import get_nllb_tokenizer_hub
from torch.testing import assert_close

from sonar.inference_pipelines import (
    SpeechInferenceParams,
    SpeechToEmbeddingPipeline,
    SpeechToTextPipeline,
)
from sonar.models.sonar_speech import get_sonar_speech_encoder_hub
from sonar.models.sonar_text import get_sonar_text_decoder_hub
from sonar.models.sonar_translation import SonarEncoderDecoderModel

DEVICE = torch.device("cpu")
DATA_DIR = Path(__file__).parent.joinpath("data")


class TestSonarTextClass:
    encoder_hub = get_sonar_speech_encoder_hub()
    encoder = encoder_hub.load_model("sonar_speech_encoder_eng", device=DEVICE)
    encoder.eval()
    # from sonar import setup_logging_here
    # setup_logging_here()
    tokenizer_hub = get_nllb_tokenizer_hub()
    tokenizer = tokenizer_hub.load_tokenizer("text_sonar_basic_encoder")

    decoder_hub = get_sonar_text_decoder_hub()
    decoder = decoder_hub.load_model("text_sonar_basic_decoder", device=DEVICE)
    decoder.eval()

    params = SpeechInferenceParams(
        data_file=DATA_DIR.joinpath("audio_ref.tsv"),
        audio_root_dir=DATA_DIR.joinpath("audio_files"),
        audio_path_index=1,
        target_lang="fra_Latn",
        batch_size=4,
        pad_idx=0,
        device=DEVICE,
        fbank_dtype=torch.float32,
        n_parallel=1,
    )

    embedding = torch.load(DATA_DIR.joinpath("speech_embedding.pt"))
    """Binary containing a batch of expected embeddings for input features.
    *Shape* :math:`(N,M)` where :math:`N` is the batch size, and :math:`M` the
    embedding size (i.e. model dimension)
    """

    def test_speech_to_text_pipeline(self) -> None:
        encoder_decoder = SonarEncoderDecoderModel(self.encoder, self.decoder)
        dp_builder = SpeechToTextPipeline(encoder_decoder, self.tokenizer)
        dp = dp_builder.build_pipeline(self.params)
        it = iter(dp)
        actual = next(it)

        expected = [
            "Les rapports de la télévision montrent une fumée blanche provenant de l'usine.",
            "Ces couples peuvent décider de faire un plan d'adoption pour leur bébé.",
        ]

        assert actual["audio"]["data"] == expected

    def test_speech_to_embedding_pipeline(self) -> None:
        dp_builder = SpeechToEmbeddingPipeline(self.encoder)
        dp = dp_builder.build_pipeline(self.params)

        it = iter(dp)
        actual = next(it)

        expected = self.embedding
        assert_close(actual["audio"]["data"].sentence_embeddings, expected)
