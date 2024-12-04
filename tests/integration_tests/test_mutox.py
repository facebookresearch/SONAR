# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.mutox.loader import load_mutox_model


@pytest.mark.parametrize(
    "input_texts, source_lang, expected_outputs",
    [
        (
            ["De peur que le pays ne se prostitue et ne se remplisse de crimes."],
            "fra_Latn",
            [-19.7812],
        ),
        (
            ["She worked hard and made a significant contribution to the team."],
            "eng_Latn",
            [-53.5938],
        ),
        (
            [
                "El no tiene ni el más mínimo talento, todo lo que ha logrado ha sido gracias a sobornos y manipulaciones."
            ],
            "spa_Latn",
            [-21.4062],
        ),
    ],
)
def test_sonar_mutox_classifier_integration(input_texts, source_lang, expected_outputs):
    """Integration test to compare classifier outputs with expected values."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    t2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
    )

    classifier = load_mutox_model("sonar_mutox", device=device, dtype=dtype).eval()

    with torch.inference_mode():
        embeddings = t2vec_model.predict(input_texts, source_lang=source_lang)
        outputs = classifier(embeddings.to(device).to(dtype)).squeeze()

    if outputs.dim() == 0:
        outputs = [outputs.item()]
    else:
        outputs = outputs.tolist()

    # Compare the outputs to expected values within a small tolerance
    for output, expected in zip(outputs, expected_outputs):
        assert abs(output - expected) < 0.1, (
            f"Expected output {expected}, but got {output}. "
            "Outputs should be close to expected values."
        )


@pytest.mark.parametrize(
    "input_texts, source_lang, expected_probabilities",
    [
        (
            ["De peur que le pays ne se prostitue et ne se remplisse de crimes."],
            "fra_Latn",
            [0.0],
        ),
        (
            [
                "Dammit, that was a terrible launch, it will piss the director and make the mission fail."
            ],
            "eng_Latn",
            [0.23],
        ),
        (
            [
                "El no tiene ni el más mínimo talento, todo lo que ha logrado ha sido gracias a sobornos y manipulaciones."
            ],
            "spa_Latn",
            [0.0],
        ),
    ],
)
def test_sonar_mutox_classifier_probability_integration(
    input_texts, source_lang, expected_probabilities
):
    """Integration test to verify classifier output probabilities."""

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    t2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
    )

    classifier = load_mutox_model("sonar_mutox", device=device, dtype=dtype).eval()

    for text, lang, expected_prob in zip(
        input_texts, [source_lang] * len(input_texts), expected_probabilities
    ):
        with torch.inference_mode():
            emb = t2vec_model.predict([text], source_lang=lang)

            prob = classifier(emb.to(device).to(dtype), output_prob=True)

            assert abs(prob.item() - expected_prob) < 0.01, (
                f"Expected probability {expected_prob}, but got {prob.item()}. "
                "Output probability should be within a reasonable range."
            )
