from typing import List, Union

import numpy as np
import pytest

from huggingface_pipelines.text import (
    EmbeddingToTextPipelineConfig,
    HFEmbeddingToTextPipeline,
    HFTextToEmbeddingPipeline,
    TextToEmbeddingPipelineConfig,
)


@pytest.fixture
def text_to_embedding_config():
    return TextToEmbeddingPipelineConfig(
        encoder_model="text_sonar_basic_encoder",
        columns=["text"],
        output_column_suffix="embedding",
        batch_size=2,
        device="cpu",
        source_lang="eng_Latn",
        output_path="test",
    )


@pytest.fixture
def embedding_to_text_config():
    return EmbeddingToTextPipelineConfig(
        decoder_model="text_sonar_basic_decoder",
        columns=["embedding"],
        output_column_suffix="text",
        batch_size=2,
        device="cpu",
        target_lang="eng_Latn",
        output_path="test",
    )


def test_text_to_embedding_process_batch(text_to_embedding_config):
    pipeline = HFTextToEmbeddingPipeline(text_to_embedding_config)
    batch = {"text": [["Hello", "World"], ["Test", "Sentence"]]}
    result = pipeline.process_batch(batch)
    assert "text_embedding" in result
    assert len(result["text_embedding"]) == 2
    assert all(isinstance(item, np.ndarray) for item in result["text_embedding"])


def test_embedding_to_text_process_batch(embedding_to_text_config):
    pipeline = HFEmbeddingToTextPipeline(embedding_to_text_config)

    embedding_dim = 1024
    num_embeddings = 4

    embeddings: List[np.ndarray] = [
        np.random.rand(embedding_dim).astype(np.float32) for _ in range(num_embeddings)
    ]

    batch = {"embedding": embeddings}

    result = pipeline.process_batch(batch)
    assert "embedding_text" in result
    assert len(result["embedding_text"]) == 1
    assert isinstance(result["embedding_text"][0], list)
    assert all(isinstance(item, str) for item in result["embedding_text"][0])


@pytest.mark.parametrize(
    "invalid_batch",
    [
        {"text": "Not a list"},
        {"text": [1, 2, 3]},
    ],
)
def test_text_to_embedding_invalid_input(text_to_embedding_config, invalid_batch):
    pipeline = HFTextToEmbeddingPipeline(text_to_embedding_config)
    with pytest.raises(ValueError):
        pipeline.process_batch(invalid_batch)


@pytest.mark.parametrize(
    "invalid_batch",
    [
        {"embedding": "Not a list"},
        {"embedding": [1, 2, 3]},
    ],
)
def test_embedding_to_text_invalid_input(embedding_to_text_config, invalid_batch):
    pipeline = HFEmbeddingToTextPipeline(embedding_to_text_config)
    with pytest.raises(ValueError):
        pipeline.process_batch(invalid_batch)


def test_text_to_embedding_to_text_pipeline(
    text_to_embedding_config, embedding_to_text_config
):
    text_to_embedding_pipeline = HFTextToEmbeddingPipeline(text_to_embedding_config)
    embedding_to_text_pipeline = HFEmbeddingToTextPipeline(embedding_to_text_config)

    test_cases: List[Union[List[str], List[List[str]]]] = [
        ["Hello, world!", "This is a test.", "Multiple sentences here."],
        [
            ["Hello, world!"],
            ["This is a test.", "Multiple sentences here."],
            ["Short.", "Medium length sentence.", "Longer sentence with more words."],
        ],
    ]

    for original_texts in test_cases:
        batch = {"text": original_texts}
        encoded_result = text_to_embedding_pipeline.process_batch(batch)
        assert "text_embedding" in encoded_result
        assert len(encoded_result["text_embedding"]) == len(original_texts)

        decode_batch = {"embedding": encoded_result["text_embedding"]}
        decoded_result = embedding_to_text_pipeline.process_batch(decode_batch)
        assert "embedding_text" in decoded_result
        assert len(decoded_result["embedding_text"]) == 1

        decoded_texts = decoded_result["embedding_text"][0]
        assert len(decoded_texts) == len(original_texts)

        for original, decoded in zip(original_texts, decoded_texts):
            if isinstance(original, list):
                # Join list of strings into a single string
                original = " ".join(original)

            assert len(decoded.split()) > 0


def test_text_to_embedding_to_text_pipeline_single_sentence(
    text_to_embedding_config, embedding_to_text_config
):
    text_to_embedding_pipeline = HFTextToEmbeddingPipeline(text_to_embedding_config)
    embedding_to_text_pipeline = HFEmbeddingToTextPipeline(embedding_to_text_config)

    test_cases: List[Union[List[str], List[List[str]]]] = [
        ["The quick brown fox jumps over the lazy dog."],
        [["The quick brown fox jumps over the lazy dog."]],
    ]

    for original_text in test_cases:
        batch = {"text": original_text}
        encoded_result = text_to_embedding_pipeline.process_batch(batch)
        assert "text_embedding" in encoded_result
        assert len(encoded_result["text_embedding"]) == 1

        decode_batch = {"embedding": encoded_result["text_embedding"]}
        decoded_result = embedding_to_text_pipeline.process_batch(decode_batch)
        assert "embedding_text" in decoded_result
        assert len(decoded_result["embedding_text"]) == 1

        original = original_text[0]
        decoded = decoded_result["embedding_text"][0][0]

        if isinstance(original, list):
            original = original[0]

        assert len(decoded.split()) > 0
        assert any(word.lower() in decoded.lower() for word in original.split())
