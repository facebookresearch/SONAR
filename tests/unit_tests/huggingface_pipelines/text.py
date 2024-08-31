import pytest
import numpy as np
from unittest.mock import patch
import torch
from huggingface_pipelines.text import (
    TextToEmbeddingPipelineConfig,
    EmbeddingToTextPipelineConfig,
    HFTextToEmbeddingPipeline,
    HFEmbeddingToTextPipeline,
)


class MockTextToEmbeddingModelPipeline:
    def __init__(self, encoder, tokenizer, device):
        pass

    def predict(self, texts, source_lang, batch_size, max_seq_len):
        # Create a tensor with shape (len(texts), 3)
        return torch.tensor([[0.1, 0.2, 0.3] for _ in texts], dtype=torch.float32)


class MockEmbeddingToTextModelPipeline:
    def __init__(self, decoder, tokenizer, device):
        pass

    def predict(self, embeddings, target_lang, batch_size):
        return ["decoded_text" for _ in range(len(embeddings))]


@pytest.fixture
def text_to_embedding_config():
    return TextToEmbeddingPipelineConfig(
        encoder_model="test_encoder",
        columns=["text"],
        output_column_suffix="embedding",
        batch_size=2,
        device="cpu",
        source_lang="eng_Latn",
        output_path="test"
    )


@pytest.fixture
def embedding_to_text_config():
    return EmbeddingToTextPipelineConfig(
        decoder_model="test_decoder",
        columns=["embedding"],
        output_column_suffix="text",
        batch_size=2,
        device="cpu",
        target_lang="eng_Latn",
        output_path="test"
    )


@pytest.fixture
def mock_text_to_embedding_model():
    with patch('huggingface_pipelines.text.TextToEmbeddingModelPipeline', MockTextToEmbeddingModelPipeline):
        yield


@pytest.fixture
def mock_embedding_to_text_model():
    with patch('huggingface_pipelines.text.EmbeddingToTextModelPipeline', MockEmbeddingToTextModelPipeline):
        yield


def test_text_to_embedding_pipeline_initialization(text_to_embedding_config, mock_text_to_embedding_model):
    pipeline = HFTextToEmbeddingPipeline(text_to_embedding_config)
    assert pipeline.config == text_to_embedding_config
    assert isinstance(pipeline.t2vec_model, MockTextToEmbeddingModelPipeline)


def test_embedding_to_text_pipeline_initialization(embedding_to_text_config, mock_embedding_to_text_model):
    pipeline = HFEmbeddingToTextPipeline(embedding_to_text_config)
    assert pipeline.config == embedding_to_text_config
    assert isinstance(pipeline.t2t_model, MockEmbeddingToTextModelPipeline)


def test_text_to_embedding_process_batch(text_to_embedding_config, mock_text_to_embedding_model):
    pipeline = HFTextToEmbeddingPipeline(text_to_embedding_config)
    batch = {"text": [["Hello", "World"], ["Test", "Sentence"]]}
    result = pipeline.process_batch(batch)
    assert "text_embedding" in result
    assert len(result["text_embedding"]) == 2
    assert all(isinstance(item, np.ndarray)
               for sublist in result["text_embedding"] for item in sublist)


def test_embedding_to_text_process_batch(embedding_to_text_config, mock_embedding_to_text_model):
    pipeline = HFEmbeddingToTextPipeline(embedding_to_text_config)
    batch = {"embedding": [
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9]]]}
    result = pipeline.process_batch(batch)
    assert "embedding_text" in result
    assert len(result["embedding_text"]) == 2
    assert all(isinstance(text, list) for text in result["embedding_text"])
    assert all(isinstance(item, str)
               for sublist in result["embedding_text"] for item in sublist)


@pytest.mark.parametrize("invalid_batch", [
    {"text": "Not a list"},
    {"text": [1, 2, 3]},
])
def test_text_to_embedding_invalid_input(text_to_embedding_config, mock_text_to_embedding_model, invalid_batch):
    pipeline = HFTextToEmbeddingPipeline(text_to_embedding_config)
    with pytest.raises(ValueError):
        pipeline.process_batch(invalid_batch)


@pytest.mark.parametrize("invalid_batch", [
    {"embedding": "Not a list"},
    {"embedding": [1, 2, 3]},
])
def test_embedding_to_text_invalid_input(embedding_to_text_config, mock_embedding_to_text_model, invalid_batch):
    pipeline = HFEmbeddingToTextPipeline(embedding_to_text_config)
    with pytest.raises(ValueError):
        pipeline.process_batch(invalid_batch)


def test_text_to_embedding_large_batch(text_to_embedding_config, mock_text_to_embedding_model):
    pipeline = HFTextToEmbeddingPipeline(text_to_embedding_config)
    large_batch = {"text": [["Hello"] * 100, ["World"] * 100]}
    result = pipeline.process_batch(large_batch)
    assert "text_embedding" in result
    assert len(result["text_embedding"]) == 2
    assert all(len(emb) == 100 for emb in result["text_embedding"])


def test_embedding_to_text_large_batch(embedding_to_text_config, mock_embedding_to_text_model):
    pipeline = HFEmbeddingToTextPipeline(embedding_to_text_config)
    large_batch = {"embedding": [
        [np.array([0.1, 0.2, 0.3])] * 100, [np.array([0.4, 0.5, 0.6])] * 100]}
    result = pipeline.process_batch(large_batch)
    assert "embedding_text" in result
    assert len(result["embedding_text"]) == 2
    assert all(len(text) == 100 for text in result["embedding_text"])


def test_text_to_embedding_process_batch_complicated(text_to_embedding_config, mock_text_to_embedding_model):
    pipeline = HFTextToEmbeddingPipeline(text_to_embedding_config)
    batch = {
        "text": [
            ["The quick brown fox jumps over the lazy dog."],
            ["This is a test sentence.", "And another one."],
            [],
            ["Short.", "Medium length sentence.", "Longer sentence with more words.",
                "Very long sentence that goes on and on with many more words than the others."]
        ]
    }
    result = pipeline.process_batch(batch)

    assert "text_embedding" in result
    assert len(result["text_embedding"]) == 4
    assert all(isinstance(item, np.ndarray)
               for sublist in result["text_embedding"] for item in sublist)
    assert len(result["text_embedding"][0]) == 1
    assert len(result["text_embedding"][1]) == 2
    assert len(result["text_embedding"][2]) == 0
    assert len(result["text_embedding"][3]) == 4


def test_embedding_to_text_process_batch_variable_length_lists(embedding_to_text_config, mock_embedding_to_text_model):
    pipeline = HFEmbeddingToTextPipeline(embedding_to_text_config)
    batch = {
        "embedding": [
            [np.array([0.1, 0.2, 0.3])],
            [np.array([0.4, 0.5, 0.6]), np.array([0.7, 0.8, 0.9])],
            [],
            [np.array([0.1, 0.1, 0.1]), np.array([0.2, 0.2, 0.2]),
                np.array([0.3, 0.3, 0.3]), np.array([0.4, 0.4, 0.4])]
        ]
    }
    result = pipeline.process_batch(batch)

    assert "embedding_text" in result
    assert len(result["embedding_text"]) == 4
    assert all(isinstance(text, list) for text in result["embedding_text"])
    assert all(isinstance(item, str)
               for sublist in result["embedding_text"] for item in sublist)
    assert len(result["embedding_text"][0]) == 1
    assert len(result["embedding_text"][1]) == 2
    assert len(result["embedding_text"][2]) == 0
    assert len(result["embedding_text"][3]) == 4

