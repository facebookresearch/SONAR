from typing import Any, Dict, List, Sequence, Union
from unittest.mock import patch

import numpy as np
import pytest
import torch

from huggingface_pipelines.audio import (  # types: ignore
    HFAudioToEmbeddingPipeline,
    HFAudioToEmbeddingPipelineConfig,
    SpeechToEmbeddingModelPipeline,
)


@pytest.fixture
class MockSpeechToEmbeddingModelPipeline(SpeechToEmbeddingModelPipeline):
    def __init__(self, encoder: Any, device: Any, fbank_dtype: Any):
        pass

    def predict(
        self,
        input: Union[Sequence[str], Sequence[torch.Tensor]],
        batch_size: int = 32,
        n_parallel: int = 1,
        pad_idx: int = 0,
        n_prefetched_batches: int = 1,
        progress_bar: bool = False,
    ) -> torch.Tensor:
        return torch.stack([torch.tensor([[0.1, 0.2, 0.3]]) for _ in input])


@pytest.fixture
def pipeline_config():
    return HFAudioToEmbeddingPipelineConfig(
        encoder_model="test_encoder",
        device="cpu",
        batch_size=2,
        audio_column="audio",
        columns=["test"],
        output_path="test",
        output_column_suffix="test_embeddings",
    )


@pytest.fixture
def sample_audio_data():
    return {"array": np.random.rand(16000), "sampling_rate": 16000}


def test_pipeline_initialization(pipeline_config, mock_speech_to_embedding_model):
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    assert pipeline.config == pipeline_config
    assert isinstance(pipeline.model, SpeechToEmbeddingModelPipeline)


def test_process_batch_valid_input(
    pipeline_config, mock_speech_to_embedding_model, sample_audio_data
):
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    batch: Dict[str, List[Dict[str, Any]]] = {
        "audio": [sample_audio_data, sample_audio_data]
    }
    result = pipeline.process_batch(batch)
    assert "audio_embedding" in result
    assert isinstance(result["audio_embedding"], np.ndarray)
    assert result["audio_embedding"].shape == (
        2,
        3,
    )  # 2 samples, 3 embedding dimensions


def test_process_batch_empty_input(pipeline_config, mock_speech_to_embedding_model):
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    batch: Dict[str, List[Dict[str, Any]]] = {"audio": []}
    result = pipeline.process_batch(batch)
    assert "audio_embedding" not in result


def test_process_batch_invalid_audio_data(
    pipeline_config, mock_speech_to_embedding_model
):
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    batch: Dict[str, List[Dict[str, Any]]] = {"audio": [{"invalid": "data"}]}
    result = pipeline.process_batch(batch)
    assert "audio_embedding" not in result


def test_process_batch_mixed_valid_invalid_data(
    pipeline_config, mock_speech_to_embedding_model, sample_audio_data
):
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    batch: Dict[str, List[Dict[str, Any]]] = {
        "audio": [sample_audio_data, {"invalid": "data"}, sample_audio_data]
    }
    result = pipeline.process_batch(batch)
    assert "audio_embedding" in result
    assert isinstance(result["audio_embedding"], np.ndarray)
    # 2 valid samples, 3 embedding dimensions
    assert result["audio_embedding"].shape == (2, 3)


@patch("huggingface_pipelines.speech.SpeechToEmbeddingModelPipeline")
def test_error_handling_in_model_predict(
    mock_predict, pipeline_config, sample_audio_data
):
    mock_predict.return_value.predict.side_effect = Exception("Model prediction error")
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    batch: Dict[str, List[Dict[str, Any]]] = {"audio": [sample_audio_data]}

    with pytest.raises(Exception, match="Model prediction error"):
        pipeline.process_batch(batch)


def test_process_large_batch(
    pipeline_config, mock_speech_to_embedding_model, sample_audio_data
):
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    large_batch: Dict[str, List[Dict[str, Any]]] = {
        "audio": [sample_audio_data] * 100
    }  # 100 audio samples
    result = pipeline.process_batch(large_batch)
    assert "audio_embedding" in result
    assert isinstance(result["audio_embedding"], np.ndarray)
    assert result["audio_embedding"].shape == (
        100,
        3,
    )  # 100 samples, 3 embedding dimensions
