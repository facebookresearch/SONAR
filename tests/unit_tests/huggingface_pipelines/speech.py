import pytest
import numpy as np
import torch
import tempfile
import soundfile as sf
from unittest.mock import Mock, patch
from typing import Dict, Any

from huggingface_pipelines.speech import (
    HFSpeechToEmbeddingPipelineConfig,
    HFSpeechToEmbeddingPipeline,
    SpeechToEmbeddingModelPipeline
)


@pytest.fixture
def mock_speech_to_embedding_model():
    class MockSpeechToEmbeddingModelPipeline(SpeechToEmbeddingModelPipeline):

        def __init__(self, encoder, device, fbank_dtype):
            pass

        def predict(self, input, batch_size, n_parallel, pad_idx):
            return [torch.tensor([[0.1, 0.2, 0.3]]) for _ in input]

    with patch('huggingface_pipelines.speech.SpeechToEmbeddingModelPipeline', MockSpeechToEmbeddingModelPipeline):
        yield


class MockModelWithDifferentSizes:
    def __init__(self, encoder, device, fbank_dtype):
        pass

    def predict(self, input, batch_size, n_parallel, pad_idx):
        return [torch.tensor([[0.1, 0.2, 0.3]]), torch.tensor([[0.4, 0.5], [0.6, 0.7]])]


@pytest.fixture
def pipeline_config():
    return HFSpeechToEmbeddingPipelineConfig(
        encoder_model="test_encoder",
        device="cpu",
        batch_size=2,
        audio_column="audio",
        columns=["test"],
        output_path="test",
        output_column_suffix="test_embeddings"
    )


@pytest.fixture
def sample_audio_data():
    return {
        'array': np.random.rand(16000),
        'sampling_rate': 16000
    }


def test_pipeline_initialization(pipeline_config, mock_speech_to_embedding_model):
    pipeline = HFSpeechToEmbeddingPipeline(pipeline_config)
    assert pipeline.config == pipeline_config
    assert isinstance(pipeline.model, SpeechToEmbeddingModelPipeline)


def test_process_batch_valid_input(pipeline_config, mock_speech_to_embedding_model, sample_audio_data):
    pipeline = HFSpeechToEmbeddingPipeline(pipeline_config)
    batch = {"audio": [sample_audio_data, sample_audio_data]}
    result = pipeline.process_batch(batch)
    assert "audio_embedding" in result
    assert isinstance(result["audio_embedding"], np.ndarray)
    assert result["audio_embedding"].shape == (
        2, 3)  # 2 samples, 3 embedding dimensions


def test_process_batch_empty_input(pipeline_config, mock_speech_to_embedding_model):
    pipeline = HFSpeechToEmbeddingPipeline(pipeline_config)
    batch = {"audio": []}
    result = pipeline.process_batch(batch)
    assert "audio_embedding" not in result


def test_process_batch_invalid_audio_data(pipeline_config, mock_speech_to_embedding_model):
    pipeline = HFSpeechToEmbeddingPipeline(pipeline_config)
    batch = {"audio": [{"invalid": "data"}]}
    result = pipeline.process_batch(batch)
    assert "audio_embedding" not in result


def test_process_batch_mixed_valid_invalid_data(pipeline_config, mock_speech_to_embedding_model, sample_audio_data):
    pipeline = HFSpeechToEmbeddingPipeline(pipeline_config)
    batch = {"audio": [sample_audio_data, {
        "invalid": "data"}, sample_audio_data]}
    result = pipeline.process_batch(batch)
    assert "audio_embedding" in result
    assert isinstance(result["audio_embedding"], np.ndarray)
    # 2 valid samples, 3 embedding dimensions
    assert result["audio_embedding"].shape == (2, 3)


@patch('soundfile.write')
@patch('os.unlink')
def test_temporary_file_creation_and_deletion(mock_unlink, mock_sf_write, pipeline_config, mock_speech_to_embedding_model, sample_audio_data):
    pipeline = HFSpeechToEmbeddingPipeline(pipeline_config)
    batch = {"audio": [sample_audio_data]}

    with patch('tempfile.NamedTemporaryFile') as mock_temp_file:
        mock_temp_file.return_value.name = 'test.wav'
        pipeline.process_batch(batch)

        mock_temp_file.assert_called_once()
        mock_sf_write.assert_called_once_with(
            'test.wav', sample_audio_data['array'], sample_audio_data['sampling_rate'])
        mock_temp_file.return_value.close.assert_called_once()
        mock_unlink.assert_called_once_with('test.wav')


@patch('huggingface_pipelines.speech.SpeechToEmbeddingModelPipeline')
def test_error_handling_in_model_predict(mock_predict, pipeline_config, sample_audio_data):
    mock_predict.return_value.predict.side_effect = Exception(
        "Model prediction error")
    pipeline = HFSpeechToEmbeddingPipeline(pipeline_config)
    batch = {"audio": [sample_audio_data]}

    with pytest.raises(Exception, match="Model prediction error"):
        pipeline.process_batch(batch)


def test_config_with_overwrites():
    base_config = HFSpeechToEmbeddingPipelineConfig(
        encoder_model="base_encoder",
        device="cpu",
        batch_size=2,
        audio_column="audio",
        columns=["test"],
        output_path="test",
        output_column_suffix="test_embeddings",
    )
    overwrites = {
        "encoder_model": "new_encoder",
        "batch_size": 4,
        "fbank_dtype": torch.float64
    }
    new_config = base_config.with_overwrites(overwrites)

    assert new_config.encoder_model == "new_encoder"
    assert new_config.batch_size == 4
    assert new_config.fbank_dtype == torch.float64
    assert new_config.device == "cpu"  # Unchanged


def test_process_large_batch(pipeline_config, mock_speech_to_embedding_model, sample_audio_data):
    pipeline = HFSpeechToEmbeddingPipeline(pipeline_config)
    large_batch = {"audio": [sample_audio_data] * 100}  # 100 audio samples
    result = pipeline.process_batch(large_batch)
    assert "audio_embedding" in result
    assert isinstance(result["audio_embedding"], np.ndarray)
    assert result["audio_embedding"].shape == (
        100, 3)  # 100 samples, 3 embedding dimensions
