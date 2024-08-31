from typing import Any, Dict, List, Optional, Sequence, Generator

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from huggingface_pipelines.audio import HFAudioToEmbeddingPipeline  # type: ignore
from huggingface_pipelines.audio import (
    HFAudioToEmbeddingPipelineConfig,  # type: ignore
)
from huggingface_pipelines.audio import SpeechToEmbeddingModelPipeline  # type: ignore
from torch import Tensor


@pytest.fixture
def mock_speech_to_embedding_model() -> Generator[None, None, None]:
    class MockSpeechToEmbeddingModelPipeline(SpeechToEmbeddingModelPipeline):
        def __init__(self, encoder: Any, device: str, fbank_dtype: Any) -> None:
            pass

        def predict(
            self,
            input: Sequence[str] | Sequence[Tensor],
            batch_size: int = 32,
            n_parallel: int = 1,
            pad_idx: int = 0,
            n_prefetched_batches: int = 2,
            progress_bar: bool = False
        ) -> Tensor:
            return torch.tensor([[0.1, 0.2, 0.3]] * len(input))

    with patch(
        "huggingface_pipelines.speech.SpeechToEmbeddingModelPipeline",
        MockSpeechToEmbeddingModelPipeline,
    ):
        yield


@pytest.fixture
def pipeline_config() -> HFAudioToEmbeddingPipelineConfig:
    return HFAudioToEmbeddingPipelineConfig(
        encoder_model="sonar_speech_encoder_eng",
        device="cpu",
        batch_size=2,
        columns=["audio"],
        output_path="test",
        output_column_suffix="embedding",
    )


@pytest.fixture
def sample_audio_data() -> Dict[str, Any]:
    return {"array": np.random.rand(16000), "sampling_rate": 16000}


@pytest.fixture
def complex_audio_data() -> Dict[str, Dict[str, Any]]:
    return {
        "short_audio": {"array": np.random.rand(8000), "sampling_rate": 16000},
        "long_audio": {"array": np.random.rand(32000), "sampling_rate": 16000},
        "multi_channel": {"array": np.random.rand(2, 16000), "sampling_rate": 16000},
        "high_sample_rate": {"array": np.random.rand(48000), "sampling_rate": 48000},
        "low_sample_rate": {"array": np.random.rand(8000), "sampling_rate": 8000},
        "float64_audio": {
            "array": np.random.rand(16000).astype(np.float64),
            "sampling_rate": 16000,
        },
        "int16_audio": {
            "array": (np.random.rand(16000) * 32767).astype(np.int16),
            "sampling_rate": 16000,
        },
    }


def test_pipeline_initialization(pipeline_config: HFAudioToEmbeddingPipelineConfig, mock_speech_to_embedding_model: None) -> None:
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    assert pipeline.config == pipeline_config
    assert isinstance(pipeline.model, SpeechToEmbeddingModelPipeline)


def test_process_batch_valid_input(
    pipeline_config: HFAudioToEmbeddingPipelineConfig,
    mock_speech_to_embedding_model: None,
    sample_audio_data: Dict[str, Any]
) -> None:
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    batch: Dict[str, List[Dict[str, Any]]] = {
        "audio": [sample_audio_data, sample_audio_data]}
    result = pipeline.process_batch(batch)
    assert "audio_embedding" in result
    assert isinstance(result["audio_embedding"], np.ndarray)
    assert result["audio_embedding"].shape == (
        2, 3)  # 2 samples, 3 embedding dimensions


def test_process_batch_empty_input(pipeline_config: HFAudioToEmbeddingPipelineConfig, mock_speech_to_embedding_model: None) -> None:
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    batch: Dict[str, List[Any]] = {"audio": []}
    with pytest.raises(ValueError, match="No valid audio inputs found in column audio"):
        pipeline.process_batch(batch)


def test_process_batch_invalid_audio_data(
    pipeline_config: HFAudioToEmbeddingPipelineConfig,
    mock_speech_to_embedding_model: None
) -> None:
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    batch: Dict[str, List[Dict[str, str]]] = {"audio": [{"invalid": "data"}]}
    with pytest.raises(ValueError, match="Invalid audio data format in column"):
        pipeline.process_batch(batch)


def test_process_batch_mixed_valid_invalid_data(
    pipeline_config: HFAudioToEmbeddingPipelineConfig,
    mock_speech_to_embedding_model: None,
    sample_audio_data: Dict[str, Any]
) -> None:
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    batch: Dict[str, List[Dict[str, Any]]] = {
        "audio": [sample_audio_data, {"invalid": "data"}, sample_audio_data]
    }
    with pytest.raises(ValueError, match="Invalid audio data format in column"):
        pipeline.process_batch(batch)


@patch("huggingface_pipelines.speech.SpeechToEmbeddingModelPipeline")
def test_error_handling_in_model_predict(
    mock_predict: MagicMock,
    pipeline_config: HFAudioToEmbeddingPipelineConfig,
    sample_audio_data: Dict[str, Any]
) -> None:
    mock_predict.return_value.predict.side_effect = Exception(
        "Model prediction error")
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    batch: Dict[str, List[Dict[str, Any]]] = {"audio": [sample_audio_data]}
    with pytest.raises(
        ValueError,
        match="Error in model.predict for column audio: Model prediction error",
    ):
        pipeline.process_batch(batch)


def test_process_large_batch(
    pipeline_config: HFAudioToEmbeddingPipelineConfig,
    mock_speech_to_embedding_model: None,
    sample_audio_data: Dict[str, Any]
) -> None:
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    large_batch: Dict[str, List[Dict[str, Any]]] = {
        "audio": [sample_audio_data] * 100
    }  # 100 audio samples
    result = pipeline.process_batch(large_batch)
    assert "audio_embedding" in result
    assert isinstance(result["audio_embedding"], np.ndarray)
    assert result["audio_embedding"].shape == (
        100, 3)  # 100 samples, 3 embedding dimensions


def test_collect_valid_audio_inputs(pipeline_config: HFAudioToEmbeddingPipelineConfig, sample_audio_data: Dict[str, Any]) -> None:
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    audio_data_list: List[Dict[str, Any]] = [
        sample_audio_data, sample_audio_data]
    result = pipeline.collect_valid_audio_inputs(audio_data_list)
    assert len(result) == 2
    assert all(isinstance(tensor, torch.Tensor) for tensor in result)
    assert all(tensor.shape == (1, 16000) for tensor in result)


def test_collect_valid_audio_inputs_invalid_data(pipeline_config: HFAudioToEmbeddingPipelineConfig) -> None:
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    invalid_data: List[Dict[str, str]] = [{"invalid": "data"}]
    with pytest.raises(ValueError, match="Invalid audio data format in column"):
        pipeline.collect_valid_audio_inputs(invalid_data)


def test_collect_valid_audio_inputs_multi_channel(pipeline_config: HFAudioToEmbeddingPipelineConfig) -> None:
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    multi_channel_data: Dict[str, Any] = {
        "array": np.random.rand(2, 16000), "sampling_rate": 16000}
    result = pipeline.collect_valid_audio_inputs([multi_channel_data])
    assert len(result) == 1
    assert result[0].shape == (1, 16000)


def test_process_complex_audio_data(
    pipeline_config: HFAudioToEmbeddingPipelineConfig,
    mock_speech_to_embedding_model: None,
    complex_audio_data: Dict[str, Dict[str, Any]]
) -> None:
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    batch: Dict[str, List[Dict[str, Any]]] = {
        "audio": list(complex_audio_data.values())}
    result = pipeline.process_batch(batch)
    assert "audio_embedding" in result
    assert isinstance(result["audio_embedding"], np.ndarray)
    assert result["audio_embedding"].shape == (len(complex_audio_data), 3)


def test_collect_valid_audio_inputs_complex(pipeline_config: HFAudioToEmbeddingPipelineConfig, complex_audio_data: Dict[str, Dict[str, Any]]) -> None:
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    result = pipeline.collect_valid_audio_inputs(
        list(complex_audio_data.values()))
    assert len(result) == len(complex_audio_data)
    assert all(isinstance(tensor, torch.Tensor) for tensor in result)
    assert all(tensor.dim() == 2 and tensor.size(0) == 1 for tensor in result)


def test_process_batch_with_missing_column(
    pipeline_config: HFAudioToEmbeddingPipelineConfig,
    mock_speech_to_embedding_model: None,
    sample_audio_data: Dict[str, Any]
) -> None:
    pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    batch: Dict[str, List[Dict[str, Any]]] = {
        "wrong_column": [sample_audio_data]}
    result = pipeline.process_batch(batch)
    assert "audio_embedding" not in result


def test_process_batch_with_multiple_columns(
    pipeline_config: HFAudioToEmbeddingPipelineConfig,
    mock_speech_to_embedding_model: None,
    sample_audio_data: Dict[str, Any]
) -> None:
    config = HFAudioToEmbeddingPipelineConfig(
        **{**pipeline_config.__dict__, "columns": ["audio1", "audio2"]}
    )
    pipeline = HFAudioToEmbeddingPipeline(config)
    batch: Dict[str, List[Dict[str, Any]]] = {
        "audio1": [sample_audio_data],
        "audio2": [sample_audio_data],
    }
    result = pipeline.process_batch(batch)
    assert "audio1_embedding" in result
    assert "audio2_embedding" in result


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_handling(
    pipeline_config: HFAudioToEmbeddingPipelineConfig,
    mock_speech_to_embedding_model: None,
    sample_audio_data: Dict[str, Any],
    device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    config = HFAudioToEmbeddingPipelineConfig(
        **{**pipeline_config.__dict__, "device": device}
    )
    pipeline = HFAudioToEmbeddingPipeline(config)
    batch: Dict[str, List[Dict[str, Any]]] = {"audio": [sample_audio_data]}
    result = pipeline.process_batch(batch)
    assert "audio_embedding" in result


def test_batch_size_handling(
    pipeline_config: HFAudioToEmbeddingPipelineConfig,
    mock_speech_to_embedding_model: None,
    sample_audio_data: Dict[str, Any]
) -> None:
    config = HFAudioToEmbeddingPipelineConfig(
        **{**pipeline_config.__dict__, "batch_size": 1}
    )
    pipeline = HFAudioToEmbeddingPipeline(config)
    batch: Dict[str, List[Dict[str, Any]]] = {"audio": [sample_audio_data] * 5}
    result = pipeline.process_batch(batch)
    assert "audio_embedding" in result
    assert result["audio_embedding"].shape == (5, 3)

