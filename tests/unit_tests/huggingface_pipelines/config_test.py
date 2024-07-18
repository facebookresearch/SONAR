import pytest
from dataclasses import dataclass

from huggingface_pipelines.config import (
    PipelineConfig, DatasetConfig, TextToEmbeddingPipelineConfig,
    EmbeddingToTextPipelineConfig, AudioPipelineConfig, MetricPipelineConfig,
    PipelineOverwrites, DatasetOverwrites, TextToEmbeddingOverwrites,
    EmbeddingToTextOverwrites, AudioOverwrites, MetricOverwrites
)


@dataclass
class MockPipelineConfig(PipelineConfig):
    pass


def test_pipeline_config():
    config = MockPipelineConfig(
        columns=["text"],
        batch_size=32,
        device="cpu",
        take=100,
        output_dir="output",
        output_file_name="results.txt"
    )
    assert config.columns == ["text"]
    assert config.batch_size == 32
    assert config.device == "cpu"
    assert config.take == 100
    assert config.output_dir == "output"
    assert config.output_file_name == "results.txt"
    assert not config.cache_to_arrow


def test_pipeline_config_with_overwrites():
    config = MockPipelineConfig(
        columns=["text"],
        batch_size=32,
        device="cpu",
        take=100,
        output_dir="output",
        output_file_name="results.txt"
    )
    overwrites = PipelineOverwrites(batch_size=64, device="cuda")
    new_config = config.with_overwrites(overwrites)
    assert new_config.batch_size == 64
    assert new_config.device == "cuda"
    assert new_config.columns == ["text"]  # Unchanged


def test_dataset_config():
    config = DatasetConfig(
        dataset_name="example_dataset",
        dataset_split="train"
    )
    assert config.dataset_name == "example_dataset"
    assert config.dataset_split == "train"
    assert config.num_shards == 1
    assert config.shard_id == 0
    assert config.cache_dir is None
    assert not config.trust_remote_code


@pytest.mark.parametrize("Config, Overwrites", [
    (TextToEmbeddingPipelineConfig, TextToEmbeddingOverwrites),
    (EmbeddingToTextPipelineConfig, EmbeddingToTextOverwrites),
    (AudioPipelineConfig, AudioOverwrites),
    (MetricPipelineConfig, MetricOverwrites)
])
def test_specific_pipeline_configs(Config, Overwrites):
    base_args = {
        "columns": ["text"],
        "batch_size": 32,
        "device": "cpu",
        "take": 100,
        "output_dir": "output",
        "output_file_name": "results.txt"
    }
    config = Config(**base_args)

    # Test default values
    assert isinstance(config, PipelineConfig)

    # Test with_overwrites
    overwrites = Overwrites(batch_size=64)
    new_config = config.with_overwrites(overwrites)
    assert new_config.batch_size == 64
    assert new_config.device == "cpu"  # Unchanged


def test_dataset_config_load_dataset(mocker):
    # Mock the load_dataset function
    mock_load_dataset = mocker.patch('datasets.load_dataset')
    mock_dataset = mocker.Mock()
    mock_load_dataset.return_value = mock_dataset

    config = DatasetConfig(
        dataset_name="example_dataset",
        dataset_split="train",
        num_shards=2,
        shard_id=1
    )

    dataset = config.load_dataset()

    # Check if load_dataset was called with correct arguments
    mock_load_dataset.assert_called_once_with(
        "example_dataset",
        split="train",
        cache_dir=None,
        trust_remote_code=False,
        config=None
    )

    # Check if dataset was sharded
    mock_dataset.shard.assert_called_once_with(num_shards=2, index=1)


def test_dataset_config_with_overwrites():
    config = DatasetConfig(
        dataset_name="example_dataset",
        dataset_split="train"
    )
    overwrites = DatasetOverwrites(dataset_name="new_dataset", num_shards=3)
    new_config = config.with_overwrites(overwrites)
    assert new_config.dataset_name == "new_dataset"
    assert new_config.num_shards == 3
    assert new_config.dataset_split == "train"  # Unchanged


if __name__ == "__main__":
    pytest.main()

