import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch
from datasets import load_metric

from huggingface_pipelines.metric_analyzer import (
    MetricAnalyzerPipeline,
    MetricPipelineConfig,
    MetricAnalyzerPipelineFactory,
)


@pytest.fixture
def sample_config():
    return {
        "metrics": ["bleu", "rouge"],
        "low_score_threshold": 0.6,
        "columns": ["text"],
        "reconstructed_columns": ["reconstructed_text"],
        "output_column_suffix": "score",
    }


@pytest.fixture
def sample_batch():
    return {
        "text": ["This is a test sentence.", "Another example sentence."],
        "reconstructed_text": ["This is a test sentence.", "A different example sentence."],
    }


def test_metric_pipeline_config():
    config = MetricPipelineConfig(
        metrics=["bleu", "rouge"],
        low_score_threshold=0.6,
        columns=["text"],
        reconstructed_columns=["reconstructed_text"],
        output_column_suffix="score",
    )
    assert config.metrics == ["bleu", "rouge"]
    assert config.low_score_threshold == 0.6
    assert config.columns == ["text"]
    assert config.reconstructed_columns == ["reconstructed_text"]
    assert config.output_column_suffix == "score"


@patch("huggingface_pipelines.metric_analyzer.load_metric")
def test_metric_analyzer_pipeline_init(mock_load_metric, sample_config):
    mock_load_metric.return_value = Mock()
    pipeline = MetricAnalyzerPipeline(MetricPipelineConfig(**sample_config))
    assert len(pipeline.metrics) == 2
    assert "bleu" in pipeline.metrics
    assert "rouge" in pipeline.metrics
    mock_load_metric.assert_any_call("bleu")
    mock_load_metric.assert_any_call("rouge")


@patch("huggingface_pipelines.metric_analyzer.load_metric")
def test_compute_metric(mock_load_metric, sample_config):
    mock_metric = Mock()
    mock_metric.compute.return_value = {"score": 0.8}
    mock_load_metric.return_value = mock_metric

    pipeline = MetricAnalyzerPipeline(MetricPipelineConfig(**sample_config))
    result = pipeline.compute_metric(
        "bleu",
        [["This is a reference."]],
        ["This is a prediction."]
    )

    assert result == {"score": 0.8}
    mock_metric.compute.assert_called_once_with(
        predictions=["This is a prediction."],
        references=[["This is a reference."]]
    )


@patch("huggingface_pipelines.metric_analyzer.load_metric")
def test_process_batch(mock_load_metric, sample_config, sample_batch):
    mock_metric = Mock()
    mock_metric.compute.return_value = {"score": 0.8}
    mock_load_metric.return_value = mock_metric

    pipeline = MetricAnalyzerPipeline(MetricPipelineConfig(**sample_config))
    result = pipeline.process_batch(sample_batch)

    assert "text_bleu_score" in result
    assert "text_rouge_score" in result
    assert "text_bleu_score_low" in result
    assert "text_rouge_score_low" in result
    assert result["text_bleu_score"] == [0.8, 0.8]
    assert result["text_rouge_score"] == [0.8, 0.8]
    assert result["text_bleu_score_low"] == [False, False]
    assert result["text_rouge_score_low"] == [False, False]


@patch("huggingface_pipelines.metric_analyzer.load_metric")
def test_process_batch_with_list_input(mock_load_metric, sample_config):
    mock_metric = Mock()
    mock_metric.compute.return_value = {"score": 0.8}
    mock_load_metric.return_value = mock_metric

    pipeline = MetricAnalyzerPipeline(MetricPipelineConfig(**sample_config))
    batch = {
        "text": [["This", "is", "a", "test"], ["Another", "example"]],
        "reconstructed_text": [["This", "is", "a", "test"], ["A", "different", "example"]],
    }
    result = pipeline.process_batch(batch)

    assert "text_bleu_score" in result
    assert "text_rouge_score" in result
    assert result["text_bleu_score"] == [0.8, 0.8]
    assert result["text_rouge_score"] == [0.8, 0.8]


def test_process_batch_mismatch_columns():
    config = MetricPipelineConfig(
        metrics=["bleu"],
        columns=["text1", "text2"],
        reconstructed_columns=["reconstructed_text1"],
    )
    pipeline = MetricAnalyzerPipeline(config)

    with pytest.raises(ValueError, match="Mismatch in number of columns"):
        pipeline.process_batch(
            {"text1": ["Test"], "reconstructed_text1": ["Test"]})


def test_metric_analyzer_pipeline_factory(sample_config):
    factory = MetricAnalyzerPipelineFactory()
    pipeline = factory.create_pipeline(sample_config)
    assert isinstance(pipeline, MetricAnalyzerPipeline)
    assert pipeline.config.metrics == sample_config["metrics"]
    assert pipeline.config.low_score_threshold == sample_config["low_score_threshold"]
    assert pipeline.config.columns == sample_config["columns"]
    assert pipeline.config.reconstructed_columns == sample_config["reconstructed_columns"]
    assert pipeline.config.output_column_suffix == sample_config["output_column_suffix"]


if __name__ == "__main__":
    pytest.main()
