from unittest.mock import Mock, patch

import pytest

from huggingface_pipelines.metric_analyzer import (
    MetricAnalyzerPipeline,
    MetricAnalyzerPipelineFactory,
    MetricPipelineConfig,
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
        "text": ["Hello world", "This is a test"],
        "reconstructed_text": ["Hello earth", "This is a quiz"],
    }


def test_metric_pipeline_config():
    config = MetricPipelineConfig(
        metrics=["bleu"],
        low_score_threshold=0.7,
        columns=["col1"],
        reconstructed_columns=["rec_col1"],
        output_column_suffix="test",
    )
    assert config.metrics == ["bleu"]
    assert config.low_score_threshold == 0.7
    assert config.columns == ["col1"]
    assert config.reconstructed_columns == ["rec_col1"]
    assert config.output_column_suffix == "test"


@patch("evaluate.load")
def test_metric_analyzer_pipeline_init(mock_load, sample_config):
    mock_load.return_value = Mock()
    pipeline = MetricAnalyzerPipeline(MetricPipelineConfig(**sample_config))
    assert len(pipeline.metrics) == 2
    mock_load.assert_any_call("bleu")
    mock_load.assert_any_call("rouge")


def test_compute_metric(sample_config):
    pipeline = MetricAnalyzerPipeline(MetricPipelineConfig(**sample_config))
    pipeline.metrics["bleu"] = Mock()
    pipeline.metrics["bleu"].compute.return_value = {"score": 0.8}

    result = pipeline.compute_metric("bleu", [["Hello", "world"]], ["Hello", "earth"])
    assert result == {"score": 0.8}
    pipeline.metrics["bleu"].compute.assert_called_once_with(
        predictions=["Hello", "earth"], references=[["Hello", "world"]]
    )


def test_process_batch(sample_config, sample_batch):
    pipeline = MetricAnalyzerPipeline(MetricPipelineConfig(**sample_config))
    pipeline.compute_metric = Mock(return_value={"score": 0.75})

    result = pipeline.process_batch(sample_batch)

    assert "text_references" in result
    assert "text_predictions" in result
    assert "text_bleu_score" in result
    assert "text_rouge_score" in result
    assert "text_bleu_score_low" in result
    assert "text_rouge_score_low" in result

    assert result["text_bleu_score"] == [0.75, 0.75]
    assert result["text_bleu_score_low"] == [False, False]


def test_process_batch_mismatch_columns():
    config = MetricPipelineConfig(
        metrics=["bleu"], columns=["col1", "col2"], reconstructed_columns=["rec_col1"]
    )
    pipeline = MetricAnalyzerPipeline(config)

    with pytest.raises(ValueError, match="Mismatch in number of columns"):
        pipeline.process_batch({"col1": ["text"], "rec_col1": ["text"]})


def test_process_batch_list_input(sample_config):
    config = MetricPipelineConfig(**sample_config)
    pipeline = MetricAnalyzerPipeline(config)
    pipeline.compute_metric = Mock(return_value={"score": 0.8})

    batch = {
        "text": [["Hello", "world"], ["This", "is", "a", "test"]],
        "reconstructed_text": [["Hello", "earth"], ["This", "is", "a", "quiz"]],
    }

    result = pipeline.process_batch(batch)

    assert result["text_references"] == [
        [["Hello", "world"]],
        [["This", "is", "a", "test"]],
    ]
    assert result["text_predictions"] == [
        ["Hello", "earth"],
        ["This", "is", "a", "quiz"],
    ]


def test_metric_analyzer_pipeline_factory(sample_config):
    factory = MetricAnalyzerPipelineFactory()
    pipeline = factory.create_pipeline(sample_config)

    assert isinstance(pipeline, MetricAnalyzerPipeline)
    assert pipeline.config.metrics == sample_config["metrics"]
    assert pipeline.config.low_score_threshold == sample_config["low_score_threshold"]
    assert pipeline.config.columns == sample_config["columns"]
    assert (
        pipeline.config.reconstructed_columns == sample_config["reconstructed_columns"]
    )
    assert pipeline.config.output_column_suffix == sample_config["output_column_suffix"]


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.7, [False, False]),
        (0.5, [True, True]),
        (0.6, [False, False]),
    ],
)
def test_low_score_threshold(sample_config, sample_batch, score, expected):
    pipeline = MetricAnalyzerPipeline(MetricPipelineConfig(**sample_config))
    pipeline.compute_metric = Mock(return_value={"score": score})

    result = pipeline.process_batch(sample_batch)

    assert result["text_bleu_score_low"] == expected
    assert result["text_rouge_score_low"] == expected
