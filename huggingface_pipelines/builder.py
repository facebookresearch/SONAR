from pathlib import Path

from .text import TextToEmbeddingPipelineFactory, EmbeddingToTextPipelineFactory, TextSegmentationPipelineFactory
from .metric_analyzer import MetricAnalyzerPipelineFactory
from typing import Dict, Any, Literal
from .pipeline import Pipeline
import yaml
import logging

logger = logging.getLogger(__name__)


class PipelineBuilder:
    def __init__(self, config_dir: str = "huggingface_pipelines/datacards"):
        self.config_dir = Path(config_dir)
        self.pipeline_factories = {
            "text_to_embedding": TextToEmbeddingPipelineFactory(),
            "embedding_to_text": EmbeddingToTextPipelineFactory(),
            "text_segmentation": TextSegmentationPipelineFactory(),
            "analyze_metric": MetricAnalyzerPipelineFactory(),
        }

    def load_config(self, dataset_name: str, operation: Literal["text_to_embedding", "embedding_to_text", "text_segmentation", "audio_preprocessing", "analyze_metric"]) -> Dict[str, Any]:
        config_file = self.config_dir / \
            f"{dataset_name}/{operation}.yaml"

        if not config_file.exists():
            logger.error(f"Config File Path: {config_file}")
            raise FileNotFoundError(
                f"No configuration file found for dataset: {dataset_name}")

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def create_pipeline(self, dataset_name: str, operation: str) -> Pipeline:
        config = self.load_config(dataset_name, operation)

        if operation not in self.pipeline_factories:
            raise ValueError(f"Unsupported operation: {operation}")

        return self.pipeline_factories[operation].create_pipeline(config)
