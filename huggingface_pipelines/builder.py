import logging
from pathlib import Path
from typing import Any, Dict, Literal, Union

import yaml

from .audio import AudioToEmbeddingPipelineFactory  # type: ignore
from .metric_analyzer import MetricAnalyzerPipelineFactory  # type: ignore
from .pipeline import Pipeline, PipelineFactory  # type: ignore
from .text import (  # type: ignore[import]
    EmbeddingToTextPipelineFactory,
    TextSegmentationPipelineFactory,
    TextToEmbeddingPipelineFactory,
)

logger = logging.getLogger(__name__)

# Define a custom type for supported operations
SupportedOperation = Literal[
    "text_to_embedding",
    "embedding_to_text",
    "text_segmentation",
    "analyze_metric",
    "audio_to_embedding",
]


class PipelineBuilder:
    """
    A class for building and managing different types of processing pipelines.

    This class provides methods to create pipelines for various operations such as
    text-to-embedding, embedding-to-text, text segmentation, metric analysis, and
    audio-to-embedding. It uses the Factory pattern to create specific pipeline
    instances based on the operation type and configuration.


    Attributes:
        config_dir (Path): The directory containing configuration files for pipelines.
        pipeline_factories (Dict[SupportedOperation, PipelineFactory]): A dictionary mapping
            operations to their respective factory classes.

    Args:
        config_dir (Union[str, Path], optional): The directory containing configuration
            files. Defaults to "huggingface_pipelines/datacards".

    Example:
        >>> builder = PipelineBuilder()
        >>> text_to_embedding_pipeline = builder.create_pipeline("sonar", "text_to_embedding")
        >>> processed_dataset = text_to_embedding_pipeline(input_dataset)
    """

    def __init__(
        self, config_dir: Union[str, Path] = "huggingface_pipelines/datacards"
    ):
        self.config_dir = Path(config_dir)
        self.pipeline_factories: Dict[SupportedOperation, PipelineFactory] = {
            "text_to_embedding": TextToEmbeddingPipelineFactory(),
            "embedding_to_text": EmbeddingToTextPipelineFactory(),
            "text_segmentation": TextSegmentationPipelineFactory(),
            "analyze_metric": MetricAnalyzerPipelineFactory(),
            "audio_to_embedding": AudioToEmbeddingPipelineFactory(),
        }

    def load_config(
        self, dataset_name: str, operation: SupportedOperation
    ) -> Dict[str, Any]:
        """
        Load the configuration for a specific dataset and operation.

        This method reads the YAML configuration file for the specified dataset and operation.
        The configuration is used to set up the PipelineConfig for the requested pipeline.

        Args:
            dataset_name (str): The name of the dataset.
            operation (SupportedOperation): The type of operation to perform.

        Returns:
            Dict[str, Any]: The configuration dictionary.

        Raises:
            FileNotFoundError: If the configuration file is not found.

        """
        config_file = self.config_dir / f"{dataset_name}/{operation}.yaml"
        try:
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config File not found: {config_file}")
            raise FileNotFoundError(
                f"No configuration file found for dataset: {dataset_name} and operation: {operation}"
            )

    def create_pipeline(
        self, dataset_name: str, operation: SupportedOperation
    ) -> Pipeline:
        """
        Create a pipeline for a specific dataset and operation.

        This method uses the appropriate PipelineFactory to create a Pipeline instance
        based on the specified operation and configuration. The created Pipeline
        adheres to the abstract Pipeline class structure and uses a PipelineConfig
        for its configuration.

        Args:
            dataset_name (str): The name of the dataset.
            operation (SupportedOperation): The type of operation to perform.

        Returns:
            Pipeline: The created pipeline instance, which can be called with a dataset.

        Raises:
            ValueError: If the operation is not supported.

        Example:
            >>> builder = PipelineBuilder()
            >>> text_to_embedding_pipeline = builder.create_pipeline("dataset_name", "text_to_embedding")
            >>> processed_dataset = text_to_embedding_pipeline(input_dataset)

            >>> audio_to_embedding_pipeline = builder.create_pipeline("dataset_name", "audio_to_embedding")
            >>> processed_audio_dataset = audio_to_embedding_pipeline(input_audio_dataset)
        """
        if operation not in self.pipeline_factories:
            raise ValueError(
                f"Unsupported operation: {operation}. Supported operations are: {', '.join(self.pipeline_factories.keys())}"
            )
        config = self.load_config(dataset_name, operation)
        return self.pipeline_factories[operation].create_pipeline(config)
