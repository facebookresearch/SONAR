from abc import ABC, abstractmethod
from typing import List, TypedDict, Dict, Any
import logging
from dataclasses import dataclass, replace
from datasets import Dataset, IterableDataset
import os
from contextlib import contextmanager
import torch
from .dataset import DatasetConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineOverwrites(TypedDict):
    """
    A TypedDict representing the possible overwrites for a PipelineConfig.

    Attributes:
        batch_size (int): The new batch size to use for processing.
        device (str): The new device to use for processing (e.g., 'cpu', 'cuda').
        cache_to_arrow (bool): Whether to cache results to Arrow format.
        take (int): The new number of batches to process (-1 for all).
        output_dir (str): The new output directory for results.
        output_file_name (str): The new output file name for results.
        columns (List[str]): The new list of columns to process.
    """

    batch_size: int
    device: str
    cache_to_arrow: bool
    take: int
    output_dir: str
    output_file_name: str
    columns: List[str]


@dataclass
class PipelineConfig(ABC):
    """
    Abstract base class for pipeline configurations.

    This class defines the common configuration parameters for all pipelines.
    Specific pipeline implementations should inherit from this class and add
    any additional configuration parameters they need.

    Attributes:
        columns (List[str]): The columns to be transformed by the pipeline.
        dataset_config (DatasetConfig): The configuration related to loading datasets.
        output_column_suffix (str): The suffix to append to output column names.
        batch_size (int): The batch size to be used for processing.
        device (str): The device to use for inference (e.g., 'cpu', 'cuda').
        take (int): The number of batches to take for processing (-1 for all).
        cache_to_arrow (bool): Whether to cache results to Arrow format.
        encoder_model (str): The name of the encoder model to use.
        source_lang (str): The source language code (e.g., 'eng_Latn').
    """

    columns: List[str]
    dataset_config: DatasetConfig
    output_column_suffix: str
    batch_size: int = 5
    device: str = "cpu"
    take: int = -1
    cache_to_arrow: bool = False
    encoder_model: str = "text_sonar_basic_encoder"
    source_lang: str = "eng_Latn"

    def with_overwrites(self, overwrites: PipelineOverwrites) -> 'PipelineConfig':
        """
        Create a new PipelineConfig with the specified overwrites.

        Args:
            overwrites (PipelineOverwrites): A dictionary of configuration overwrites.

        Returns:
            PipelineConfig: A new PipelineConfig instance with the applied overwrites.
        """
        return replace(self, **overwrites)


class Pipeline(ABC):
    """
    Abstract base class for different pipelines.

    This class defines the common structure and methods for all pipeline implementations.
    Specific pipeline classes should inherit from this class and implement the
    `process_batch` method.

    Attributes:
        config (PipelineConfig): The configuration for this pipeline.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the Pipeline with the given configuration.

        Args:
            config (PipelineConfig): The configuration for this pipeline.
        """
        self.config = config

    @contextmanager
    def resource_manager(self):
        """
        Context manager to efficiently initialize and free pipeline resources.

        This method ensures that CUDA memory is properly managed when using GPU.
        It clears the CUDA cache before and after pipeline execution.

        Yields:
            None
        """
        try:
            if torch.cuda.is_available() and self.config.device == 'cuda':
                torch.cuda.empty_cache()
            yield
        finally:
            if torch.cuda.is_available() and self.config.device == 'cuda':
                torch.cuda.empty_cache()

    @abstractmethod
    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single batch of data and returns the updated batch.

        This method should be implemented by all concrete pipeline classes.

        Args:
            batch (Dict[str, Any]): A batch of data to process.

        Returns:
            Dict[str, Any]: The processed batch of data.
        """
        pass

    def __call__(self, dataset: Dataset) -> Dataset:
        """
        Processes the entire dataset using the pipeline.

        This method applies the `process_batch` method to the entire dataset,
        handling batching, caching, and error management.

        Args:
            dataset (Dataset): The dataset to process.

        Returns:
            Dataset: The processed dataset.

        Raises:
            Exception: If there's an error during dataset processing.
        """
        try:
            logger.info("Starting to process dataset...")
            os.makedirs(
                f"{self.config.dataset_config.output_dir}_{self.config.dataset_config.dataset_name}_{self.config.dataset_config.uuid}",
                exist_ok=True
            )

            if isinstance(dataset, IterableDataset):
                return self.process_streaming_dataset(dataset)
            else:
                return self.process_regular_dataset(dataset)
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

    def process_streaming_dataset(self, dataset: IterableDataset) -> IterableDataset:
        """
        Process a streaming dataset.

        Args:
            dataset (IterableDataset): The streaming dataset to process.

        Returns:
            IterableDataset: The processed streaming dataset.
        """
        updated_dataset = dataset.map(
            self.process_batch,
            batched=True,
            batch_size=self.config.batch_size,
        )

        if self.config.take > 0:
            updated_dataset = updated_dataset.take(
                self.config.take * self.config.batch_size)

        return updated_dataset

    def process_regular_dataset(self, dataset: Dataset) -> Dataset:
        """
        Process a regular (non-streaming) dataset.

        Args:
            dataset (Dataset): The regular dataset to process.

        Returns:
            Dataset: The processed dataset.
        """
        if self.config.take > 0:
            dataset = dataset.select(
                range(self.config.take * self.config.batch_size))

        cache_file_name = f"cache_{self.__class__.__name__}.arrow"
        cache_file_path = os.path.join(
            f"{self.config.dataset_config.output_dir}_{self.config.dataset_config.dataset_name}_{self.config.dataset_config.uuid}",
            cache_file_name
        )

        updated_dataset = dataset.map(
            self.process_batch,
            batched=True,
            batch_size=self.config.batch_size,
            load_from_cache_file=self.config.cache_to_arrow,
            cache_file_name=cache_file_path,
            desc="Processing dataset",
            num_proc=1
        )

        return updated_dataset

