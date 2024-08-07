from abc import ABC, abstractmethod
from typing import List, TypedDict, Dict, Any
import logging
from dataclasses import dataclass, replace
from datasets import Dataset, IterableDataset
import os
from contextlib import contextmanager
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineOverwrites(TypedDict):
    """
    A TypedDict representing the possible overwrites for a PipelineConfig.

    This allows for dynamic modification of pipeline configuration parameters.

    Attributes:
        batch_size (int): The batch size for processing data.
        device (str): The device to use for computation (e.g., 'cpu', 'cuda').
        cache_to_arrow (bool): Whether to cache results in Arrow format.
        take (int): The number of batches to process (-1 for all).
        output_path (str): The directory path for output files.
        output_file_name (str): The name of the output file.
        columns (List[str]): The columns to be processed in the dataset.
    """
    batch_size: int
    device: str
    cache_to_arrow: bool
    take: int
    output_path: str
    output_file_name: str
    columns: List[str]


@dataclass
class PipelineConfig(ABC):

    """
    Abstract base class for pipeline configurations.

    This class defines the common configuration parameters for all pipelines.
    It is designed to be flexible and can be used with various types of models,
    including but not limited to PyTorch models.

    Attributes:
        columns (List[str]): The columns to be processed by the pipeline.
        output_path (str): The directory path for output files.
        output_column_suffix (str): The suffix to append to output column names.
            This is used to distinguish processed columns from original ones.
        load_from_cache_file (bool): Whether to load the dataset from cache file.
            This can significantly speed up repeated processing of the same dataset.
        batch_size (int): The batch size for processing data. This affects both
            memory usage and processing speed. Adjust based on available resources.
        device (str): The device to use for computation (e.g., 'cpu', 'cuda').
            This is relevant as all torch models for the instantiated pipelines will use this device.

        take (int): The number of batches to process (-1 for all). Useful for
            debugging or processing subsets of large datasets.
        encoder_model (str): The name or path of the encoder model to use.
            This is a placeholder and its usage depends on the specific pipeline implementation.
        source_lang (str): The source language code (e.g., 'eng_Latn').
            This is used for language-specific processing tasks.
    """
    columns: List[str]
    output_path: str
    output_column_suffix: str
    load_from_cache_file: bool = True
    batch_size: int = 5
    device: str = "cpu"
    take: int = -1

    def with_overwrites(self, overwrites: PipelineOverwrites) -> 'PipelineConfig':
        """
        Create a new PipelineConfig with the specified overwrites.

        This method allows for the creation of a new configuration object
        with selective parameter updates without modifying the original.

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

        This method should be implemented by all concrete pipeline classes to define
        the specific data processing logic for each pipeline.

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
        handling batching, caching, and error management. It supports both
        regular and streaming datasets.

        Args:
            dataset (Dataset): The dataset to process.

        Returns:
            Dataset: The processed dataset.

        Raises:
            Exception: If there's an error during dataset processing.
        """
        try:
            logger.info("Starting to process dataset...")
            os.makedirs(self.config.output_path, exist_ok=True)

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

        This method applies the pipeline's processing logic to a streaming dataset,
        which is useful for large datasets that don't fit in memory.

        Args:
            dataset (IterableDataset): The streaming dataset to process.

        Returns:
            IterableDataset: The processed streaming dataset.
        """

        if self.config.take > 0:
            updated_dataset = dataset.take(
                self.config.take * self.config.batch_size)

        updated_dataset = dataset.map(
            self.process_batch,
            batched=True,
            batch_size=self.config.batch_size,
        )
        return updated_dataset

    def process_regular_dataset(self, dataset: Dataset) -> Dataset:
        """
        Process a regular (non-streaming) dataset.

        This method applies the pipeline's processing logic to a regular dataset,
        with support for caching and selective processing.

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
            self.config.output_path, cache_file_name)

        updated_dataset = dataset.map(
            self.process_batch,
            batched=True,
            batch_size=self.config.batch_size,
            load_from_cache_file=self.config.load_from_cache_file,
            cache_file_name=cache_file_path,
            desc="Processing dataset",
        )

        return updated_dataset

