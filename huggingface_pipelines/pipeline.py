import gc
import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List

import torch
from datasets import Dataset, IterableDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        gc_collect_frequency (int): Frequency of garbage collection in terms of batches processed. Defaults to every 100 batches.
            Set to 0 to disable explicit garbage collection.
    """
    columns: List[str]
    output_path: str
    output_column_suffix: str = "results"
    load_from_cache_file: bool = True
    batch_size: int = 5
    device: str = "cpu"
    take: int = -1
    gc_collect_frequency: int = 100


class Pipeline(ABC):
    """
    Abstract base class for different pipelines.

    This class defines the common structure and methods for all pipeline implementations.
    Specific pipeline classes should inherit from this class and implement the
    `process_batch` method.

    Attributes:
        config (PipelineConfig): The configuration for this pipeline.
        batch_count (int): Counter for the number of batches processed.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the Pipeline with the given configuration.

        Args:
            config (PipelineConfig): The configuration for this pipeline.
        """
        self.config = config
        self.batch_count = 0

    @contextmanager
    def resource_manager(self):
        try:
            yield
        finally:
            if torch.cuda.is_available() and self.config.device == 'cuda':
                if self.config.gc_collect_frequency > 0 and self.batch_count % self.config.gc_collect_frequency == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

    def manage_resources(func):
        @wraps(func)
        def wrapper(self, batch):
            with self.resource_manager():
                result = func(self, batch)
                self.batch_count += 1
                return result
        return wrapper

    @abstractmethod
    @manage_resources
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
            dataset = dataset.take(self.config.take * self.config.batch_size)

        def process_and_manage_resources(batch):
            with self.resource_manager():
                return self.process_batch(batch)

        updated_dataset = dataset.map(
            process_and_manage_resources,
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

        def process_and_manage_resources(batch):
            with self.resource_manager():
                return self.process_batch(batch)

        updated_dataset = dataset.map(
            process_and_manage_resources,
            batched=True,
            batch_size=self.config.batch_size,
            load_from_cache_file=self.config.load_from_cache_file,
            cache_file_name=cache_file_path,
            desc="Processing dataset",
        )

        return updated_dataset


class PipelineFactory(ABC):
    @abstractmethod
    def create_pipeline(self, config: Dict[str, Any]) -> Pipeline:
        pass
