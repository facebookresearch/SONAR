from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
from datasets import Dataset
from .pipeline_config import PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline(ABC):
    """
    Abstract base class for different pipelines.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results = []

    @abstractmethod
    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single batch of data and returns the updated batch.

        Args:
            batch (Dict[str, Any]): A batch of data.

        Returns:
            Dict[str, Any]: The updated batch.
        """
        pass

    def __call__(self, dataset: Dataset) -> Dataset:
        """
        Processes the dataset and updates it.

        Args:
            dataset (Dataset): The dataset to process.

        Returns:
            Dataset: The updated dataset.
        """
        try:
            logger.info("Starting to process dataset...")

            if self.config.num_shards > 1:
                dataset = dataset.shard(
                    num_shards=self.config.num_shards, index=self.config.shard_id)

            if self.config.take > 0:
                dataset = dataset.select(
                    range(self.config.take * self.config.batch_size))

            updated_dataset = dataset.map(
                lambda batch: self.process_batch(batch),
                batched=True,
                batch_size=self.config.batch_size,
                load_from_cache_file=False
            )
            return updated_dataset
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

