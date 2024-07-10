from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
from datasets import Dataset as HFDataset
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
    def process_batch(self, batch: Dict[str, Any], dataset: HFDataset) -> HFDataset:
        """
        Processes a single batch of data and returns the updated dataset.

        Args:
            batch (Dict[str, Any]): A batch of data.
            dataset (HFDataset): The dataset to update.

        Returns:
            HFDataset: The updated dataset.
        """
        pass

    def __call__(self, dataset: HFDataset) -> HFDataset:
        """
        Processes the dataset and updates it.

        Args:
            dataset (HFDataset): The dataset to process.

        Returns:
            HFDataset: The updated dataset.
        """
        try:
            logger.info("Starting to process dataset...")
            if self.config.num_shards > 1:
                dataset = dataset.shard(
                    num_shards=self.config.num_shards, index=self.config.shard_id)

            updated_dataset = dataset.map(
                lambda batch: self.process_batch(batch, dataset),
                batched=True,
                batch_size=self.config.batch_size,
                load_from_cache_file=False
            )
            return updated_dataset
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

