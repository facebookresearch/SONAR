from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline(ABC):
    """
    Abstract base class for different pipelines.
    """

    def __init__(self, config):
        self.config = config
        self.results = []

    @abstractmethod
    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single batch of data.

        Args:
            batch (Dict[str, Any]): A batch of data.

        Returns:
            Dict[str, Any]: Processed batch results.
        """
        pass

    def process_batches(self):
        """
        Processes all batches in the dataset and stores the results.
        """
        try:
            logger.info("Starting to process batches...")
            if self.config.num_shards == 1:
                # Process the entire dataset
                dataset_shard = self.dataset
            else:
                # Select the shard
                dataset_shard = self.dataset.shard(
                    num_shards=self.config.num_shards, index=self.config.shard_id)

            # Process the shard or entire dataset
            results = dataset_shard.map(
                lambda batch: self.process_batch(batch),
                batched=True,
                batch_size=self.config.batch_size,
                remove_columns=dataset_shard.column_names,
                load_from_cache_file=False
            )
            self.results.extend([{k: v[i] for k, v in results.items()}
                                 for i in range(len(results[next(iter(results))]))])

            logger.info("Data processed. Caching results...")
            if self.config.cache_to_arrow:
                self.cache_results_arrow()
                logger.info("Results cached successfully to Arrow file.")
            else:
                self.cache_results()
                logger.info("Results cached successfully to disk.")
        except Exception as e:
            logger.error(f"Error processing batches: {e}")

    @abstractmethod
    def cache_results(self):
        """
        Caches the results to a file.
        """
        pass

    @abstractmethod
    def cache_results_arrow(self):
        """
        Caches the results to an Arrow file.
        """
        pass
