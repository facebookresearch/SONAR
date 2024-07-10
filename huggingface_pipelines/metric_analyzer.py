import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from .pipeline import Pipeline
from .pipeline_config import MetricPipelineConfig
from datasets import Dataset
from evaluate import load

logger = logging.getLogger(__name__)


@dataclass
class MetricAnalyzerPipeline(Pipeline):
    """
    A pipeline to analyze metrics for different data types.
    """
    config: MetricPipelineConfig

    def __post_init__(self):
        self.results = []
        logger.info(f"Loading metric: {self.config.metric_name}...")
        self.metric = load(self.config.metric_name)
        logger.info(f"Metric {self.config.metric_name} loaded successfully.")

    def compute_metric(self, original_data: List[Any], reconstructed_data: List[Any]) -> Dict[str, Any]:
        """
        Computes the metric score between original and reconstructed data.

        Args:
            original_data (List[Any]): A list of original data.
            reconstructed_data (List[Any]): A list of reconstructed data.

        Returns:
            Dict[str, Any]: A dictionary containing the metric score.
        """
        logger.info(f"Computing {self.config.metric_name} score...")
        references = [[text] for text in original_data]
        predictions = reconstructed_data

        # Compute the metric
        metric_score = self.metric.compute(
            predictions=predictions, references=references)
        logger.info(
            f"{self.config.metric_name} score computed: {metric_score}")
        return metric_score

    def process_batch(self, batch: Dict[str, Any], dataset: Dataset) -> Dataset:
        """
        Processes a single batch of data by computing the metric and updating the dataset.

        Args:
            batch (Dict[str, Any]): A batch of data.
            dataset (Dataset): The dataset to update.

        Returns:
            Dataset: The updated dataset.

        """
        for column in self.config.columns:
            original_data = batch[column + '_original']
            reconstructed_data = batch[column + '_reconstructed']
            metric_score = self.compute_metric(
                original_data, reconstructed_data)
            dataset = dataset.add_column(
                column + '_metric_score', [metric_score] * len(original_data))
        return dataset

    def analyze_results(self, results: List[Dict[str, Any]]):
        """
        Analyzes the results to determine the percentage of batches with low scores.

        Args:
            results (List[Dict[str, Any]]): A list of results containing original, reconstructed data and metric scores.
        """
        if not results:
            logger.warning("No results to analyze.")
            return

        logger.info(f"Analyzing results for {self.config.metric_name}...")
        low_score_count = sum(
            1 for result in results if result['metric_score'][self.config.metric_name] < self.config.low_score_threshold)
        total_batches = len(results)
        low_score_percentage = (low_score_count / total_batches) * 100

        logger.info(
            f"Percentage of batches with {self.config.metric_name} score below {self.config.low_score_threshold}: {low_score_percentage:.2f}%")
        self.report_low_scores(results)

    def report_low_scores(self, results: List[Dict[str, Any]]):
        """
        Reports batches with scores below the threshold.

        Args:
            results (List[Dict[str, Any]]): A list of results containing original, reconstructed data and metric scores.
        """
        for result in results:
            if result['metric_score'][self.config.metric_name] < self.config.low_score_threshold:
                logger.info(
                    f"Low {self.config.metric_name} score detected: {result['metric_score']}")
                logger.info(f"Original Data: {result['original']}")
                logger.info(f"Reconstructed Data: {result['reconstructed']}")

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

