import logging
from typing import List, Dict, Any
from dataclasses import dataclass, replace
from .pipeline import Pipeline, PipelineOverwrites, PipelineConfig
from datasets import Dataset
from evaluate import load

logger = logging.getLogger(__name__)


class MetricOverwrites(PipelineOverwrites, total=False):
    metric_name: str
    low_score_threshold: float


@dataclass
class MetricPipelineConfig(PipelineConfig):
    """
    Configuration class for metrics.

    Attributes:
        metric_name (str): The name of the metric to be used for evaluation.
        low_score_threshold (float): The threshold below which the score is considered low.
    """
    metric_name: str = "bleu"
    low_score_threshold: float = 0.5

    def with_overwrites(self, overwrites: MetricOverwrites):
        return replace(self, **overwrites)


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

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single batch of data by computing the metric and updating the current batch.

        Args:
            batch (Dict[str, Any]): A batch of data.

        Returns:
            batch: The updated batch with the 'metric_score' column.

        """

        for column in self.config.columns:
            original_data = batch[column]
            reconstructed_data = batch[column + '_reconstructed']
            metric_score = self.compute_metric(
                original_data, reconstructed_data)
            batch[column + '_metric_score'] = [metric_score] * \
                len(original_data)
        return batch

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
