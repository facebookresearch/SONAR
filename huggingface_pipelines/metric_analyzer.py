import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datasets import load_metric
from .pipeline import PipelineConfig, Pipeline, PipelineOverwrites

logger = logging.getLogger(__name__)


class MetricOverwrites(PipelineOverwrites, total=False):
    metrics: List[str]
    low_score_threshold: float
    reconstructed_columns: List[str]


@dataclass
class MetricPipelineConfig(PipelineConfig):
    """
    Configuration class for metrics.

    Attributes:
        metrics (List[str]): List of metric names to be used for evaluation.
        low_score_threshold (float): Threshold below which scores are considered low for all metrics.
        columns (List[str]): List of original columns to compute metrics for.
        reconstructed_columns (List[str]): List of reconstructed columns corresponding to original columns.
        output_column_suffix (str): Suffix for the output column names.
    """
    metrics: List[str] = field(default_factory=list)
    low_score_threshold: float = 0.5
    reconstructed_columns: List[str] = field(default_factory=list)

    def with_overwrites(self, overwrites: MetricOverwrites) -> 'MetricPipelineConfig':
        return MetricPipelineConfig(**{**self.__dict__, **overwrites})


class MetricAnalyzerPipeline(Pipeline):
    """
    A pipeline to analyze multiple metrics for different data types and reconstructed columns.
    """

    def __init__(self, config: MetricPipelineConfig):
        self.config = config
        self.metrics = {}
        for metric_name in self.config.metrics:
            logger.info(f"Loading metric: {metric_name}...")
            self.metrics[metric_name] = load_metric(metric_name)
            logger.info(f"Metric {metric_name} loaded successfully.")

    def compute_metric(self, metric_name: str, references: List[List[str]], predictions: List[str]) -> Dict[str, Any]:
        """
        Computes the metric score between references and predictions.

        Args:
            metric_name (str): Name of the metric to compute.
            references (List[List[str]]): A list of reference texts, each wrapped in a list.
            predictions (List[str]): A list of predicted texts.

        Returns:
            Dict[str, Any]: A dictionary containing the metric score.
        """
        logger.info(f"Computing {metric_name} score...")

        metric_score = self.metrics[metric_name].compute(
            predictions=predictions, references=references)
        logger.info(f"{metric_name} score computed: {metric_score}")
        return metric_score

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single batch of data by computing metrics and updating the current batch.

        Args:
            batch (Dict[str, Any]): A batch of data.

        Returns:
            Dict[str, Any]: The updated batch with the metric scores, predictions, and references.
        """
        for column, reconstructed_column in zip(self.config.columns, self.config.reconstructed_columns):
            original_data = batch[column]
            reconstructed_data = batch[reconstructed_column]

            # Join back into strings

            original_data = [' '.join(item) for item in original_data]
            reconstructed_data = [' '.join(item)
                                  for item in reconstructed_data]

            references = [[ref.split()] for ref in original_data]
            predictions = [pred.split() for pred in reconstructed_data]

            batch[f"{column}_references"] = references
            batch[f"{column}_predictions"] = predictions

            for metric_name in self.config.metrics:
                metric_score = self.compute_metric(
                    metric_name, batch[f"{column}_references"], batch[f"{column}_predictions"])

                output_column = f"{column}_{metric_name}_{self.config.output_column_suffix}"
                score_value = metric_score[list(metric_score.keys())[0]]
                batch[output_column] = [score_value] * len(original_data)

                # Add a flag for low scores
                low_score_flag = f"{output_column}_low"
                batch[low_score_flag] = [
                    score < self.config.low_score_threshold for score in batch[output_column]]

        return batch

