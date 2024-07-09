import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .pipeline_config import MetricConfig

logger = logging.getLogger(__name__)


class MetricAnalyzer(ABC):
    """
    Abstract base class to analyze metrics for different data types.
    """

    def __init__(self, config: MetricConfig):
        self.config = config
        self.results = []

    @abstractmethod
    def compute_metric(self, original_data: List[Any], reconstructed_data: List[Any]) -> Dict[str, Any]:
        """
        Computes the metric score between original and reconstructed data.

        Args:
            original_data (List[Any]): A list of original data.
            reconstructed_data (List[Any]): A list of reconstructed data.

        Returns:
            Dict[str, Any]: A dictionary containing the metric score.
        """
        pass

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

