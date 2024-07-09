import logging
from evaluate import load
from typing import List, Dict, Any
from .pipeline_config import MetricConfig

logger = logging.getLogger(__name__)

class MetricAnalyzer:
    """
    A class to analyze metrics for text-to-text pipelines.
    """
    def __init__(self, config: MetricConfig):
        self.config = config
        self.metric = load(self.config.metric_name)
        self.results = []

    def compute_metric(self, original_texts: List[str], reconstructed_texts: List[str]) -> Dict[str, Any]:
        """
        Computes the metric score between original and reconstructed texts.

        Args:
            original_texts (List[str]): A list of original texts.
            reconstructed_texts (List[str]): A list of reconstructed texts.

        Returns:
            Dict[str, Any]: A dictionary containing the metric score.
        """
        logger.info(f"Computing {self.config.metric_name} score...")

        references = [[text.split()] for text in original_texts]
        predictions = reconstructed_texts

        metric_score = self.metric.compute(predictions=predictions, references=references)
        logger.info(f"{self.config.metric_name} score computed: {metric_score}")
        return metric_score

    def analyze_results(self, results: List[Dict[str, Any]]):
        """
        Analyzes the results to determine the percentage of batches with low scores.

        Args:
            results (List[Dict[str, Any]]): A list of results containing original, reconstructed texts and metric scores.
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
            results (List[Dict[str, Any]]): A list of results containing original, reconstructed texts and metric scores.
        """
        for result in results:
            if result['metric_score'][self.config.metric_name] < self.config.low_score_threshold:
                logger.info(f"Low {self.config.metric_name} score detected: {result['metric_score']}")
                logger.info(f"Original Text: {result['original']}")
                logger.info(f"Reconstructed Text: {result['reconstructed']}")
