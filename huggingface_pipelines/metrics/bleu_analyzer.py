import logging
from evaluate import load
from typing import List, Dict, Any
from ..metric_analyzer import MetricAnalyzer
from ..pipeline_config import MetricConfig

logger = logging.getLogger(__name__)


class BleuAnalyzer(MetricAnalyzer):
    """
    BLEU score analyzer for text data.
    """

    def __init__(self, config: MetricConfig):
        super().__init__(config)
        self.metric = load(self.config.metric_name)

    def compute_metric(self, original_texts: List[str], reconstructed_texts: List[str]) -> Dict[str, Any]:
        """
        Computes the BLEU score between original and reconstructed texts.

        Args:
            original_texts (List[str]): A list of original texts.
            reconstructed_texts (List[str]): A list of reconstructed texts.

        Returns:
            Dict[str, Any]: A dictionary containing the BLEU score.
        """
        logger.info(f"Computing {self.config.metric_name} score...")

        references = [[text.split()] for text in original_texts]
        predictions = reconstructed_texts

        metric_score = self.metric.compute(
            predictions=predictions, references=references)
        logger.info(
            f"{self.config.metric_name} score computed: {metric_score}")
        return metric_score
