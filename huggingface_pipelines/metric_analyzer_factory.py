from .config import MetricConfig
from .metric_analyzer import MetricAnalyzer
from .bleu_analyzer import BleuAnalyzer


class MetricAnalyzerFactory:
    """
    Factory class for creating MetricAnalyzer instances.
    """

    @staticmethod
    def create_analyzer(config: MetricConfig) -> MetricAnalyzer:
        """
        Creates a MetricAnalyzer instance based on the metric configuration.

        Args:
            config (MetricConfig): Configuration for the metric analyzer.

        Returns:
            MetricAnalyzer: An instance of a MetricAnalyzer.
        """
        if config.metric_name == "bleu":
            return BleuAnalyzer(config)
        else:
            raise ValueError(f"Unsupported metric: {config.metric_name}")
