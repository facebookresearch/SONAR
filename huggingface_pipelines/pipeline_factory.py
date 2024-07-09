from .text import SonarHFTextToTextPipeline
from .pipeline_config import PipelineConfig


class PipelineFactory:
    """
    Factory class for creating pipeline instances.
    """

    @staticmethod
    def create_pipeline(config: PipelineConfig) -> SonarHFTextToTextPipeline:
        return SonarHFTextToTextPipeline(config=config)
