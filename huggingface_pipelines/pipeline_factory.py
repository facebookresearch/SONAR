from .text import TextToTextHFPipeline
from .speech import AudioToTextHFPipeline
from .pipeline_config import PipelineConfig, TextPipelineConfig, ASRPipelineConfig
from .pipeline import Pipeline


class PipelineFactory:
    """
    Factory class for creating pipeline instances.
    """

    @staticmethod
    def create_pipeline(config: PipelineConfig) -> Pipeline:
        if isinstance(config, TextPipelineConfig):
            return TextToTextHFPipeline(config=config)
        elif isinstance(config, ASRPipelineConfig):
            return AudioToTextHFPipeline(config=config)
        else:
            raise ValueError(f"Unsupported pipeline type: {type(config)}")

