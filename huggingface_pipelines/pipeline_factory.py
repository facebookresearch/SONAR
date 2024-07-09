

class PipelineFactory:
    """
    Factory class for creating SonarHFTextToTextPipeline instances.
    """

    @staticmethod
    def create_pipeline(config: PipelineConfig) -> SonarHFTextToTextPipeline:
        return SonarHFTextToTextPipeline(config=config)