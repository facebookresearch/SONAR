import logging
from huggingface_pipelines.pipeline_config import TextPipelineConfig, MetricConfig
from huggingface_pipelines.pipeline_factory import PipelineFactory
from huggingface_pipelines.metric_analyzer_factory import MetricAnalyzerFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    text_config = Config(...)
    pipeline = SonarPipeline(config)
    dataset = hf.load_dataset(dataset_name)
    datasets = pipeline(dataset)

    for next(iter(dataset)):


if __name__ == "__main__":
    main()

