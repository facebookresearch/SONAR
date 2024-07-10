import logging
from huggingface_pipelines.pipeline_config import TextPipelineConfig, MetricConfig
from huggingface_pipelines.pipeline_factory import PipelineFactory
from huggingface_pipelines.metric_analyzer_factory import MetricAnalyzerFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():

    pipeline_config = TextPipelineConfig(
        encoder_model="text_sonar_basic_encoder",
        decoder_model="text_sonar_basic_decoder",
        dataset_name="ag_news",
        dataset_split="test",
        source_lang="eng_Latn",
        target_lang="eng_Latn",
        batch_size=5,
        columns=["text"],  # Specify the columns to process
        num_shards=1,
        shard_id=0,
        device="cpu",
        cache_to_arrow=True,
        output_file_name="ag_news_results",
    )

    metric_config = MetricConfig(
        metric_name="bleu",
        low_score_threshold=0.5
    )

    pipeline = PipelineFactory.create_pipeline(pipeline_config)
    pipeline.process_batches()

    metric_analyzer = MetricAnalyzerFactory.create_analyzer(metric_config)
    metric_analyzer.analyze_results(pipeline.results)


if __name__ == "__main__":
    main()

