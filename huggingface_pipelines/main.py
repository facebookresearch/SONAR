import logging
from .pipeline_config import PipelineConfig, MetricConfig
from .pipeline_factory import PipelineFactory
from .metric_analyzer import MetricAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():

    pipeline_config = PipelineConfig(
        encoder_model="text_sonar_basic_encoder",
        decoder_model="text_sonar_basic_decoder",
        dataset_name="ag_news",
        dataset_split="test",
        source_lang="eng_Latn",
        target_lang="eng_Latn",
        batch_size=5,
        num_shards=1,
        shard_id=0,
        device="cpu",
        cache_to_arrow=True
    )

    metric_config = MetricConfig(
        metric_name="bleu",
        low_score_threshold=0.5
    )

    pipeline = PipelineFactory.create_pipeline(pipeline_config)
    pipeline.process_batches()

    metric_analyzer = MetricAnalyzer(metric_config)
    metric_analyzer.analyze_results(pipeline.results)

if __name__ == "__main__":
    main()