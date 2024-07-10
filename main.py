import logging
from huggingface_pipelines.pipeline_config import MetricPipelineConfig, TextToEmbeddingPipelineConfig, EmbeddingToTextPipelineConfig
from huggingface_pipelines.text import HFEmbeddingToTextPipeline,  HFTextToEmbeddingPipeline
from huggingface_pipelines.metric_analyzer import MetricAnalyzerPipeline
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():

    # Load the initial dataset
    dataset = load_dataset("ag_news", split="test")

    # Build configuration for text to embedding pipeline
    text_to_embedding_config = TextToEmbeddingPipelineConfig(
        dataset_name="ag_news",
        columns=["text"]
    ).with_encoder_model("text_sonar_basic_encoder")\
        .with_source_lang("eng_Latn")\
        .with_num_shards(1)\
        .with_shard_id(0)\
        .with_cache_to_arrow(True)\
        .with_output_file_name("ag_news_results")

    # Initialize and run the text to embedding pipeline
    text_to_embedding_pipeline = HFTextToEmbeddingPipeline(
        text_to_embedding_config)
    dataset = text_to_embedding_pipeline(dataset)

    # Build configuration for embedding to text pipeline
    embedding_to_text_config = EmbeddingToTextPipelineConfig(
        dataset_name="ag_news",
        columns=["text_embeddings"]
    ).with_decoder_model("text_sonar_basic_decoder")\
        .with_target_lang("eng_Latn")\
        .with_num_shards(1)\
        .with_shard_id(0)\
        .with_cache_to_arrow(True)\
        .with_output_file_name("ag_news_results")

    # Initialize and run the embedding to text pipeline
    embedding_to_text_pipeline = HFEmbeddingToTextPipeline(
        embedding_to_text_config)
    dataset = embedding_to_text_pipeline(dataset)

    # Initialize the metric pipeline config
    metric_config = MetricPipelineConfig(
        dataset_name="ag_news",
        dataset_split="test",
        batch_size=5,
        device="cpu",
        pipeline_type="text",
        columns=["text"],
        metric_name="bleu",
        low_score_threshold=0.5
    ).with_num_shards(1)\
        .with_shard_id(0)\
        .with_cache_to_arrow(True)\
        .with_output_file_name("ag_news_results")

    metrics_pipeline = MetricAnalyzerPipeline(metric_config)

    # Run metrics pipeline

    dataset = metrics_pipeline(dataset)

    # Save the dataset to disk
    dataset.save_to_disk(f'{metric_config.output_file_name}.arrow')


if __name__ == "__main__":
    main()

