import logging
import os
from huggingface_pipelines.pipeline_config import (
    MetricPipelineConfig,
    TextToEmbeddingPipelineConfig,
    EmbeddingToTextPipelineConfig,
    DatasetConfig,
    PipelineOverwrites,
    TextToEmbeddingOverwrites,
    EmbeddingToTextOverwrites,
    MetricOverwrites
)
from huggingface_pipelines.text import HFEmbeddingToTextPipeline, HFTextToEmbeddingPipeline
from huggingface_pipelines.metric_analyzer import MetricAnalyzerPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configure and load the initial dataset
    dataset_config = DatasetConfig(
        dataset_name="ag_news",
        dataset_split="test"
    )
    dataset = dataset_config.load_dataset()

    # Build configuration for text to embedding pipeline
    text_to_embedding_config = TextToEmbeddingPipelineConfig(
        columns=["text"],
        batch_size=32,
        device="cpu",
        take=1,
        output_dir="results",
        output_file_name="ag_news_results",
        encoder_model="text_sonar_basic_encoder",
        source_lang="eng_Latn"
    )
    text_to_embedding_overwrites = TextToEmbeddingOverwrites(
        cache_to_arrow=True
    )
    text_to_embedding_config = text_to_embedding_config.with_overwrites(
        text_to_embedding_overwrites)

    # Initialize and run the text to embedding pipeline
    text_to_embedding_pipeline = HFTextToEmbeddingPipeline(
        text_to_embedding_config)
    dataset = text_to_embedding_pipeline(dataset)

    # Build configuration for embedding to text pipeline
    embedding_to_text_config = EmbeddingToTextPipelineConfig(
        columns=["text"],
        batch_size=32,
        device="cpu",
        take=1,
        output_dir="results",
        output_file_name="ag_news_results",
        decoder_model="text_sonar_basic_decoder",
        target_lang="eng_Latn"
    )
    embedding_to_text_overwrites = EmbeddingToTextOverwrites(
        cache_to_arrow=True
    )
    embedding_to_text_config = embedding_to_text_config.with_overwrites(
        embedding_to_text_overwrites)

    # Initialize and run the embedding to text pipeline
    embedding_to_text_pipeline = HFEmbeddingToTextPipeline(
        embedding_to_text_config)
    dataset = embedding_to_text_pipeline(dataset)

    # Initialize the metric pipeline config
    metric_config = MetricPipelineConfig(
        columns=["text"],
        batch_size=5,
        device="cpu",
        take=1,
        output_dir="results",
        output_file_name="ag_news_results",
        metric_name="bleu",
        low_score_threshold=0.5
    )
    metric_overwrites = MetricOverwrites(
        cache_to_arrow=True
    )
    metric_config = metric_config.with_overwrites(metric_overwrites)

    metrics_pipeline = MetricAnalyzerPipeline(metric_config)

    # Run metrics pipeline
    dataset = metrics_pipeline(dataset)

    # Save the dataset to disk
    output_path = os.path.join(
        metric_config.output_dir, f'{metric_config.output_file_name}.parquet')
    dataset.save_to_disk(output_path)


if __name__ == "__main__":
    main()

