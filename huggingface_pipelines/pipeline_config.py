from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """
    Configuration class for the SonarHFTextToTextPipeline.
    """
    encoder_model: str
    decoder_model: str
    dataset_name: str
    dataset_split: str
    source_lang: str
    target_lang: str
    batch_size: int
    num_shards: int = 1
    shard_id: int = 0
    low_bleu_threshold: float = 0.5
    device: str = 'cpu'
    cache_to_arrow: bool = False,
    output_file_name: str = "results"
    columns = ["text"]


@dataclass
class MetricConfig:
    """
    Configuration class for metrics.
    """
    metric_name: str
    low_score_threshold: float
