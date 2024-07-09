from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineConfig:
    """
    Configuration class for the SonarHFTextToTextPipeline.

    Attributes:
        encoder_model (str): The name or path of the model to be used for encoding texts into embeddings.
        decoder_model (str): The name or path of the model to be used for decoding embeddings back into texts.
        dataset_name (str): The name of the HuggingFace dataset to be used.
        dataset_split (str): The dataset split to be used (e.g., 'train', 'test', 'validation').
        source_lang (str): The source language code for the texts to be encoded.
        target_lang (str): The target language code for the texts to be decoded.
        batch_size (int): The batch size to be used for encoding and decoding.
        num_shards (int): The number of shards to split the dataset into. Defaults to 1.
        shard_id (int): The ID of the shard to process. Defaults to 0.
        low_bleu_threshold (float): The threshold for reporting low BLEU scores. Defaults to 0.5.
        device (str): The device to use for inference (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
        cache_to_arrow (bool): Whether to cache results to Arrow format. Defaults to False.
        output_file_name (str): The base name of the file where results will be saved. Defaults to "results".
        columns (List[str]): The columns of the dataset to process. Defaults to ["text"].
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
    cache_to_arrow: bool = False
    output_file_name: str = "results"
    columns: List[str] = field(default_factory=lambda: ["text"])


@dataclass
class MetricConfig:
    """
    Configuration class for metrics.

    Attributes:
        metric_name (str): The name of the metric to be used for evaluation.
        low_score_threshold (float): The threshold below which the score is considered low.
    """
    metric_name: str
    low_score_threshold: float

