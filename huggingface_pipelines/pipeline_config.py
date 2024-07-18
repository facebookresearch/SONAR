from abc import ABC
from dataclasses import dataclass, replace
from typing import List, TypedDict


class PipelineOverwrites(TypedDict, total=False):
    batch_size: int
    device: str
    cache_to_arrow: bool
    take: int
    output_dir: str
    output_file_name: str
    columns: List[str]


class DatasetOverwrites(TypedDict, total=False):
    dataset_name: str
    dataset_split: str
    num_shards: int
    shard_id: int
    cache_dir: str


class TextToEmbeddingOverwrites(PipelineOverwrites, total=False):
    encoder_model: str
    source_lang: str


class EmbeddingToTextOverwrites(PipelineOverwrites, total=False):
    decoder_model: str
    target_lang: str


class AudioOverwrites(PipelineOverwrites, total=False):
    encoder_model: str
    decoder_model: str
    reference_transcriptions: str
    data_file: str
    audio_root_dir: str
    audio_path_index: int
    target_lang: str
    pad_idx: int
    fbank_dtype: str
    n_parallel: int


class MetricOverwrites(PipelineOverwrites, total=False):
    metric_name: str
    low_score_threshold: float


@dataclass
class PipelineConfig(ABC):
    """
    Abstract base class for pipeline configurations.

    Attributes:
        dataset_name (str): The name of the HuggingFace dataset to be used.
        dataset_split (str): The dataset split to be used (e.g., 'train', 'test', 'validation').
        batch_size (int): The batch size to be used for processing.
        device (str): The device to use for inference (e.g., 'cpu', 'cuda').
        pipeline_type (str): The type of pipeline (e.g., 'text', 'audio').
        num_shards (int): The number of shards to split the dataset into. Defaults to 1.
        shard_id (int): The ID of the shard to process. Defaults to 0.
        cache_to_arrow (bool): Whether to cache results to Arrow format. Defaults to False.
        output_dir (str): The directory to save the output to. Defaults to 'results'.
        take (int): The number of batches to take for processing. Defaults to -1 (process all).

    """
    columns: List[str]
    batch_size: int
    device: str
    take: int
    output_dir: str
    output_file_name: str
    cache_to_arrow: bool = False

    def with_overwrites(self, overwrites: PipelineOverwrites):
        return replace(self, **overwrites)

# TODO: Apply transformer to detect language : Not critical


@dataclass
class DatasetConfig():
    """
    Configuration class for loading and sharding datasets.

    Attributes:
        dataset_name (str): The name of the dataset to load.
        dataset_split (str): The split of the dataset to use (e.g., 'train', 'test', 'validation').
        num_shards (int): The number of shards to split the dataset into. Defaults to 1.
        shard_id (int): The ID of the shard to retrieve. Defaults to 0.
        cache_dir (str): The directory to cache the loaded dataset. Defaults to None.
        keep_in_memory (bool): Whether to keep the dataset in memory after loading. Defaults to False.
        trust_remote_code (bool): Whether to trust remote code when loading the dataset. Defaults to False.
    """
    dataset_name: str
    dataset_split: str
    config: str = None
    num_shards: int = 1
    shard_id: int = 0
    cache_dir: str = None
    trust_remote_code: bool = False

    def load_dataset(self):
        """
        Loads and shards the dataset based on the configuration settings.

        Returns:
            datasets.Dataset: The loaded and sharded dataset.
        """
        from datasets import load_dataset

        # Load the dataset
        dataset = load_dataset(
            self.dataset_name,
            split=self.dataset_split,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            config=self.config
        )

        # Shard the dataset
        if self.num_shards > 1:
            dataset = dataset.shard(
                num_shards=self.num_shards, index=self.shard_id)

        return dataset

    def with_overwrites(self, overwrites: DatasetOverwrites):
        return replace(self, **overwrites)


@dataclass
class TextToEmbeddingPipelineConfig(PipelineConfig):
    """
    Configuration class for text-to-embedding pipelines.

    Attributes:
        encoder_model (str): The name or path of the model to be used for encoding texts into embeddings.
        source_lang (str): The source language code for the texts to be encoded.
    """
    encoder_model: str = "text_sonar_basic_encoder"
    source_lang: str = "eng_Latn"

    def with_overwrites(self, overwrites: TextToEmbeddingOverwrites):
        return replace(self, **overwrites)


@dataclass
class EmbeddingToTextPipelineConfig(PipelineConfig):
    """
    Configuration class for embedding-to-text pipelines.

    Attributes:
        decoder_model (str): The name or path of the model to be used for decoding embeddings back into texts.
        target_lang (str): The target language code for the texts to be decoded.
    """
    decoder_model: str = "text_sonar_basic_decoder"
    target_lang: str = "eng_Latn"

    def with_overwrites(self, overwrites: EmbeddingToTextOverwrites):
        return replace(self, **overwrites)


@dataclass
class AudioPipelineConfig(PipelineConfig):
    """
    Configuration class for ASR pipelines.
    """
    encoder_model: str = "sonar_speech_encoder_eng"
    decoder_model: str = "text_sonar_basic_decoder"
    target_lang: str = None
    pad_idx: int = 0
    fbank_dtype: str = None
    n_parallel: int = 4

    def with_overwrites(self, overwrites: AudioOverwrites):
        return replace(self, **overwrites)


@dataclass
class MetricPipelineConfig(PipelineConfig):
    """
    Configuration class for metrics.

    Attributes:
        metric_name (str): The name of the metric to be used for evaluation.
        low_score_threshold (float): The threshold below which the score is considered low.
    """
    metric_name: str = "bleu"
    low_score_threshold: float = 0.5

    def with_overwrites(self, overwrites: MetricOverwrites):
        return replace(self, **overwrites)
