from abc import ABC
from dataclasses import dataclass, field
from typing import List


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
        output_file_name (str): The base name of the file where results will be saved. Defaults to "results".
    """
    dataset_name: str
    dataset_split: str = "test"
    batch_size: int = 5
    device: str = "cpu"
    num_shards: int = 1
    shard_id: int = 0
    cache_to_arrow: bool = False
    output_file_name: str = "results"


@dataclass
class TextPipelineConfig(PipelineConfig):
    """
    Configuration class for text pipelines.

    Attributes:
        encoder_model (str): The name or path of the model to be used for encoding texts into embeddings.
        decoder_model (str): The name or path of the model to be used for decoding embeddings back into texts.
        source_lang (str): The source language code for the texts to be encoded.
        target_lang (str): The target language code for the texts to be decoded.
        columns (List[str]): The columns of the dataset to process. Defaults to ["text"].
    """
    encoder_model: str = "text_sonar_basic_encoder"
    decoder_model: str = "text_sonar_basic_decoder"
    source_lang: str = "eng_Latn"
    target_lang: str = "eng_Latn"
    columns: List[str] = field(default_factory=lambda: ["text"])


@dataclass
class ASRPipelineConfig(PipelineConfig):
    """
    Configuration class for ASR pipelines.

    Attributes:
        model_name (str): The name or path of the model to be used for transcribing audio to text.
        audio_column (str): The column of the dataset containing audio data.
        reference_transcriptions (str): The column of the dataset containing reference transcriptions.
        data_file (str): The file containing audio data paths and metadata.
        audio_root_dir (str): The root directory where audio files are stored.
        audio_path_index (int): The index of the audio path in the data file.
        target_lang (str): The target language for transcription.
        pad_idx (int): Padding index for batching.
        fbank_dtype (str): The data type for filter bank features.
        n_parallel (int): Number of parallel workers.
    """
    encoder_model = "sonar_speech_encoder_eng"
    decoder_model = "text_sonar_basic_decoder"
    audio_column: str = "audio"
    reference_transcriptions: str = "reference_transcription"
    data_file: str = None
    audio_root_dir: str = None
    audio_path_index: int = None
    target_lang: str = None
    pad_idx: int = 0
    fbank_dtype: str = None
    n_parallel: int = 4


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

