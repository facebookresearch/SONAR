from abc import ABC
from dataclasses import dataclass, replace
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

    def with_dataset_name(self, dataset_name: str):
        return replace(self, dataset_name=dataset_name)

    def with_dataset_split(self, dataset_split: str):
        return replace(self, dataset_split=dataset_split)

    def with_batch_size(self, batch_size: int):
        return replace(self, batch_size=batch_size)

    def with_device(self, device: str):
        return replace(self, device=device)

    def with_pipeline_type(self, pipeline_type: str):
        return replace(self, pipeline_type=pipeline_type)

    def with_num_shards(self, num_shards: int):
        return replace(self, num_shards=num_shards)

    def with_shard_id(self, shard_id: int):
        return replace(self, shard_id=shard_id)

    def with_cache_to_arrow(self, cache_to_arrow: bool):
        return replace(self, cache_to_arrow=cache_to_arrow)

    def with_output_file_name(self, output_file_name: str):
        return replace(self, output_file_name=output_file_name)


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
    columns: List[str]

    def with_encoder_model(self, encoder_model: str):
        return replace(self, encoder_model=encoder_model)

    def with_decoder_model(self, decoder_model: str):
        return replace(self, decoder_model=decoder_model)

    def with_source_lang(self, source_lang: str):
        return replace(self, source_lang=source_lang)

    def with_target_lang(self, target_lang: str):
        return replace(self, target_lang=target_lang)

    def with_columns(self, columns: List[str]):
        return replace(self, columns=columns)


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
    audio_column: str
    reference_transcriptions: str = "reference_transcription"
    data_file: str = None
    audio_root_dir: str = None
    audio_path_index: int = None
    target_lang: str = None
    pad_idx: int = 0
    fbank_dtype: str = None
    n_parallel: int = 4

    def with_model_name(self, model_name: str):
        return replace(self, model_name=model_name)

    def with_audio_column(self, audio_column: str):
        return replace(self, audio_column=audio_column)

    def with_reference_transcriptions(self, reference_transcriptions: str):
        return replace(self, reference_transcriptions=reference_transcriptions)

    def with_data_file(self, data_file: str):
        return replace(self, data_file=data_file)

    def with_audio_root_dir(self, audio_root_dir: str):
        return replace(self, audio_root_dir=audio_root_dir)

    def with_audio_path_index(self, audio_path_index: int):
        return replace(self, audio_path_index=audio_path_index)

    def with_target_lang(self, target_lang: str):
        return replace(self, target_lang=target_lang)

    def with_pad_idx(self, pad_idx: int):
        return replace(self, pad_idx=pad_idx)

    def with_fbank_dtype(self, fbank_dtype: str):
        return replace(self, fbank_dtype=fbank_dtype)

    def with_n_parallel(self, n_parallel: int):
        return replace(self, n_parallel=n_parallel)


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

    def with_metric_name(self, metric_name: str):
        return replace(self, metric_name=metric_name)

    def with_low_score_threshold(self, low_score_threshold: float):
        return replace(self, low_score_threshold=low_score_threshold)
