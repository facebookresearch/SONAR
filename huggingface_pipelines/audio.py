import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Audio  # type: ignore

from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

from .dataset import DatasetConfig  # type: ignore
from .pipeline import Pipeline, PipelineConfig  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioDatasetConfig(DatasetConfig):
    """
    Configuration for audio datasets.
    This class inherits from DatasetConfig and includes
    audio-specific attributes and processing.

    Attributes:
        sampling_rate (int): The target sampling rate for audio data.
        audio_column (str): The column that contains the audio data.

    Example:
        dataset_config = AudioDatasetConfig(
            dataset_name="librispeech_asr",
            dataset_split="train.clean.100",
            output_dir="/path/to/output",
            config="clean",
            trust_remote_code=True,
            sampling_rate=16000,
            audio_column="audio"
        )
    """

    sampling_rate: int = 16000
    audio_column: str = "audio"

    def load_dataset(self):
        """
        Loads and optionally shards the dataset based on the configuration settings.
        This method extends the base load_dataset method to include audio-specific processing.

        Returns:
            datasets.Dataset: The loaded, potentially sharded, and audio-processed dataset.

        Raises:
            ValueError: If the dataset cannot be loaded with the given configuration.
            ImportError: If the 'datasets' library is not installed.
        """
        dataset = super().load_dataset()
        return self.process_audio_column(dataset)

    def process_audio_column(self, dataset):
        """
        Processes the audio column of the dataset.

        Args:
            dataset (datasets.Dataset): The loaded dataset.

        Returns:
            datasets.Dataset: The dataset with processed audio column.
        """
        if self.audio_column in dataset.column_names:
            dataset = dataset.cast_column(
                self.audio_column, Audio(sampling_rate=self.sampling_rate)
            )
        else:
            raise ValueError(
                f"Error: {self.audio_column} column not found in the dataset. Skipping audio processing."
            )

        return dataset


@dataclass
class HFAudioToEmbeddingPipelineConfig(PipelineConfig):
    """
    Configuration class for HFAudioToEmbeddingPipeline.

    Attributes:
        encoder_model (str): The name or path of the encoder model to use.
        fbank_dtype (torch.dtype): The dtype for the fbank features. Defaults to torch.float32.
        n_parallel (int): Number of parallel processes for audio processing. Defaults to 4.
        pad_idx (int): The index used for padding. Defaults to 0.
        audio_column (str): The name of the column containing audio data. Defaults to "audio".
    Example:

        pipeline_config = HFAudioToEmbeddingPipelineConfig(
            encoder_model="sonar_speech_encoder_large",
            fbank_dtype=torch.float16,
            n_parallel=4,
            pad_idx=0,
            audio_column="audio",
            device="cuda",
            batch_size=32,
            columns=["audio", "text"],
            output_path="/path/to/output",
            output_column_suffix="embedding"
        )

    """

    encoder_model: str = "text_sonar_basic_encoder"
    fbank_dtype: torch.dtype = torch.float32
    n_parallel: int = 4
    pad_idx: int = 0
    dtype: np.dtype = np.dtype(np.float32)


class HFAudioToEmbeddingPipeline(Pipeline):
    """
    A pipeline for converting audio to embeddings using a HuggingFace model.

    This pipeline processes batches of audio data, converting them to embeddings
    using a specified encoder model. It handles temporary file creation for audio
    processing and ensures consistent embedding shapes across the batch.

    Attributes:
        config (HFAudioToEmbeddingPipelineConfig): The configuration for this pipeline.
        model (SpeechToEmbeddingModelPipeline): The underlying model used for embedding generation.

    Example:

        pipeline_config = HFAudioToEmbeddingPipelineConfig(
            encoder_model="sonar_speech_encoder",
            device="cuda",
            batch_size=16,
            audio_column="audio"
        )

        pipeline = HFAudioToEmbeddingPipeline(pipeline_config)
    """

    def __init__(self, config: HFAudioToEmbeddingPipelineConfig):
        """
        Initialize the HFAudioToEmbeddingPipeline.

        Args:
            config (HFAudioToEmbeddingPipelineConfig): The configuration for this pipeline.
        """
        super().__init__(config)
        self.config = config
        self.model = SpeechToEmbeddingModelPipeline(
            encoder=self.config.encoder_model,
            device=torch.device(self.config.device),
            fbank_dtype=self.config.fbank_dtype,
        )

    def collect_valid_audio_inputs(
        self, audio_data_list: List[Dict[str, Any]]
    ) -> List[torch.Tensor]:
        """
        Collect and process valid audio inputs from a list of audio data dictionaries.

        This method processes a list of audio data dictionaries, extracting valid audio inputs
        and converting them to PyTorch tensors. It handles multi-channel audio by taking the
        mean across channels and ensures that the output tensors are 2D with shape (1, num_samples).

        Args:
            audio_data_list (List[Dict[str, Any]]): A list of dictionaries containing audio data.
                Each dictionary is expected to have 'array' and 'sampling_rate' keys.

        Returns:
            List[torch.Tensor]: A list of valid audio inputs as PyTorch tensors.

        Raises:
            ValueError: If the input is not a list, if any audio data has an invalid format,
                        or if the resulting tensor has an unexpected shape.

        """
        audio_inputs = []

        # Ensure audio_data_list is always a list
        if not isinstance(audio_data_list, list):
            raise ValueError("Audio data must be in list format.")

        for audio_data in audio_data_list:
            if (
                isinstance(audio_data, dict)
                and "array" in audio_data
                and "sampling_rate" in audio_data
            ):
                # Handle multi-channel audio by taking the mean across channels
                audio_array = audio_data["array"]
                if audio_array.ndim > 1:
                    audio_array = np.mean(audio_array, axis=0)

                # Convert numpy array to torch tensor
                audio_tensor = torch.from_numpy(audio_array).float()

                # Ensure the tensor is 2D with shape (1, num_samples)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                elif audio_tensor.dim() > 2:
                    raise ValueError(
                        f"Unexpected audio tensor shape: {audio_tensor.shape}"
                    )

                audio_inputs.append(audio_tensor)
            else:
                logger.error(
                    f"Invalid audio data format in batch {audio_data_list}: {audio_data}"
                )
                raise ValueError(
                    f"Invalid audio data format in column {audio_data_list}: {audio_data}"
                )

        return audio_inputs

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of audio data, converting it to embeddings.

        This method handles the conversion of audio data to temporary WAV files,
        generates embeddings using the model, and ensures consistent embedding
        shapes across the batch.

        Args:
            batch (Dict[str, Any]): A dictionary containing the batch data.
                Expected to have an 'audio' key with a list of audio data dictionaries.

        Returns:
            Dict[str, Any]: The input batch dictionary with an additional key
                '{column}_{suffix}' containing the generated embeddings.

        Raises:
            Exception: If there's an error during batch processing or embedding generation.
        """

        try:
            for column in self.config.columns:
                if column not in batch:
                    logger.warning(f"Column {column} not found in batch. Skipping.")
                    continue

                audio_data_list: List[Dict[str, Any]] = batch[column]

                audio_inputs = self.collect_valid_audio_inputs(audio_data_list)

                if not audio_inputs:

                    raise ValueError(f"No valid audio inputs found in column {column}/")

                try:
                    # Move tensors to the specified device
                    audio_inputs = [
                        tensor.to(self.config.device) for tensor in audio_inputs
                    ]

                    all_embeddings = self.model.predict(
                        input=audio_inputs,
                        batch_size=self.config.batch_size,
                        n_parallel=self.config.n_parallel,
                        pad_idx=self.config.pad_idx,
                    )

                    final_embeddings = (
                        all_embeddings.cpu().numpy().astype(self.config.dtype)
                    )

                    batch[f"{column}_{self.config.output_column_suffix}"] = (
                        final_embeddings
                    )

                except Exception as e:
                    logger.error(
                        f"Error in model.predict for column {column}: {str(e)}"
                    )
                    raise ValueError(
                        f"Error in model.predict for column {column}: {str(e)}"
                    )

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            logger.error(f"Batch content: {batch}")
            raise ValueError(f"Error processing batch: {str(e)}")

        return batch


class AudioToEmbeddingPipelineFactory:
    """
    Factory class for creating AudioToEmbedding pipelines.

    This factory creates HFAudioToEmbeddingPipeline instances based on the provided configuration.

    Example:
        factory = AudioToEmbeddingPipelineFactory()
        config = {
            "encoder_model": "sonar_speech_encoder_large",
            "fbank_dtype": torch.float16,
            "n_parallel": 4,
            "pad_idx": 0,
            "audio_column": "audio",
            "device": "cuda",
            "batch_size": 32,
            "columns": ["audio"],
            "output_path": "/path/to/output",
            "output_column_suffix": "embedding"
        }
        pipeline = factory.create_pipeline(config)
    """

    def create_pipeline(self, config: Dict[str, Any]) -> Pipeline:
        """
        Create an AudioToEmbedding pipeline based on the provided configuration.

        Returns:
            Pipeline: An instance of HFAudioToEmbeddingPipeline.
        """
        pipeline_config = HFAudioToEmbeddingPipelineConfig(**config)
        return HFAudioToEmbeddingPipeline(pipeline_config)
