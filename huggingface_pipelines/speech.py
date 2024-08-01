import os
import torch
from typing import Dict, Any
from dataclasses import dataclass, replace
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
import tempfile
import soundfile as sf
import logging
from .pipeline import Pipeline, PipelineConfig, PipelineOverwrites

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFSpeechToEmbeddingOverwrites(PipelineOverwrites, total=False):
    """
    Overwrite options for HFSpeechToEmbeddingPipeline configuration.

    Attributes:
        encoder_model (str): The name or path of the encoder model to use.
        fbank_dtype (torch.dtype): The dtype for the fbank features.
        n_parallel (int): Number of parallel processes for audio processing.
        pad_idx (int): The index used for padding.
        audio_column (str): The name of the column containing audio data.
    """
    encoder_model: str
    fbank_dtype: torch.dtype
    n_parallel: int
    pad_idx: int
    audio_column: str


@dataclass
class HFSpeechToEmbeddingPipelineConfig(PipelineConfig):
    """
    Configuration class for HFSpeechToEmbeddingPipeline.

    Attributes:
        encoder_model (str): The name or path of the encoder model to use.
        fbank_dtype (torch.dtype): The dtype for the fbank features. Defaults to torch.float32.
        n_parallel (int): Number of parallel processes for audio processing. Defaults to 4.
        pad_idx (int): The index used for padding. Defaults to 0.
        audio_column (str): The name of the column containing audio data. Defaults to "audio".
    """
    encoder_model: str
    fbank_dtype: torch.dtype = torch.float32
    n_parallel: int = 4
    pad_idx: int = 0
    audio_column: str = "audio"

    def with_overwrites(self, overwrites: HFSpeechToEmbeddingOverwrites) -> 'HFSpeechToEmbeddingPipelineConfig':
        """
        Create a new configuration with the specified overwrites.

        Args:
            overwrites (HFSpeechToEmbeddingOverwrites): Overwrite values for the configuration.

        Returns:
            HFSpeechToEmbeddingPipelineConfig: A new configuration instance with applied overwrites.
        """
        return replace(self, **overwrites)


class HFSpeechToEmbeddingPipeline(Pipeline):
    """
    A pipeline for converting speech audio to embeddings using a HuggingFace model.

    This pipeline processes batches of audio data, converting them to embeddings
    using a specified encoder model. It handles temporary file creation for audio
    processing and ensures consistent embedding shapes across the batch.

    Attributes:
        config (HFSpeechToEmbeddingPipelineConfig): The configuration for this pipeline.
        model (SpeechToEmbeddingModelPipeline): The underlying model used for embedding generation.
    """

    def __init__(self, config: HFSpeechToEmbeddingPipelineConfig):
        """
        Initialize the HFSpeechToEmbeddingPipeline.

        Args:
            config (HFSpeechToEmbeddingPipelineConfig): The configuration for this pipeline.
        """
        self.config = config
        self.model = SpeechToEmbeddingModelPipeline(
            encoder=self.config.encoder_model,
            device=torch.device(self.config.device),
            fbank_dtype=self.config.fbank_dtype
        )

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
                '{audio_column}_embedding' containing the generated embeddings.

        Raises:
            Exception: If there's an error during batch processing or embedding generation.
        """
        try:
            audio_inputs = []
            temp_files = []

            for audio_data in batch[self.config.audio_column]:
                if isinstance(audio_data, dict) and 'array' in audio_data and 'sampling_rate' in audio_data:
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False, suffix='.wav')
                    temp_files.append(temp_file)
                    sf.write(temp_file.name,
                             audio_data['array'], audio_data['sampling_rate'])
                    audio_inputs.append(temp_file.name)
                else:
                    logger.warning(f"Invalid audio data format: {audio_data}")

            if not audio_inputs:
                logger.warning("No valid audio inputs found in batch.")
                return batch

            try:
                all_embeddings = self.model.predict(
                    input=audio_inputs,
                    batch_size=self.config.batch_size,
                    n_parallel=self.config.n_parallel,
                    pad_idx=self.config.pad_idx
                )

                # Ensure all embeddings are 2D
                all_embeddings = [emb.unsqueeze(0) if emb.dim(
                ) == 1 else emb for emb in all_embeddings]

                # Get the maximum sequence length and embedding dimension
                max_seq_len = max(emb.shape[0] for emb in all_embeddings)

                # Pad embeddings to have the same sequence length
                padded_embeddings = [torch.nn.functional.pad(
                    emb, (0, 0, 0, max_seq_len - emb.shape[0])) for emb in all_embeddings]

                # Stack embeddings into a single tensor
                stacked_embeddings = torch.stack(padded_embeddings)

                batch[f"{self.config.audio_column}_embedding"] = stacked_embeddings.cpu(
                ).numpy()

            except Exception as e:
                logger.error(f"Error in model.predict: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            logger.error(f"Batch content: {batch}")
            raise

        finally:
            for temp_file in temp_files:
                temp_file.close()
                os.unlink(temp_file.name)

        return batch

