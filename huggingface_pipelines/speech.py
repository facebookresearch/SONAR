import logging
from datasets import Dataset
from typing import List, Dict, Any
from dataclasses import dataclass
from .pipeline import Pipeline
from .pipeline_config import AudioPipelineConfig
from sonar.inference_pipelines.speech import SpeechInferenceParams, SpeechToTextPipeline
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioToTextHFPipeline(Pipeline):
    """
    A pipeline for transcribing audio datasets from HuggingFace into text using SONAR.
    """
    config: AudioPipelineConfig

    def __post_init__(self):
        """
        Initializes the SONAR models.
        """
        self.speech_to_text_pipeline = SpeechToTextPipeline.load_from_name(
            encoder_name=self.config.encoder_model,
            decoder_name=self.config.decoder_model
        )
        logger.info("SONAR models initialized.")

    def transcribe_audio(self, audio_data: List[Dict[str, Any]]) -> List[str]:
        """
        Transcribes a list of audio data to text using SONAR.
        Args:
            audio_data (List[Dict[str, Any]]): A list of audio data dictionaries.
        Returns:
            List[str]: A list of transcribed texts.
        """
        try:
            logger.info(f"Transcribing {len(audio_data)} audio samples...")
            speech_ctx = SpeechInferenceParams(
                target_lang=self.config.target_lang,
                batch_size=self.config.batch_size,
                pad_idx=self.config.pad_idx,
                device=self.config.device,
                fbank_dtype=self.config.fbank_dtype,
                n_parallel=self.config.n_parallel
            )
            speech_to_text_dp = self.speech_to_text_pipeline.build_pipeline(
                speech_ctx)

            transcriptions = []
            with torch.inference_mode():
                for batch in speech_to_text_dp:
                    transcriptions.extend(batch)

            logger.info("Audio transcribed successfully.")
            return transcriptions
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Processes a batch of data by transcribing audio.
        Args:
            batch (Dict[str, Any]): A batch of data containing audio.
        Returns:
            Dict[str, List[str]]: A dictionary with transcribed texts.
        """
        result = {}
        for column in self.config.columns:
            audio_data = batch[column]
            transcribed_texts = self.transcribe_audio(audio_data)
            result[column] = transcribed_texts
        return result

    def __call__(self, dataset: Dataset) -> Dataset:
        """
        Processes the dataset and updates it.
        Args:
            dataset (Dataset): The dataset to process.
        Returns:
            Dataset: The updated dataset.
        """
        try:
            logger.info("Starting to process dataset...")
            updated_dataset = dataset.map(
                self.process_batch,
                batched=True,
                batch_size=self.config.batch_size,
                remove_columns=dataset.column_names,
                load_from_cache_file=False,
                desc="Transcribing audio"
            )
            return updated_dataset
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise
