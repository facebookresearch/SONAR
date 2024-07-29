import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, replace
from .pipeline import Pipeline, PipelineConfig, PipelineOverwrites
import torch
from sonar.inference_pipelines.speech import SpeechToTextModelPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechToTextOverwrites(PipelineOverwrites, total=False):
    encoder_model: str
    decoder_model: str
    target_lang: str
    pad_idx: int
    fbank_dtype: str
    n_parallel: int


@dataclass
class SpeechToTextPipelineConfig(PipelineConfig):
    """
    Configuration class for speech-to-text pipelines.
    """
    encoder_model: str = "sonar_speech_encoder_eng"
    decoder_model: str = "text_sonar_basic_decoder"
    target_lang: Optional[str] = None
    pad_idx: int = 0
    fbank_dtype: torch.dtype = torch.float32
    n_parallel: int = 4

    def with_overwrites(self, overwrites: SpeechToTextOverwrites):
        return replace(self, **overwrites)


@dataclass
class HFSpeechToTextPipeline(Pipeline):
    """
    A pipeline for transcribing preprocessed audio data into text using SONAR.
    """
    config: SpeechToTextPipelineConfig

    def __post_init__(self):
        """
        Initializes the SONAR SpeechToTextModelPipeline.
        """
        self.sonar_pipeline = SpeechToTextModelPipeline(
            encoder=self.config.encoder_model,
            decoder=self.config.decoder_model,
            tokenizer=self.config.decoder_model,
            device=self.config.device,
            fbank_dtype=self.fbank_dtype
        )
        logger.info("SONAR SpeechToTextModelPipeline initialized.")

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a batch of data by transcribing preprocessed audio.
        """
        for column in self.config.columns:
            if column not in batch:
                logger.warning(f"Column {column} not found in batch.")
                continue

            audio_features = batch[f"{column}_preprocessed"]

            # Convert audio features to the format expected by SONAR pipeline
            audio_input = [torch.tensor(features)
                           for features in audio_features]

            transcribed_texts = self.sonar_pipeline.predict(
                input=audio_input,
                target_lang=self.config.target_lang,
                batch_size=self.config.batch_size,
                n_parallel=self.config.n_parallel,
                pad_idx=self.config.pad_idx
            )

            batch[f"{column}_transcribed"] = transcribed_texts

        return batch

    def __call__(self, dataset):
        """
        Processes the dataset and updates it with transcriptions.
        """
        try:
            logger.info("Starting to transcribe dataset...")
            updated_dataset = super().__call__(dataset)
            logger.info("Transcription completed.")
            return updated_dataset
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

