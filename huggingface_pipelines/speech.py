import json
import logging
from datasets import load_dataset, Dataset
from typing import List, Dict, Any
from dataclasses import dataclass
from .pipeline import Pipeline
from .pipeline_config import ASRPipelineConfig
from sonar.inference_pipelines.speech import SpeechInferenceParams, SpeechToTextPipeline
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioToTextHFPipeline(Pipeline):
    """
    A pipeline for transcribing audio datasets from HuggingFace into text and evaluating the quality using metrics.
    """
    config: ASRPipelineConfig

    def __post_init__(self):
        """
        Initializes the dataset, models, and metric after the instance is created.
        """
        logger.info(
            f"Loading dataset {self.config.dataset_name} with split {self.config.dataset_split}...")
        self.dataset = load_dataset(
            self.config.dataset_name, split=self.config.dataset_split)
        logger.info("Dataset loaded. Initializing models...")
        self.speech_to_text_pipeline = SpeechToTextPipeline.load_from_name(
            encoder_name=self.config.model_name,
            decoder_name=self.config.model_name
        )
        logger.info("Models initialized.")

    def transcribe_audio(self, audio_data: List[Any]) -> List[str]:
        """
        Transcribes a list of audio data to text.

        Args:
            audio_data (List[Any]): A list of audio data to be transcribed.

        Returns:
            List[str]: A list of transcribed texts.
        """
        try:
            logger.info(f"Transcribing {len(audio_data)} audio samples...")
            speech_ctx = SpeechInferenceParams(
                data_file=self.config.data_file,
                audio_root_dir=self.config.audio_root_dir,
                audio_path_index=self.config.audio_path_index,
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

    def process_batch(self, batch: Dict[str, Any], dataset: Dataset) -> Dataset:
        """
        Processes a single batch of data by transcribing audio and updating the dataset.

        Args:
            batch (Dict[str, Any]): A batch of data containing audio.
            dataset (Dataset): The dataset to update.

        Returns:
            Dataset: The updated dataset.
        """
        for column in self.config.columns:
            audio_data = batch[column]
            transcribed_texts = self.transcribe_audio(audio_data)
            dataset = dataset.add_column(
                column + '_transcribed', transcribed_texts)
        return dataset

    def cache_results(self):
        """
        Caches the results to a JSON file.

        The results are saved in a file named 'output_file_name_shard_{shard_id}.json'.
        """
        try:
            file_name = f'{self.config.output_file_name}_shard_{self.config.shard_id}.json'
            logger.info(f"Caching results to {file_name}...")
            with open(file_name, 'w') as f:
                json.dump(self.results, f)
            logger.info("Results cached successfully.")
        except Exception as e:
            logger.error(f"Error caching results: {e}")

    def cache_results_arrow(self):
        """
        Caches the results to an Arrow file.

        The results are saved in a file named 'output_file_name_shard_{shard_id}.arrow'.
        """
        try:
            file_name = f'{self.config.output_file_name}_shard_{self.config.shard_id}.arrow'
            logger.info(f"Caching results to {file_name}...")
            dataset = Dataset.from_dict({
                "original": [result['original'] for result in self.results],
                "transcribed": [result['transcribed'] for result in self.results],
                "reference": [result['reference'] for result in self.results]
            })
            dataset.save_to_disk(file_name)
            logger.info("Results cached successfully.")
        except Exception as e:
            logger.error(f"Error caching results: {e}")

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
            if self.config.num_shards > 1:
                dataset = dataset.shard(
                    num_shards=self.config.num_shards, index=self.config.shard_id)

            updated_dataset = dataset.map(
                lambda batch: self.process_batch(batch, dataset),
                batched=True,
                batch_size=self.config.batch_size,
                load_from_cache_file=False
            )
            return updated_dataset
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

