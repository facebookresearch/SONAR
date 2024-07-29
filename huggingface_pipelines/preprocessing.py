from transformers import WhisperProcessor, WhisperFeatureExtractor
import numpy as np
import spacy
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from .pipeline import Pipeline, PipelineConfig
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPACY_MODELS = {
    "eng_Latn": "en_core_web_sm",
    "fra_Latn": "fr_core_news_sm",
    "deu_Latn": "de_core_news_sm",
    "spa_Latn": "es_core_news_sm",
    "ita_Latn": "it_core_news_sm",
    "por_Latn": "pt_core_news_sm",
    "nld_Latn": "nl_core_news_sm",
}


@dataclass
class PreprocessingPipelineConfig(PipelineConfig):
    """
    Configuration class for preprocessing pipelines.
    Add any specific preprocessing configuration parameters here.
    """
    pass


class PreprocessingPipeline(Pipeline, ABC):
    """
    Abstract base class for preprocessing pipelines.
    """

    def __init__(self, config: PreprocessingPipelineConfig):
        super().__init__(config)

    @abstractmethod
    def preprocess_text(self, text: str) -> List[str]:
        """
        Abstract method to preprocess a single text.
        """
        pass

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a batch of data by applying preprocessing to specified columns.
        """
        for column in self.config.columns:
            if column in batch:
                batch[f"{column}_preprocessed"] = [
                    self.preprocess_text(text) for text in batch[column]]
            else:
                logger.warning(f"Column {column} not found in batch.")
        return batch


@dataclass
class TextPreprocessingPipelineConfig(PreprocessingPipelineConfig):
    """
    Configuration class for text preprocessing pipelines.
    """
    handle_missing: str = 'skip'  # Options: 'skip', 'remove', or 'fill'
    fill_value: Optional[str] = None  # Used when handle_missing is 'fill'
    source_lang: str = "eng_Latn"


@dataclass
class TextPreprocessingPipeline(PreprocessingPipeline):
    """
    A pipeline for preprocessing text data, including handling of null and missing values,
    and performing sentence segmentation.
    """

    config: TextPreprocessingPipelineConfig

    def __post_init__(self):
        self.nlp = self.load_spacy_model(self.config.source_lang)
        logger.info("Model initialized.")

    def load_spacy_model(self, lang_code: str):
        """
        Loads the appropriate spaCy model based on the language code.
        """
        if lang_code not in SPACY_MODELS:
            raise ValueError(
                f"Unsupported language code: {lang_code}. Please add it to the SPACY_MODELS dictionary.")

        model_name = SPACY_MODELS[lang_code]
        try:
            return spacy.load(model_name)
        except OSError:
            logger.warning(
                f"SpaCy model {model_name} not found. Attempting to download...")
            spacy.cli.download(model_name)
            return spacy.load(model_name)

    def preprocess_text(self, text: Optional[str]) -> List[str]:
        """
        Preprocesses a single text by segmenting it into sentences.
        Handles null or missing values according to the configuration.
        """
        if text is None or (isinstance(text, str) and text.strip() == ''):
            if self.config.handle_missing == 'skip':
                return []
            elif self.config.handle_missing == 'remove':
                return []
            elif self.config.handle_missing == 'fill':
                return [self.config.fill_value] if self.config.fill_value else []
            else:
                raise ValueError(
                    f"Invalid handle_missing option: {self.config.handle_missing}")

        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioPreprocessingPipelineConfig(PipelineConfig):
    """
    Configuration class for audio preprocessing pipelines.
    """
    whisper_model: str = "openai/whisper-base"
    # Maximum duration in seconds, None for no limit
    max_duration: Optional[float] = None


@dataclass
class AudioPreprocessingPipeline(Pipeline):
    """
    A pipeline for preprocessing audio data using Whisper's feature extractor.
    """
    config: AudioPreprocessingPipelineConfig

    def __post_init__(self):
        self.whisper_processor = WhisperProcessor.from_pretrained(
            self.config.whisper_model)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config.whisper_model)
        logger.info("Whisper processor and feature extractor initialized.")

    def preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocesses a single audio array using Whisper's feature extractor.
        """
        # Convert to float32 if not already
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Limit duration if specified
        if self.config.max_duration is not None:
            max_samples = int(self.config.max_duration * sample_rate)
            audio = audio[:max_samples]

        # Extract features using Whisper's feature extractor
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="np"
        )

        return inputs.input_features.squeeze()

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a batch of audio data.
        """
        for column in self.config.columns:
            if column not in batch:
                logger.warning(f"Column {column} not found in batch.")
                continue

            audio_data = batch[column]
            processed_audio = []

            for audio in audio_data:
                if isinstance(audio, dict) and 'array' in audio and 'sampling_rate' in audio:
                    audio_array, sample_rate = audio['array'], audio['sampling_rate']
                elif isinstance(audio, np.ndarray):
                    audio_array, sample_rate = audio, self.feature_extractor.sampling_rate
                else:
                    raise ValueError(
                        f"Unsupported audio data format: {type(audio)}")

                processed = self.preprocess_audio(audio_array, sample_rate)
                processed_audio.append(processed)

            batch[f"{column}_preprocessed"] = processed_audio

        return batch
