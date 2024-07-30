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

    This class can be extended to include additional preprocessing-specific
    configuration parameters as needed.
    """
    pass


class PreprocessingPipeline(Pipeline, ABC):
    """
    Abstract base class for preprocessing pipelines.

    This class defines the structure for preprocessing pipelines and includes
    an abstract method for text preprocessing.
    """

    def __init__(self, config: PreprocessingPipelineConfig):
        """
        Initialize the preprocessing pipeline with the given configuration.

        Args:
            config (PreprocessingPipelineConfig): Configuration for the pipeline.
        """
        super().__init__(config)

    @abstractmethod
    def preprocess_text(self, text: str) -> List[str]:
        """
        Abstract method to preprocess a single text.

        Args:
            text (str): The input text to preprocess.

        Returns:
            List[str]: A list of preprocessed text segments.
        """
        pass

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of data by applying preprocessing to specified columns.

        Args:
            batch (Dict[str, Any]): A dictionary containing the batch data.

        Returns:
            Dict[str, Any]: The processed batch with new columns added for preprocessed data.
        """
        for column in self.config.columns:
            if column in batch:
                batch[f"{column}_{self.output_column_suffix}"] = [
                    self.preprocess_text(text) for text in batch[column]]
            else:
                logger.warning(f"Column {column} not found in batch.")
        return batch


@dataclass
class TextPreprocessingPipelineConfig(PreprocessingPipelineConfig):
    """
    Configuration class for text preprocessing pipelines.

    Attributes:
        handle_missing (str): Strategy for handling missing values. Options: 'skip', 'remove', or 'fill'.
        fill_value (Optional[str]): Value to use when filling missing data if handle_missing is 'fill'.
        source_lang (str): Source language code for the text data.
    """
    handle_missing: str = 'skip'
    fill_value: Optional[str] = None
    source_lang: str = "eng_Latn"


@dataclass
class TextPreprocessingPipeline(PreprocessingPipeline):
    """
    A pipeline for preprocessing text data, including handling of null and missing values,
    and performing sentence segmentation using spaCy.

    This pipeline loads the appropriate spaCy model based on the source language
    and applies it to segment the input text into sentences.
    """

    config: TextPreprocessingPipelineConfig

    def __post_init__(self):
        """
        Initialize the spaCy model after the instance is created.
        """
        self.nlp = self.load_spacy_model(self.config.source_lang)
        logger.info("SpaCy model initialized.")

    def load_spacy_model(self, lang_code: str):
        """
        Load the appropriate spaCy model based on the language code.

        Args:
            lang_code (str): The language code for the desired model.

        Returns:
            spacy.language.Language: The loaded spaCy model.

        Raises:
            ValueError: If the language code is not supported.
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
        Preprocess a single text by segmenting it into sentences.
        Handles null or missing values according to the configuration.

        Args:
            text (Optional[str]): The input text to preprocess.

        Returns:
            List[str]: A list of preprocessed sentences.

        Raises:
            ValueError: If an invalid handle_missing option is specified.
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


@dataclass
class AudioPreprocessingPipelineConfig(PipelineConfig):
    """
    Configuration class for audio preprocessing pipelines.

    Attributes:
        whisper_model (str): The name or path of the Whisper model to use.
        max_duration (Optional[float]): Maximum duration of audio in seconds. None for no limit.
    """
    whisper_model: str = "openai/whisper-base"
    max_duration: Optional[float] = None


@dataclass
class AudioPreprocessingPipeline(Pipeline):
    """
    A pipeline for preprocessing audio data using Whisper's feature extractor.

    This pipeline initializes the Whisper processor and feature extractor,
    and applies them to preprocess audio data.
    """

    config: AudioPreprocessingPipelineConfig

    def __post_init__(self):
        """
        Initialize the Whisper processor and feature extractor after the instance is created.
        """
        self.whisper_processor = WhisperProcessor.from_pretrained(
            self.config.whisper_model)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config.whisper_model)
        logger.info("Whisper processor and feature extractor initialized.")

    def preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess a single audio array using Whisper's feature extractor.

        Args:
            audio (np.ndarray): The input audio array.
            sample_rate (int): The sample rate of the audio.

        Returns:
            np.ndarray: The preprocessed audio features.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if self.config.max_duration is not None:
            max_samples = int(self.config.max_duration * sample_rate)
            audio = audio[:max_samples]

        inputs = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="np"
        )
        return inputs.input_features.squeeze()

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of audio data.

        Args:
            batch (Dict[str, Any]): A dictionary containing the batch data.

        Returns:
            Dict[str, Any]: The processed batch with new columns added for preprocessed audio.

        Raises:
            ValueError: If the audio data format is unsupported.
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

