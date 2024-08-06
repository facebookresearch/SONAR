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

    This class extends the base PipelineConfig and can be used to add
    specific configuration parameters for preprocessing tasks. Currently,
    it doesn't add any additional fields, but it serves as a placeholder
    for future extensions.
    """
    pass


class PreprocessingPipeline(Pipeline, ABC):
    """
    Abstract base class for preprocessing pipelines.

    This class defines the structure for all preprocessing pipelines,
    ensuring they implement the necessary methods for text preprocessing
    and batch processing.

    Attributes:
        config (PreprocessingPipelineConfig): Configuration for the preprocessing pipeline.
    """

    def __init__(self, config: PreprocessingPipelineConfig):
        """
        Initialize the PreprocessingPipeline with the given configuration.

        Args:
            config (PreprocessingPipelineConfig): Configuration for the preprocessing pipeline.
        """
        super().__init__(config)

    @abstractmethod
    def preprocess_text(self, text: str) -> List[str]:
        """
        Abstract method to preprocess a single text.

        This method should be implemented by subclasses to define
        specific text preprocessing logic.

        Args:
            text (str): The input text to preprocess.

        Returns:
            List[str]: A list of preprocessed text segments.
        """
        pass

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a batch of data by applying preprocessing to specified columns.

        This method iterates over the specified columns in the configuration,
        applies the preprocessing to each text in those columns, and adds
        the results to new columns with a '_preprocessed' suffix.

        Args:
            batch (Dict[str, Any]): A dictionary containing the batch data.

        Returns:
            Dict[str, Any]: The input batch with additional preprocessed columns.
        """
        for column in self.config.columns:
            if column in batch:
                batch[f"{column}_preprocessed"] = [
                    self.preprocess_text(text) for text in batch[column]]
        return batch


@dataclass
class TextPreprocessingPipelineConfig(PreprocessingPipelineConfig):
    """
    Configuration class for text preprocessing pipelines.

    This class extends PreprocessingPipelineConfig with additional
    parameters specific to text preprocessing tasks.

    Attributes:
        handle_missing (str): Strategy for handling missing values. 
                              Options: 'skip', 'remove', or 'fill'.
        fill_value (Optional[str]): Value to use when filling missing data.
        source_lang (str): Source language code for the text data.
    """
    handle_missing: str = 'skip'
    fill_value: Optional[str] = None
    source_lang: str = "eng_Latn"


class TextPreprocessingPipeline(PreprocessingPipeline):
    """
    A pipeline for preprocessing text data, including handling of null and missing values,
    and performing sentence segmentation.

    This pipeline uses spaCy for text processing and can handle various
    languages based on the provided configuration.

    Attributes:
        config (TextPreprocessingPipelineConfig): Configuration for the text preprocessing pipeline.
        nlp: Loaded spaCy language model for text processing.
    """

    def __init__(self, config: TextPreprocessingPipelineConfig):
        """
        Initialize the TextPreprocessingPipeline with the given configuration.

        Args:
            config (TextPreprocessingPipelineConfig): Configuration for the text preprocessing pipeline.
        """
        self.config = config
        self.nlp = self.load_spacy_model(self.config.source_lang)
        logger.info("Text preprocessing model initialized.")

    def load_spacy_model(self, lang_code: str):
        """
        Loads the appropriate spaCy model based on the language code.

        This method attempts to load the specified spaCy model. If the model
        is not found, it attempts to download and load it.

        Args:
            lang_code (str): The language code for the desired spaCy model.

        Returns:
            The loaded spaCy language model.

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
        Preprocesses a single text by segmenting it into sentences.

        This method handles null or missing values according to the configuration
        and performs sentence segmentation on valid text input.

        Args:
            text (Optional[str]): The input text to preprocess.

        Returns:
            List[str]: A list of preprocessed sentences.

        Raises:
            ValueError: If an invalid handle_missing option is specified in the configuration.
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

    This class extends PipelineConfig with additional parameters
    specific to audio preprocessing tasks.

    Attributes:
        whisper_model (str): The name or path of the Whisper model to use.
        max_duration (Optional[float]): Maximum duration of audio in seconds.
                                        None for no limit.
    """
    whisper_model: str = "openai/whisper-base"
    max_duration: Optional[float] = None


class AudioPreprocessingPipeline(Pipeline):
    """
    A pipeline for preprocessing audio data using Whisper's feature extractor.

    This pipeline uses the Whisper model to extract features from audio data,
    preparing it for further processing or model input.

    Attributes:
        config (AudioPreprocessingPipelineConfig): Configuration for the audio preprocessing pipeline.
        whisper_processor (WhisperProcessor): Whisper processor for audio processing.
        feature_extractor (WhisperFeatureExtractor): Whisper feature extractor for audio feature extraction.
    """

    def __init__(self, config: AudioPreprocessingPipelineConfig):
        """
        Initialize the AudioPreprocessingPipeline with the given configuration.

        Args:
            config (AudioPreprocessingPipelineConfig): Configuration for the audio preprocessing pipeline.
        """
        self.config = config
        self.whisper_processor = WhisperProcessor.from_pretrained(
            self.config.whisper_model)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config.whisper_model)
        logger.info("Whisper processor and feature extractor initialized.")

    def preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocesses a single audio array using Whisper's feature extractor.

        This method converts the audio to float32, limits its duration if specified,
        and extracts features using the Whisper feature extractor.

        Args:
            audio (np.ndarray): The input audio array.
            sample_rate (int): The sample rate of the input audio.

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
        Processes a batch of audio data.

        This method iterates over the specified columns in the configuration,
        applies the audio preprocessing to each audio sample in those columns,
        and adds the results to new columns with a '_preprocessed' suffix.

        Args:
            batch (Dict[str, Any]): A dictionary containing the batch data.

        Returns:
            Dict[str, Any]: The input batch with additional preprocessed columns.

        Raises:
            ValueError: If an unsupported audio data format is encountered.
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

