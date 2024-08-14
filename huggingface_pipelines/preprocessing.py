from transformers import WhisperProcessor, WhisperFeatureExtractor
import numpy as np
import spacy
from typing import List, Dict, Any, Optional, Literal
from .pipeline import Pipeline, PipelineConfig, PipelineFactory
import logging
from dataclasses import dataclass
from spacy.language import Language

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextSegmentationPipelineConfig(PipelineConfig):
    """
    Configuration class for text preprocessing pipelines.
    This class extends PreprocessingPipelineConfig with additional
    parameters specific to text preprocessing tasks.

    Attributes:
        fill_value (Optional[str]): Value to use when filling missing data.
        source_lang (str): Source language code for the text data.
        handle_missing (Literal['skip', 'remove', 'fill']): Strategy for handling missing values.

    Example:
        config = TextSegmentationPipelineConfig(
            columns=['text'],
            output_path='./output',
            fill_value='N/A',
            source_lang='eng_Latn',
            handle_missing='fill'
        )
    """
    fill_value: Optional[str] = None
    source_lang: str = "eng_Latn"
    handle_missing: Literal['skip', 'remove', 'fill'] = "skip"


class TextSegmentationPipeline(Pipeline):
    """
    A pipeline for segmenting text data into sentences, including handling of null and missing values.
    This pipeline uses spaCy for text processing and can handle various
    languages based on the provided configuration.

    Attributes:
        config (TextSegmentationPipelineConfig): Configuration for the text segmentation pipeline.
        SPACY_MODELS (Dict[str, str]): Dictionary mapping language codes to installed spaCy models.
        nlp (Language): Loaded spaCy language model for text processing.

    Example:
        config = TextSegmentationPipelineConfig(columns=['text'], output_path='./output')
        pipeline = TextSegmentationPipeline(config)
        result = pipeline({'text': ['This is a sample text.', 'Another example.']})
    """

    SPACY_MODELS = {
        "eng_Latn": "en_core_web_sm",
        "fra_Latn": "fr_core_news_sm",
        "deu_Latn": "de_core_news_sm",
        "spa_Latn": "es_core_news_sm",
        "ita_Latn": "it_core_news_sm",
        "por_Latn": "pt_core_news_sm",
        "nld_Latn": "nl_core_news_sm",
    }

    def __init__(self, config: TextSegmentationPipelineConfig):
        """
        Initialize the TextSegmentationPipeline with the given configuration.

        Args:
            config (TextSegmentationPipelineConfig): Configuration for the text segmentation pipeline.
        """
        super().__init__(config)
        self.config = config
        self.nlp = self.load_spacy_model(self.config.source_lang)
        logger.info("Text preprocessing model initialized.")

    def load_spacy_model(self, lang_code: str) -> Language:
        """
        Loads the appropriate spaCy model based on the language code.

        Args:
            lang_code (str): The language code for the desired spaCy model.

        Returns:
            Language: The loaded spaCy language model.

        Raises:
            ValueError: If the language code is not supported.

        Example:
            nlp = pipeline.load_spacy_model('en')
        """
        if lang_code not in self.SPACY_MODELS:
            raise ValueError(
                f"No installed model found for language code: {lang_code}")
        return spacy.load(self.SPACY_MODELS[lang_code])

    def segment_text(self, text: Optional[str]) -> List[str]:
        """
        Preprocesses a single text by segmenting it into sentences.

        Args:
            text (Optional[str]): The input text to preprocess.

        Returns:
            List[str]: A list of preprocessed sentences.

        Raises:
            ValueError: If an invalid handle_missing option is specified in the configuration.

        Example:
            sentences = pipeline.segment_text("This is a sample. It has two sentences.")
            print(sentences)  # ['This is a sample.', 'It has two sentences.']
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

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a batch of data by applying text segmentation to specified text columns.

        Args:
            batch (Dict[str, Any]): A dictionary containing the batch data.

        Returns:
            Dict[str, Any]: The input batch with additional preprocessed columns.

        Example:
            batch = {'text': ['Sample text.', 'Another example.']}
            result = pipeline.process_batch(batch)
            print(result)  # {'text': ['Sample text.', 'Another example.'], 'text_preprocessed': [['Sample text.'], ['Another example.']]}
        """
        for column in self.config.columns:
            if column in batch:
                batch[f"{column}_preprocessed"] = [
                    self.segment_text(text) for text in batch[column]]
        return batch


class TextSegmentationPipelineFactory(PipelineFactory):
    """
    Factory class for creating TextSegmentationPipeline instances.

    Example:
        factory = TextSegmentationPipelineFactory()
        config = {'columns': ['text'], 'output_path': './output'}
        pipeline = factory.create_pipeline(config)
    """

    def create_pipeline(self, config: Dict[str, Any]) -> Pipeline:
        """
        Create a TextSegmentationPipeline instance with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the pipeline.

        Returns:
            Pipeline: An instance of TextSegmentationPipeline.
        """
        pipeline_config = TextSegmentationPipelineConfig(**config)
        return TextSegmentationPipeline(pipeline_config)


@dataclass
class AudioPreprocessingPipelineConfig(PipelineConfig):
    """
    Configuration class for audio preprocessing pipelines.

    Attributes:
        whisper_model (str): The name or path of the Whisper model to use.
        max_duration (Optional[float]): Maximum duration of audio in seconds. None for no limit.

    Example:
        config = AudioPreprocessingPipelineConfig(
            columns=['audio'],
            output_path='./output',
            whisper_model='openai/whisper-base',
            max_duration=30.0
        )
    """
    whisper_model: str = "openai/whisper-base"
    max_duration: Optional[float] = None


class AudioPreprocessingPipeline(Pipeline):
    """
    A pipeline for preprocessing audio data using Whisper's feature extractor.

    Attributes:
        config (AudioPreprocessingPipelineConfig): Configuration for the audio preprocessing pipeline.
        whisper_processor (WhisperProcessor): Whisper processor for audio processing.
        feature_extractor (WhisperFeatureExtractor): Whisper feature extractor for audio feature extraction.

    Example:
        config = AudioPreprocessingPipelineConfig(columns=['audio'], output_path='./output')
        pipeline = AudioPreprocessingPipeline(config)
        result = pipeline({'audio': [np.array([...]), np.array([...])]})
    """

    def __init__(self, config: AudioPreprocessingPipelineConfig):
        """
        Initialize the AudioPreprocessingPipeline with the given configuration.

        Args:
            config (AudioPreprocessingPipelineConfig): Configuration for the audio preprocessing pipeline.
        """
        super().__init__(config)
        self.config = config
        self.whisper_processor = WhisperProcessor.from_pretrained(
            self.config.whisper_model)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config.whisper_model)
        logger.info("Whisper processor and feature extractor initialized.")

    def preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocesses a single audio array using Whisper's feature extractor.

        Args:
            audio (np.ndarray): The input audio array.
            sample_rate (int): The sample rate of the input audio.

        Returns:
            np.ndarray: The preprocessed audio features.

        Example:
            audio = np.array([...])  # Your audio data
            sample_rate = 16000
            features = pipeline.preprocess_audio(audio, sample_rate)
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

        Args:
            batch (Dict[str, Any]): A dictionary containing the batch data.

        Returns:
            Dict[str, Any]: The input batch with additional preprocessed columns.

        Raises:
            ValueError: If an unsupported audio data format is encountered.

        Example:
            batch = {'audio': [{'array': np.array([...]), 'sampling_rate': 16000}, {'array': np.array([...]), 'sampling_rate': 16000}]}
            result = pipeline.process_batch(batch)
            print(result.keys())  # dict_keys(['audio', 'audio_preprocessed'])
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


class AudioPreprocessingPipelineFactory(PipelineFactory):
    """
    Factory class for creating AudioPreprocessingPipeline instances.

    Example:
        factory = AudioPreprocessingPipelineFactory()
        config = {'columns': ['audio'], 'output_path': './output', 'whisper_model': 'openai/whisper-base'}
        pipeline = factory.create_pipeline(config)
    """

    def create_pipeline(self, config: Dict[str, Any]) -> Pipeline:
        """
        Create an AudioPreprocessingPipeline instance with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the pipeline.

        Returns:
            Pipeline: An instance of AudioPreprocessingPipeline.
        """
        pipeline_config = AudioPreprocessingPipelineConfig(**config)
        return AudioPreprocessingPipeline(pipeline_config)

