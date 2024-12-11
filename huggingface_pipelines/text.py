import itertools
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import spacy  # type: ignore
import torch
from numpy.typing import DTypeLike
from spacy.language import Language  # type: ignore

from sonar.inference_pipelines.text import (
    EmbeddingToTextModelPipeline,
    TextToEmbeddingModelPipeline,
)

from .dataset import DatasetConfig  # type: ignore
from .pipeline import Pipeline, PipelineConfig, PipelineFactory  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextDatasetConfig(DatasetConfig):
    """
    Configuration for text datasets.

    This class inherits from BaseDatasetConfig and can be used for
    text-specific dataset configurations.
    """

    pass


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
            handle_missing='fill',
        )
    """

    fill_value: Optional[str] = None
    source_lang: str = "eng_Latn"
    handle_missing: Literal["skip", "remove", "fill"] = "skip"


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
        config = TextSegmentationPipelineConfig(
            columns=['text'], output_path='./output')
        pipeline = TextSegmentationPipeline(config)
        result = pipeline(
            {'text': ['This is a sample text.', 'Another example.']})
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

    config: TextSegmentationPipelineConfig

    def __init__(self, config: TextSegmentationPipelineConfig):
        """
        Initialize the TextSegmentationPipeline with the given configuration.

        Args:
            config (TextSegmentationPipelineConfig): Configuration for the text segmentation pipeline.
        """
        super().__init__(config)
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
            raise ValueError(f"No installed model found for language code: {lang_code}")
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
            sentences = pipeline.segment_text(
                "This is a sample. It has two sentences.")
            print(sentences)  # ['This is a sample.', 'It has two sentences.']
        """
        if text is None or (isinstance(text, str) and text.strip() == ""):
            if self.config.handle_missing == "skip":
                return []
            elif self.config.handle_missing == "remove":
                return []
            elif self.config.handle_missing == "fill":
                return [self.config.fill_value] if self.config.fill_value else []
            else:
                raise ValueError(
                    f"Invalid handle_missing option: {self.config.handle_missing}"
                )

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
            # {'text': ['Sample text.', 'Another example.'], 'text_preprocessed': [['Sample text.'], ['Another example.']]}
            print(result)
        """
        for column in self.config.columns:
            if column in batch:
                batch[f"{column}_{self.config.output_column_suffix}"] = [
                    self.segment_text(text) for text in batch[column]
                ]

            else:
                raise ValueError(f"Column {column} not found in batch.")
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
class TextToEmbeddingPipelineConfig(PipelineConfig):
    """
    Configuration class for text-to-embedding pipelines.

    Attributes:
        encoder_model (str): The name or path of the model to be used for encoding texts into embeddings.
        source_lang (str): The source language code for the texts to be encoded.
        columns (List[str]): List of column names in the input data to process.
        output_column_suffix (str): Suffix to append to the output column names.
        batch_size (int): Number of items to process in each batch.
        device (str): The device to use for computation (e.g., 'cpu' or 'cuda').
        max_seq_len (int): Maximum sequence length for input texts.
        dtype (np.dtype): The data type of the output numpy embeddings.

    Example:
        config = TextToEmbeddingPipelineConfig(
            encoder_model="text_sonar_basic_encoder",
            source_lang="eng_Latn",
            columns=["text"],
            output_column_suffix="embedding",
            batch_size=32,
            device="cuda",
            max_seq_len=512,
            dtype = np.float32

        )
    """

    max_seq_len: int = 512
    encoder_model: str = "text_sonar_basic_encoder"
    source_lang: str = "eng_Latn"
    dtype: DTypeLike = np.float32


@dataclass
class EmbeddingToTextPipelineConfig(PipelineConfig):
    """
    Configuration class for embedding-to-text pipelines.

    Attributes:
        decoder_model (str): The name or path of the model to be used for decoding embeddings back into texts.
        target_lang (str): The target language code for the texts to be decoded.
        columns (List[str]): List of column names in the input data to process.
        output_column_suffix (str): Suffix to append to the output column names.
        batch_size (int): Number of items to process in each batch.
        device (str): The device to use for computation (e.g., 'cpu' or 'cuda').
        dtype (np.dtype): The data type of the output numpy embeddings.

    Example:
        config = EmbeddingToTextPipelineConfig(
            decoder_model="text_sonar_basic_decoder",
            target_lang="eng_Latn",
            columns=["embedding"],
            output_column_suffix="text",
            batch_size=32,
            device="cuda",
            dtype = np.float16
        )
    """

    decoder_model: str = "text_sonar_basic_decoder"
    target_lang: str = "eng_Latn"
    dtype: DTypeLike = np.float32


class HFEmbeddingToTextPipeline(Pipeline):
    """
    Pipeline for converting embeddings back to text using a Hugging Face model.

    This pipeline takes embeddings as input and decodes them into text using a specified decoder model.

    Attributes:
        config (EmbeddingToTextPipelineConfig): Configuration for the pipeline.
        t2t_model (EmbeddingToTextModelPipeline): The model used for decoding embeddings to text.
    """

    config: EmbeddingToTextPipelineConfig

    def __init__(self, config: EmbeddingToTextPipelineConfig):
        """
        Initialize the embedding-to-text pipeline.

        Args:
            config (EmbeddingToTextPipelineConfig): Configuration for the pipeline.
        """
        super().__init__(config)
        logger.info("Initializing embedding to text model...")
        self.t2t_model = EmbeddingToTextModelPipeline(
            decoder=self.config.decoder_model,
            tokenizer=self.config.decoder_model,
            device=torch.device(self.config.device),
        )
        logger.info("Model initialized.")

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of embeddings and convert them to text.
        Handles both list of individual embeddings or list of lists that contain embeddings

        Args:
            batch (Dict[str, Any]): Input batch containing texts.

        Returns:
            Dict[str, Any]: Processed batch with encoded embeddings.
        """

        for column in self.config.columns:
            if column in batch:
                embeddings = batch[column]

                if not isinstance(batch[column], (list, np.ndarray)):
                    raise ValueError(
                        f"Expected list for column {column}, got {type(batch[column])}"
                    )

                if len(batch[column]) == 0:
                    raise ValueError("Empty columns are not allowed.")

                embeddings = batch[column]
                if all(
                    isinstance(item, (np.ndarray, list))
                    and not isinstance(item[0], (list, np.ndarray))
                    for item in embeddings
                ):
                    all_embeddings = np.asarray(embeddings, dtype=self.config.dtype)
                    all_decoded_texts = self.decode_embeddings(all_embeddings)
                    batch[f"{column}_{self.config.output_column_suffix}"] = (
                        all_decoded_texts
                    )
                elif all(isinstance(item, list) for item in embeddings):
                    all_embeddings = np.vstack(
                        [
                            np.asarray(embed, dtype=self.config.dtype)
                            for item in embeddings
                            for embed in item
                        ]
                    )
                    all_decoded_texts = self.decode_embeddings(all_embeddings)
                    lengths = [len(item) for item in embeddings]
                    indices = list(itertools.accumulate(lengths))
                    reconstructed_texts = [
                        all_decoded_texts[start:end]
                        for start, end in zip([0] + indices[:-1], indices)
                    ]
                    batch[f"{column}_{self.config.output_column_suffix}"] = (
                        reconstructed_texts
                    )
                else:
                    raise ValueError(f"Invalid input type for column {column}")
                    logger.debug(
                        f"{column} column reconstructed: {batch[f'{column}_{self.config.output_column_suffix}'][:5]}"
                    )
            else:
                logger.error(f"Column {column} not found in batch.")
                raise ValueError(f"Column {column} not found in batch.")

        return batch

    def decode_embeddings(self, embeddings: np.ndarray) -> List[str]:
        """
        Decode a batch of embeddings into text.

        Args:
            embeddings (np.ndarray): Array of embeddings to decode.

        Returns:
            List[str]: List of decoded texts.

        Raises:
            Exception: If there's an error during decoding.
        """
        try:
            logger.info(f"Decoding {len(embeddings)} embeddings...")
            decoded_texts = []

            for i in range(0, len(embeddings), self.config.batch_size):
                batch_embeddings = embeddings[i : i + self.config.batch_size]
                batch_embeddings_tensor = (
                    torch.from_numpy(batch_embeddings).float().to(self.config.device)
                )

                batch_decoded = self.t2t_model.predict(
                    batch_embeddings_tensor,
                    target_lang=self.config.target_lang,
                    batch_size=self.config.batch_size,
                )

                decoded_texts.extend(batch_decoded)

            logger.info("Texts decoded successfully.")
            return decoded_texts
        except Exception as e:
            logger.error(f"Error decoding texts: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise


class EmbeddingToTextPipelineFactory:
    """
    Factory class for creating EmbeddingToText pipelines.

    This factory creates HFEmbeddingToTextPipeline instances based on the provided configuration.

    Example:
        factory = EmbeddingToTextPipelineFactory()
        config = {
            "decoder_model": "text_sonar_basic_decoder",
            "columns": ["embedding"],
            "output_column_suffix": "text",
            "batch_size": 32,
            "device": "cuda"
        }
        pipeline = factory.create_pipeline(config)
    """

    def create_pipeline(self, config: Dict[str, Any]) -> Pipeline:
        """
        Create an EmbeddingToText pipeline based on the provided configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the pipeline.

        Returns:
            Pipeline: An instance of HFEmbeddingToTextPipeline.
        """
        pipeline_config = EmbeddingToTextPipelineConfig(**config)
        return HFEmbeddingToTextPipeline(pipeline_config)


class HFTextToEmbeddingPipeline(Pipeline):
    """
    Pipeline for converting text to embeddings using a Hugging Face model.

    This pipeline takes text as input and encodes it into embeddings using a specified encoder model.

    Attributes:
        config (TextToEmbeddingPipelineConfig): Configuration for the pipeline.
        t2vec_model (TextToEmbeddingModelPipeline): The model used for encoding text to embeddings.
    """

    config: TextToEmbeddingPipelineConfig

    def __init__(self, config: TextToEmbeddingPipelineConfig):
        """
        Initialize the text-to-embedding pipeline.

        Args:
            config (TextToEmbeddingPipelineConfig): Configuration for the pipeline.
        """
        super().__init__(config)
        logger.info("Initializing text to embedding model...")
        self.t2vec_model = TextToEmbeddingModelPipeline(
            encoder=self.config.encoder_model,
            tokenizer=self.config.encoder_model,
            device=torch.device(self.config.device),
        )

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of texts and convert them to embeddings.
        Handles both individual strings and lists of sentences.

        Args:
            batch (Dict[str, Any]): Input batch containing texts.

        Returns:
            Dict[str, Any]: Processed batch with encoded embeddings.
        """
        for column in self.config.columns:
            if column in batch:
                if not isinstance(batch[column], (list, np.ndarray)):
                    raise ValueError(
                        f"Expected list for column {column}, got {type(batch[column])}"
                    )

                if len(batch[column]) == 0:
                    raise ValueError("Empty columns are not allowed.")

                # Check if the input is a list of strings or a list of lists
                if all(isinstance(item, str) for item in batch[column]):
                    # Case: List of individual strings
                    all_texts = batch[column]
                    all_embeddings = self.encode_texts(all_texts)
                    batch[f"{column}_{self.config.output_column_suffix}"] = (
                        all_embeddings
                    )
                elif all(isinstance(item, list) for item in batch[column]):
                    # Case: List of lists (sentences)
                    all_sentences = [
                        sentence for item in batch[column] for sentence in item
                    ]
                    all_embeddings = self.encode_texts(all_sentences)

                    # Calculate the cumulative sum of lengths
                    lengths = [
                        len(item) if isinstance(item, list) else 1
                        for item in batch[column]
                    ]
                    indices = list(itertools.accumulate(lengths))

                    # Use the indices to slice all_embeddings
                    sentence_embeddings = [
                        all_embeddings[start:end]
                        for start, end in zip([0] + indices[:-1], indices)
                    ]

                    batch[f"{column}_{self.config.output_column_suffix}"] = (
                        sentence_embeddings
                    )

                else:
                    raise ValueError(
                        f"Invalid input type for column {column} {type(batch[column])}"
                    )

                logger.debug(
                    f"{column} column embeddings: {batch[f'{column}_{self.config.output_column_suffix}'][:5]}"
                )
            else:
                logger.error(f"Column {column} not found in batch.")
                raise ValueError(f"Column {column} not found in batch.")

        return batch

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts (List[str]): List of texts to encode.

        Returns:
            np.ndarray: Array of embeddings.

        Raises:
            Exception: If there's an error during encoding.
        """
        try:
            embeddings: List[np.ndarray] = []
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i : i + self.config.batch_size]
                batch_embeddings = self.t2vec_model.predict(
                    batch_texts,
                    source_lang=self.config.source_lang,
                    batch_size=self.config.batch_size,
                    max_seq_len=self.config.max_seq_len,
                )
                batch_embeddings = (
                    batch_embeddings.detach().cpu().numpy().astype(self.config.dtype)
                )
                embeddings.extend(batch_embeddings)

            return np.vstack(embeddings)
        except Exception as e:
            logger.error(f"Error encoding texts or sentences: {e}")
            raise


class TextToEmbeddingPipelineFactory(PipelineFactory):
    """
    Factory class for creating TextToEmbedding pipelines.

    This factory creates HFTextToEmbeddingPipeline instances based on the provided configuration.

    Example:
        factory = TextToEmbeddingPipelineFactory()
        config = {
            "encoder_model": "text_sonar_basic_encoder",
            "columns": ["text"],
            "output_column_suffix": "embedding",
            "batch_size": 32,
            "device": "cuda",
            "dtype": "torch.float32"
        }
        pipeline = factory.create_pipeline(config)
    """

    def create_pipeline(self, config: Dict[str, Any]) -> Pipeline:
        """
        Create a TextToEmbedding pipeline based on the provided configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the pipeline.

        Returns:
            Pipeline: An instance of HFTextToEmbeddingPipeline.
        """
        pipeline_config = TextToEmbeddingPipelineConfig(**config)
        return HFTextToEmbeddingPipeline(pipeline_config)
