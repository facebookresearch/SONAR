from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
import logging
from typing import List, Dict, Any
from .pipeline import Pipeline, PipelineOverwrites, PipelineConfig
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
import torch
from dataclasses import dataclass, replace
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingToTextOverwrites(PipelineOverwrites, total=False):
    """
    Overwrite options for EmbeddingToTextPipeline configuration.

    Attributes:
        decoder_model (str): The name or path of the decoder model to use.
        target_lang (str): The target language for decoding.
    """
    decoder_model: str
    target_lang: str


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
    """

    def with_overwrites(self, overwrites: PipelineOverwrites) -> 'TextToEmbeddingPipelineConfig':
        """
        Create a new configuration with the specified overwrites.

        Args:
            overwrites (PipelineOverwrites): Overwrite values for the configuration.

        Returns:
            TextToEmbeddingPipelineConfig: A new configuration instance with applied overwrites.
        """
        return replace(self, **overwrites)


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
    """
    decoder_model: str = "text_sonar_basic_decoder"
    target_lang: str = "eng_Latn"

    def with_overwrites(self, overwrites: EmbeddingToTextOverwrites) -> 'EmbeddingToTextPipelineConfig':
        """
        Create a new configuration with the specified overwrites.

        Args:
            overwrites (EmbeddingToTextOverwrites): Overwrite values for the configuration.

        Returns:
            EmbeddingToTextPipelineConfig: A new configuration instance with applied overwrites.
        """
        return replace(self, **overwrites)


@dataclass
class HFEmbeddingToTextPipeline(Pipeline):
    """
    Pipeline for converting embeddings back to text using a Hugging Face model.

    Attributes:
        config (EmbeddingToTextPipelineConfig): Configuration for the pipeline.
        t2t_model (EmbeddingToTextModelPipeline): The model used for decoding embeddings to text.
    """
    config: EmbeddingToTextPipelineConfig

    def __post_init__(self):
        """
        Initialize the embedding-to-text model after the instance is created.
        """
        logger.info("Initializing embedding to text model...")
        self.t2t_model = EmbeddingToTextModelPipeline(
            decoder=self.config.decoder_model,
            tokenizer=self.config.decoder_model,
            device=self.config.device
        )
        logger.info("Model initialized.")

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Process a batch of embeddings and convert them back to text.

        Args:
            batch (Dict[str, Any]): A dictionary containing the batch data with embeddings.

        Returns:
            Dict[str, List[str]]: The input batch with additional columns containing the decoded texts.

        Raises:
            AssertionError: If the input columns don't contain lists of embeddings.
        """
        for column in self.config.columns:
            embeddings = batch[column]
            assert all(isinstance(item, list) for item in embeddings), \
                f"Column {column} must contain only lists of embeddings, not individual embeddings."

            all_embeddings = [embed for item in embeddings for embed in item]
            all_decoded_texts = self.decode_embeddings(all_embeddings)

            reconstructed_texts = []
            start_idx = 0
            for item in embeddings:
                end_idx = start_idx + len(item)
                reconstructed_texts.append(
                    all_decoded_texts[start_idx:end_idx])
                start_idx = end_idx

            batch[f"{column}_{self.config.output_column_suffix}"] = reconstructed_texts
            logger.info(
                f" {column} column reconstructed:  {batch[column][:5]}")

        return batch

    def decode_embeddings(self, embeddings: List[Any]) -> List[str]:
        """
        Decode a list of embeddings into text.

        Args:
            embeddings (List[Any]): A list of embeddings to decode.

        Returns:
            List[str]: A list of decoded texts corresponding to the input embeddings.

        Raises:
            Exception: If there's an error during the decoding process.
        """
        try:
            logger.info(f"Decoding {len(embeddings)} embeddings...")
            embeddings = [torch.tensor(embed) if not isinstance(
                embed, torch.Tensor) else embed for embed in embeddings]
            embeddings = [embed.unsqueeze(0) if embed.dim(
            ) == 1 else embed for embed in embeddings]
            batched_embeddings = torch.cat(embeddings, dim=0)

            decoded_texts = []
            for i in range(0, len(embeddings), self.config.batch_size):
                batch_embeddings = batched_embeddings[i:i +
                                                      self.config.batch_size]
                batch_decoded = self.t2t_model.predict(
                    batch_embeddings,
                    target_lang=self.config.target_lang,
                    batch_size=self.config.batch_size
                )
                decoded_texts.extend(batch_decoded)

            logger.info("Texts decoded successfully.")
            return decoded_texts
        except Exception as e:
            logger.error(f"Error decoding texts: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise


@dataclass
class HFTextToEmbeddingPipeline(Pipeline):
    """
    Pipeline for converting text to embeddings using a Hugging Face model.

    Attributes:
        config (TextToEmbeddingPipelineConfig): Configuration for the pipeline.
        t2vec_model (TextToEmbeddingModelPipeline): The model used for encoding text to embeddings.
    """
    config: TextToEmbeddingPipelineConfig

    def __post_init__(self):
        """
        Initialize the text-to-embedding model after the instance is created.
        """
        logger.info("Initializing text to embedding model...")
        self.t2vec_model = TextToEmbeddingModelPipeline(
            encoder=self.config.encoder_model,
            tokenizer=self.config.encoder_model,
            device=self.config.device
        )

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of texts and convert them to embeddings.

        Args:
            batch (Dict[str, Any]): A dictionary containing the batch data with text columns.

        Returns:
            Dict[str, Any]: The input batch with additional columns containing the computed embeddings.

        Raises:
            AssertionError: If the input columns don't contain lists of sentences.
        """
        for column in self.config.columns:
            if column in batch:
                assert all(isinstance(item, list) for item in batch[column]), \
                    f"Column {column} must contain only lists of sentences, not individual strings."

                all_sentences = [sentence for item in batch[column]
                                 for sentence in item]
                all_embeddings = self.encode_texts(all_sentences)

                sentence_embeddings = []
                start_idx = 0
                for item in batch[column]:
                    end_idx = start_idx + len(item)
                    sentence_embeddings.append(
                        all_embeddings[start_idx:end_idx])
                    start_idx = end_idx

                batch[f"{column}_{self.config.output_column_suffix}"] = sentence_embeddings
                logger.info(
                    f" {column} column embeddings:  {batch[column][:5]}")
            else:
                logger.warning(f"Column {column} not found in batch.")

        return batch

    def encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode a list of texts into embeddings.

        Args:
            texts (List[str]): A list of texts to encode.

        Returns:
            List[np.ndarray]: A list of embeddings corresponding to the input texts.

        Raises:
            Exception: If there's an error during the encoding process.
        """
        try:
            embeddings = []
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                batch_embeddings = self.t2vec_model.predict(
                    batch_texts, source_lang=self.config.source_lang, batch_size=self.config.batch_size)
                embeddings.extend(batch_embeddings)
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts or sentences: {e}")
            raise

