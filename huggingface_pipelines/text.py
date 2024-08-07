from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
import logging
from typing import List, Dict, Any
from .pipeline import Pipeline, PipelineOverwrites, PipelineConfig
import torch
from dataclasses import dataclass, replace
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextToEmbeddingOverwrites(PipelineOverwrites, total=False):

    """
        Overwrite options for TexToEmbeddingPipeline configuration.


        Attributes:
            max_seq_len (int): Number of a tokens per sequence. Defaults to None which means maximum for model.
    """

    max_seq_len: int
    encoder_model: str
    source_lang: str


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

    max_seq_len: int = None
    encoder_model: str = "text_sonar_basic_encoder"
    source_lang: str = "eng_Latn"

    def with_overwrites(self, overwrites: EmbeddingToTextOverwrites) -> 'TextToEmbeddingPipelineConfig':
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFEmbeddingToTextPipeline(Pipeline):
    def __init__(self, config: EmbeddingToTextPipelineConfig):
        logger.info("Initializing embedding to text model...")
        self.config = config
        self.t2t_model = EmbeddingToTextModelPipeline(
            decoder=self.config.decoder_model,
            tokenizer=self.config.decoder_model,
            device=self.config.device
        )
        logger.info("Model initialized.")

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, List[str]]:
        for column in self.config.columns:
            embeddings = batch[column]
            assert all(isinstance(item, list) for item in embeddings), \
                f"Column {column} must contain only lists of embeddings, not individual embeddings."
            all_embeddings = np.vstack([np.array(embed)
                                       for item in embeddings for embed in item])
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

    def decode_embeddings(self, embeddings: np.ndarray) -> List[str]:
        try:
            logger.info(f"Decoding {len(embeddings)} embeddings...")
            embeddings_tensor = torch.from_numpy(embeddings).float()

            decoded_texts = []
            for i in range(0, len(embeddings), self.config.batch_size):
                batch_embeddings = embeddings_tensor[i:i +
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


class HFTextToEmbeddingPipeline(Pipeline):
    def __init__(self, config: TextToEmbeddingPipelineConfig):
        logger.info("Initializing text to embedding model...")
        self.config = config
        self.t2vec_model = TextToEmbeddingModelPipeline(
            encoder=self.config.encoder_model,
            tokenizer=self.config.encoder_model,
            device=self.config.device
        )

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
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

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        try:
            embeddings = []
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                batch_embeddings = self.t2vec_model.predict(
                    batch_texts, source_lang=self.config.source_lang, batch_size=self.config.batch_size, max_seq_len=self.config.max_seq_len)

                batch_embeddings = [emb.cpu().numpy()
                                    for emb in batch_embeddings]
                embeddings.extend(batch_embeddings)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error encoding texts or sentences: {e}")
            raise

