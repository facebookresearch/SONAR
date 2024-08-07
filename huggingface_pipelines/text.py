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
    Overwrite options for TextToEmbeddingPipeline configuration.

    Attributes:
        max_seq_len (int): Maximum number of tokens per sequence. If None, uses the model's maximum.
        encoder_model (str): Name or path of the model for encoding texts into embeddings.
        source_lang (str): Source language code for the texts to be encoded.
    """
    max_seq_len: int
    encoder_model: str
    source_lang: str


class EmbeddingToTextOverwrites(PipelineOverwrites, total=False):
    """
    Overwrite options for EmbeddingToTextPipeline configuration.

    Attributes:
        decoder_model (str): Name or path of the decoder model to use.
        target_lang (str): Target language code for decoding.
    """
    decoder_model: str
    target_lang: str


@dataclass
class TextToEmbeddingPipelineConfig(PipelineConfig):
    """
    Configuration for text-to-embedding pipelines.

    Attributes:
        max_seq_len (int): Maximum sequence length. Defaults to None (use model's max).
        encoder_model (str): Name or path of the encoding model.
        source_lang (str): Source language code.
        columns (List[str]): Input data column names to process.
        output_column_suffix (str): Suffix for output column names.
        batch_size (int): Number of items to process in each batch.
        device (str): Computation device ('cpu' or 'cuda').
    """
    max_seq_len: int = None
    encoder_model: str = "text_sonar_basic_encoder"
    source_lang: str = "eng_Latn"

    def with_overwrites(self, overwrites: TextToEmbeddingOverwrites) -> 'TextToEmbeddingPipelineConfig':
        """
        Create a new configuration with specified overwrites.

        Args:
            overwrites (TextToEmbeddingOverwrites): Overwrite values for the configuration.

        Returns:
            TextToEmbeddingPipelineConfig: New configuration instance with applied overwrites.
        """
        return replace(self, **overwrites)


@dataclass
class EmbeddingToTextPipelineConfig(PipelineConfig):
    """
    Configuration for embedding-to-text pipelines.

    Attributes:
        decoder_model (str): Name or path of the decoding model.
        target_lang (str): Target language code for decoding.
        columns (List[str]): Input data column names to process.
        output_column_suffix (str): Suffix for output column names.
        batch_size (int): Number of items to process in each batch.
        device (str): Computation device ('cpu' or 'cuda').
    """
    decoder_model: str = "text_sonar_basic_decoder"
    target_lang: str = "eng_Latn"

    def with_overwrites(self, overwrites: EmbeddingToTextOverwrites) -> 'EmbeddingToTextPipelineConfig':
        """
        Create a new configuration with specified overwrites.

        Args:
            overwrites (EmbeddingToTextOverwrites): Overwrite values for the configuration.

        Returns:
            EmbeddingToTextPipelineConfig: New configuration instance with applied overwrites.
        """
        return replace(self, **overwrites)


class HFEmbeddingToTextPipeline(Pipeline):
    """
    Pipeline for converting embeddings back to text using a Hugging Face model.
    """

    def __init__(self, config: EmbeddingToTextPipelineConfig):
        """
        Initialize the embedding-to-text pipeline.

        Args:
            config (EmbeddingToTextPipelineConfig): Configuration for the pipeline.
        """
        logger.info("Initializing embedding to text model...")
        self.config = config
        self.t2t_model = EmbeddingToTextModelPipeline(
            decoder=self.config.decoder_model,
            tokenizer=self.config.decoder_model,
            device=self.config.device
        )
        logger.info("Model initialized.")

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Process a batch of embeddings and convert them to text.

        Args:
            batch (Dict[str, Any]): Input batch containing embeddings.

        Returns:
            Dict[str, List[str]]: Processed batch with decoded texts.
        """
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
            logger.info(f"{column} column reconstructed: {batch[column][:5]}")

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
    """
    Pipeline for converting text to embeddings using a Hugging Face model.
    """

    def __init__(self, config: TextToEmbeddingPipelineConfig):
        """
        Initialize the text-to-embedding pipeline.

        Args:
            config (TextToEmbeddingPipelineConfig): Configuration for the pipeline.
        """
        logger.info("Initializing text to embedding model...")
        self.config = config
        self.t2vec_model = TextToEmbeddingModelPipeline(
            encoder=self.config.encoder_model,
            tokenizer=self.config.encoder_model,
            device=self.config.device
        )

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of texts and convert them to embeddings.

        Args:
            batch (Dict[str, Any]): Input batch containing texts.

        Returns:
            Dict[str, Any]: Processed batch with embeddings.
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
                logger.info(f"{column} column embeddings: {batch[column][:5]}")
            else:
                logger.warning(f"Column {column} not found in batch.")

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
            embeddings = []
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                batch_embeddings = self.t2vec_model.predict(
                    batch_texts,
                    source_lang=self.config.source_lang,
                    batch_size=self.config.batch_size,
                    max_seq_len=self.config.max_seq_len
                )
                batch_embeddings = [emb.cpu().numpy()
                                    for emb in batch_embeddings]
                embeddings.extend(batch_embeddings)

            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error encoding texts or sentences: {e}")
            raise

