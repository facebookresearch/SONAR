from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from .pipeline_config import EmbeddingToTextPipelineConfig
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from .pipeline import Pipeline
from .pipeline_config import TextToEmbeddingPipelineConfig
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HFEmbeddingToTextPipeline(Pipeline):
    """
    A pipeline for decoding embeddings back into texts.
    """
    config: EmbeddingToTextPipelineConfig

    def __post_init__(self):
        """
        Initializes the model.
        """
        logger.info("Initializing embedding to text model...")
        self.t2t_model = EmbeddingToTextModelPipeline(
            decoder=self.config.decoder_model, tokenizer=self.config.decoder_model, device=self.config.device)
        logger.info("Model initialized.")

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Processes a single batch of data by decoding embeddings back into texts.

        Args:
            batch (Dict[str, Any]): A batch of data containing embeddings.

        Returns:
            Dict[str, List[str]]: The batch with reconstructed texts added.
        """
        for column in self.config.columns:
            embeddings = batch[column + '_embeddings']
            logger.info(f"Embeddings: {embeddings}")
            reconstructed_texts = self.decode_embeddings(embeddings)
            batch[column + '_reconstructed'] = reconstructed_texts
        return batch

    def decode_embeddings(self, embeddings: List[Any]) -> List[str]:
        """
        Decodes a list of embeddings back into texts.

        Args:
            embeddings (List[Any]): A list of embeddings to be decoded.

        Returns:
            List[str]: A list of decoded texts.
        """
        try:
            logger.info(f"Decoding {len(embeddings)} embeddings...")
            embeddings = [torch.tensor(embed) for embed in embeddings]
            decoded_texts = self.t2t_model.predict(
                embeddings, target_lang=self.config.target_lang, batch_size=self.config.batch_size)
            logger.info("Texts decoded successfully.")
            return decoded_texts
        except Exception as e:
            logger.error(f"Error decoding texts: {e}")
            raise


@dataclass
class HFTextToEmbeddingPipeline(Pipeline):
    """
    A pipeline for encoding text datasets from HuggingFace into embeddings.
    """
    config: TextToEmbeddingPipelineConfig

    def __post_init__(self):
        """
        Initializes the model.
        """
        logger.info("Initializing text to embedding model...")
        self.t2vec_model = TextToEmbeddingModelPipeline(
            encoder=self.config.encoder_model, tokenizer=self.config.encoder_model, device=self.config.device)
        logger.info("Model initialized.")

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Processes a single batch of data by encoding texts into embeddings.

        Args:
            batch (Dict[str, Any]): A batch of data containing texts.

        Returns:
            Dict[str, torch.Tensor]: The batch with embeddings added.
        """
        for column in self.config.columns:
            texts = batch[column]
            embeddings = self.encode_texts(texts)
            batch[column + '_embeddings'] = embeddings
        return batch

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encodes a list of texts into embeddings.

        Args:
            texts (List[str]): A list of texts to be encoded.

        Returns:
            Torch.Tensor A list of encoded embeddings in tensor format.
        """
        try:
            logger.info(f"Encoding {len(texts)} texts...")
            embeddings = self.t2vec_model.predict(
                texts, source_lang=self.config.source_lang, batch_size=self.config.batch_size)
            logger.info("Texts encoded successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
