import logging
from datasets import Dataset as HFDataset
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from typing import List, Dict, Any
from dataclasses import dataclass
from .pipeline import Pipeline
from .pipeline_config import TextPipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextToTextHFPipeline(Pipeline):
    """
    A pipeline for encoding text datasets from HuggingFace into embeddings, decoding embeddings back into texts,
    and evaluating the quality using metrics.
    """
    config: TextPipelineConfig

    def __post_init__(self):
        """
        Initializes the models.
        """
        logger.info("Initializing models...")
        self.t2vec_model = TextToEmbeddingModelPipeline(
            encoder=self.config.encoder_model, tokenizer=self.config.encoder_model, device=self.config.device)
        self.t2t_model = EmbeddingToTextModelPipeline(
            decoder=self.config.decoder_model, tokenizer=self.config.encoder_model, device=self.config.device)
        logger.info("Models initialized.")

    def process_batch(self, batch: Dict[str, Any], dataset: HFDataset) -> HFDataset:
        """
        Processes a single batch of data, encoding and decoding texts, and updating the dataset.

        Args:
            batch (Dict[str, Any]): A batch of data containing texts.
            dataset (HFDataset): The dataset to update.

        Returns:
            HFDataset: The updated dataset.
        """
        for column in self.config.columns:
            texts = batch[column]
            embeddings = self.encode_texts(texts)
            reconstructed_texts = self.decode_embeddings(embeddings)
            dataset = dataset.add_column(column + '_embeddings', embeddings)
            dataset = dataset.add_column(
                column + '_reconstructed', reconstructed_texts)
        return dataset

    def encode_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Encodes a list of texts into embeddings.

        Args:
            texts (List[str]): A list of texts to be encoded.

        Returns:
            List[Dict[str, Any]]: A list of encoded embeddings.
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
            decoded_texts = self.t2t_model.predict(
                embeddings, target_lang=self.config.target_lang, batch_size=self.config.batch_size)
            logger.info("Texts decoded successfully.")
            return decoded_texts
        except Exception as e:
            logger.error(f"Error decoding texts: {e}")
            raise

