from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
import logging
from typing import List, Dict, Any, Union
from .pipeline import Pipeline, PipelineOverwrites, PipelineConfig
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
import torch
from dataclasses import dataclass, replace
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextToEmbeddingOverwrites(PipelineOverwrites, total=False):
    encoder_model: str
    source_lang: str


class EmbeddingToTextOverwrites(PipelineOverwrites, total=False):
    decoder_model: str
    target_lang: str


@dataclass
class TextToEmbeddingPipelineConfig(PipelineConfig):
    """
    Configuration class for text-to-embedding pipelines.

    Attributes:
        encoder_model (str): The name or path of the model to be used for encoding texts into embeddings.
        source_lang (str): The source language code for the texts to be encoded.
    """
    encoder_model: str = "text_sonar_basic_encoder"
    source_lang: str = "eng_Latn"

    def with_overwrites(self, overwrites: TextToEmbeddingOverwrites):
        return replace(self, **overwrites)


@dataclass
class EmbeddingToTextPipelineConfig(PipelineConfig):
    """
    Configuration class for embedding-to-text pipelines.

    Attributes:
        decoder_model (str): The name or path of the model to be used for decoding embeddings back into texts.
        target_lang (str): The target language code for the texts to be decoded.
    """
    decoder_model: str = "text_sonar_basic_decoder"
    target_lang: str = "eng_Latn"

    def with_overwrites(self, overwrites: EmbeddingToTextOverwrites):
        return replace(self, **overwrites)


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
            embeddings = batch[f"{column}_embeddings"]
            logger.info(f"Embeddings: {embeddings}")
            reconstructed_texts = self.decode_embeddings(embeddings)
            logger.info(f"Reconstructed Texts: {reconstructed_texts}")
            batch[f"{column}_reconstructed"] = reconstructed_texts
        return batch

    def decode_embeddings(self, embeddings: List[Any]) -> List[str]:
        try:
            logger.info(f"Decoding {len(embeddings)} embeddings...")

            # Convert embeddings to tensors if they're not already
            embeddings = [torch.tensor(embed) if not isinstance(
                embed, torch.Tensor) else embed for embed in embeddings]

            # Process each embedding separately
            decoded_texts = []
            for embed in embeddings:
                if embed.dim() == 1:
                    embed = embed.unsqueeze(0)

                # Decode the embedding
                decoded = self.t2t_model.predict(
                    embed, target_lang=self.config.target_lang, batch_size=self.config.batch_size)

                decoded_texts.append(decoded)

            logger.info("Texts decoded successfully.")
            return decoded_texts
        except Exception as e:
            logger.error(f"Error decoding texts: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise


@dataclass
class HFTextToEmbeddingPipeline(Pipeline):
    config: TextToEmbeddingPipelineConfig

    def __post_init__(self):
        logger.info("Initializing text to embedding model...")
        self.t2vec_model = TextToEmbeddingModelPipeline(
            encoder=self.config.encoder_model,
            tokenizer=self.config.encoder_model,
            device=self.config.device
        )

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for column in self.config.columns:
            if column in batch:
                sentence_embeddings = []
                for item in batch[column]:
                    if isinstance(item, list):
                        # If it's a list of sentences, encode each sentence
                        embeddings = self.encode_texts(item)
                    else:
                        # If it's a single string, treat it as one sentence
                        embeddings = self.encode_texts([item])
                    sentence_embeddings.append(embeddings)
                batch[f"{column}_embeddings"] = sentence_embeddings
            else:
                logger.warning(f"Column {column} not found in batch.")
        return batch

    def encode_texts(self, texts: Union[str, List[str]]) -> List[np.ndarray]:
        try:
            if isinstance(texts, str):
                texts = [texts]
            elif not isinstance(texts, list):
                raise ValueError(
                    f"Expected string or list of strings, got {type(texts)}")

            embeddings = self.t2vec_model.predict(
                texts, source_lang=self.config.source_lang, batch_size=self.config.batch_size)
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts or sentences: {e}")
            raise

