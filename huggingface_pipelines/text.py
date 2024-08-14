from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
import logging
from typing import List, Dict, Any
from .pipeline import Pipeline, PipelineConfig, PipelineFactory
import torch
from dataclasses import dataclass
import numpy as np
from .dataset import DatasetConfig

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

    Example:
        config = TextToEmbeddingPipelineConfig(
            encoder_model="text_sonar_basic_encoder",
            source_lang="eng_Latn",
            columns=["text"],
            output_column_suffix="embedding",
            batch_size=32,
            device="cuda",
            max_seq_len=512
        )
    """
    max_seq_len: int = None
    encoder_model: str = "text_sonar_basic_encoder"
    source_lang: str = "eng_Latn"


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

    Example:
        config = EmbeddingToTextPipelineConfig(
            decoder_model="text_sonar_basic_decoder",
            target_lang="eng_Latn",
            columns=["embedding"],
            output_column_suffix="text",
            batch_size=32,
            device="cuda"
        )
    """
    decoder_model: str = "text_sonar_basic_decoder"
    target_lang: str = "eng_Latn"


class HFEmbeddingToTextPipeline(Pipeline):
    """
    Pipeline for converting embeddings back to text using a Hugging Face model.

    This pipeline takes embeddings as input and decodes them into text using a specified decoder model.

    Attributes:
        config (EmbeddingToTextPipelineConfig): Configuration for the pipeline.
        t2t_model (EmbeddingToTextModelPipeline): The model used for decoding embeddings to text.
    """

    def __init__(self, config: EmbeddingToTextPipelineConfig):
        """
        Initialize the embedding-to-text pipeline.

        Args:
            config (EmbeddingToTextPipelineConfig): Configuration for the pipeline.
        """
        super().__init__(config)
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
            assert all(isinstance(item, np.array) for item in embeddings), \
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
            logger.debug(f"{column} column reconstructed: {batch[column][:5]}")

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

            if isinstance(embeddings, torch.Tensor):
                embeddings_tensor = embeddings.detach().cpu()
            else:
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

    def __init__(self, config: TextToEmbeddingPipelineConfig):
        """
        Initialize the text-to-embedding pipeline.

        Args:
            config (TextToEmbeddingPipelineConfig): Configuration for the pipeline.
        """
        super().__init__(config)
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
            Dict[str, Any]: Processed batch with encoded embeddings.
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
                logger.debug(
                    f"{column} column embeddings: {batch[column][:5]}")
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

                batch_embeddings = batch_embeddings.detach().cpu().numpy()
                embeddings.extend(batch_embeddings)

            # We return this as np array for more efficient memory representation
            return np.array(embeddings)
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
            "device": "cuda"
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

