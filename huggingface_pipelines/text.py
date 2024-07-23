import spacy
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
import logging
from typing import List, Dict, Any
from .pipeline import Pipeline, PipelineOverwrites, PipelineConfig
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
import torch
from dataclasses import dataclass, replace

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
class HFTextToEmbeddingPipeline(Pipeline):
    config: TextToEmbeddingPipelineConfig

    def __post_init__(self):
        logger.info("Initializing text to embedding model...")
        self.t2vec_model = TextToEmbeddingModelPipeline(
            encoder=self.config.encoder_model,
            tokenizer=self.config.encoder_model,
            device=self.config.device
        )

        logger.info(
            f"Initializing spaCy model for language: {self.config.source_lang}")
        self.nlp = self.load_spacy_model(self.config.source_lang)

        logger.info("Models initialized.")

    def load_spacy_model(self, lang_code: str):
        """
        Loads the appropriate spaCy model based on the language code.
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

    def segment_sentences(self, text: str) -> List[str]:
        """
        Segments a text into sentences using the loaded spaCy model.
        """
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for column in self.config.columns:
            texts = batch[column]

            # Segment sentences
            sentences = [self.segment_sentences(text) for text in texts]
            batch[f'{column}_sentences'] = sentences

            # Log a subset of sentences (first 5)
            logger.info(f"Sample of {column}_sentences: {sentences[:5]}")

            # Encode sentences

            sentence_embeddings = [self.encode_texts(
                text_sentences) for text_sentences in sentences]
            batch[f'{column}_embeddings'] = sentence_embeddings

            # Log shape of the first 5 embeddings
            logger.info(
                f"Sample of {column}_embeddings shapes: {[emb.shape for emb in sentence_embeddings[:5]]}")

        return batch

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        try:
            logger.info(f"Encoding {len(texts)} texts or sentences...")
            embeddings = self.t2vec_model.predict(
                texts,
                source_lang=self.config.source_lang,
                batch_size=self.config.batch_size
            )
            logger.info("Texts or sentences encoded successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts or sentences: {e}")
            raise


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
                    embed, target_lang=self.config.target_lang, batch_size=1)

                # Join the decoded sentences into a single string
                decoded_text = " ".join([sent for sent in decoded[0]])
                decoded_texts.append(decoded_text)

            logger.info("Texts decoded successfully.")
            return decoded_texts
        except Exception as e:
            logger.error(f"Error decoding texts: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise


SPACY_MODELS = {
    "eng_Latn": "en_core_web_sm",
    "fra_Latn": "fr_core_news_sm",
    "deu_Latn": "de_core_news_sm",
    "spa_Latn": "es_core_news_sm",
    "ita_Latn": "it_core_news_sm",
    "por_Latn": "pt_core_news_sm",
    "nld_Latn": "nl_core_news_sm",
    # Add more languages and their corresponding spaCy models as needed
}


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

        logger.info(
            f"Initializing spaCy model for language: {self.config.source_lang}")
        self.nlp = self.load_spacy_model(self.config.source_lang)

        logger.info("Models initialized.")

    def load_spacy_model(self, lang_code: str):
        """
        Loads the appropriate spaCy model based on the language code.
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

    def segment_sentences(self, text: str) -> List[str]:
        """
        Segments a text into sentences using the loaded spaCy model.
        """
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for column in self.config.columns:
            texts = batch[column]

            # Segment sentences
            sentences = [self.segment_sentences(text) for text in texts]
            batch[f'{column}_sentences'] = sentences

            # Log a subset of sentences (first 5)
            logger.info(f"Sample of {column}_sentences: {sentences[:5]}")

            # Encode sentences

            sentence_embeddings = [self.encode_texts(
                text_sentences) for text_sentences in sentences]
            batch[f'{column}_embeddings'] = sentence_embeddings

            # Log shape of the first 5 embeddings
            logger.info(
                f"Sample of {column}_embeddings shapes: {[emb.shape for emb in sentence_embeddings[:5]]}")

        return batch

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        try:
            logger.info(f"Encoding {len(texts)} texts or sentences...")
            embeddings = self.t2vec_model.predict(
                texts,
                source_lang=self.config.source_lang,
                batch_size=self.config.batch_size
            )
            logger.info("Texts or sentences encoded successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts or sentences: {e}")
            raise
