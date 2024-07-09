import json
import logging
from datasets import load_dataset, Dataset
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from dataclasses import dataclass, field
from typing import List, Dict, Any
from .pipeline_config import PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SonarHFTextToTextPipeline:
    """
    A pipeline for encoding text datasets from HuggingFace into embeddings, decoding embeddings back to texts,
    and evaluating the quality using metrics.
    """
    config: PipelineConfig
    results: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """
        Initializes the dataset, models, and metric after the instance is created.
        """
        logger.info(
            f"Loading dataset {self.config.dataset_name} with split {self.config.dataset_split}...")
        self.dataset = load_dataset(
            self.config.dataset_name, split=self.config.dataset_split)
        logger.info("Dataset loaded. Initializing models...")
        self.t2vec_model = TextToEmbeddingModelPipeline(
            encoder=self.config.encoder_model, tokenizer=self.config.encoder_model, device=self.config.device)
        self.t2t_model = EmbeddingToTextModelPipeline(
            decoder=self.config.decoder_model, tokenizer=self.config.encoder_model, device=self.config.device)
        logger.info("Models initialized.")

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

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single batch of data, returning original, reconstructed texts and metric score.

        Args:
            batch (Dict[str, Any]): A batch of data containing texts.

        Returns:
            Dict[str, Any]: A dictionary containing original texts, reconstructed texts, and metric score.
        """
        logger.info("Processing batch...")
        texts = batch['text']
        embeddings = self.encode_texts(texts)
        reconstructed_texts = self.decode_embeddings(embeddings)
        return {'original': texts, 'reconstructed': reconstructed_texts}

    def process_batches(self):
        """
        Processes all batches in the dataset and stores the results.
        Splits the dataset into shards and processes the specified shard.
        """
        try:
            logger.info("Starting to process batches...")
            total_size = len(self.dataset)
            shard_size = total_size // self.config.num_shards

            # Select the shard
            start_idx = self.config.shard_id * shard_size
            end_idx = (self.config.shard_id + 1) * \
                shard_size if self.config.shard_id < self.config.num_shards - 1 else total_size
            dataset_shard = self.dataset.select(range(start_idx, end_idx))

            # Process the shard
            results = dataset_shard.map(
                lambda batch: self.process_batch(batch),
                batched=True,
                batch_size=self.config.batch_size,
                remove_columns=dataset_shard.column_names,
                load_from_cache_file=False
            )
            self.results.extend([{k: v[i] for k, v in results.items()}
                                for i in range(len(results[next(iter(results))]))])

            logger.info("Shard processed. Caching results...")
            if self.config.cache_to_arrow:
                self.cache_results_arrow()
                logger.info("Results cached successfully to Arrow file.")
            else:
                self.cache_results()
                logger.info("Results cached successfully to disk.")
        except Exception as e:
            logger.error(f"Error processing batches: {e}")

    def cache_results(self):
        """
        Caches the results to a JSON file.

        The results are saved in a file named 'results_shard_{shard_id}.json'.
        """
        try:
            logger.info(
                f"Caching results to results_shard_{self.config.shard_id}.json...")
            with open(f'results_shard_{self.config.shard_id}.json', 'w') as f:
                json.dump(self.results, f)
            logger.info("Results cached successfully.")
        except Exception as e:
            logger.error(f"Error caching results: {e}")

    def cache_results_arrow(self):
        """
        Caches the results to an Arrow file.

        The results are saved in a file named 'results_shard_{self.config.shard_id}.arrow'.
        """
        try:
            logger.info(f"Caching results to results_shard_{self.config.shard_id}.arrow...")
            dataset = Dataset.from_dict({"original": [result['original'] for result in self.results],
                                         "reconstructed": [result['reconstructed'] for result in self.results]})
            dataset.save_to_disk(f'results_shard_{self.config.shard_id}.arrow')
            logger.info("Results cached successfully.")
        except Exception as e:
            logger.error(f"Error caching results: {e}")
