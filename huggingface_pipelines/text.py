

import json
import logging
from datasets import load_dataset
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from evaluate import load
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SonarHFTextToTextPipeline:
    """
    A pipeline for encoding text dataset from HuggingFace into embeddings and decoding embeddings back to texts,
    evaluating the quality using BLEU scores.
    """
    encoder_model: str
    decoder_model: str
    dataset_name: str
    dataset_split: str
    source_lang: str
    target_lang: str
    batch_size: int
    results: List[Dict[str, Any]] = field(default_factory=list)
    low_bleu_threshold: float = 0.5

    def __post_init__(self):
        logger.info(
            f"Loading dataset {self.dataset_name} with split {self.dataset_split}...")
        self.dataset = load_dataset(
            self.dataset_name, split=self.dataset_split)
        logger.info("Dataset loaded. Initializing models...")
        self.bleu_metric = load("bleu")
        self.t2vec_model = TextToEmbeddingModelPipeline(
            encoder=self.encoder_model, tokenizer=self.encoder_model)
        self.t2t_model = EmbeddingToTextModelPipeline(
            decoder=self.decoder_model, tokenizer=self.encoder_model)
        logger.info("Models initialized.")

    def encode_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Encodes a list of texts into embeddings."""
        try:
            logger.info(f"Encoding {len(texts)} texts...")
            embeddings = self.t2vec_model.predict(
                texts, source_lang=self.source_lang, batch_size=self.batch_size)
            logger.info("Texts encoded successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return []

    def decode_embeddings(self, embeddings: List[Any]) -> List[str]:
        """Decodes a list of embeddings back into texts."""
        try:
            logger.info(f"Decoding {len(embeddings)} embeddings...")
            decoded_texts = self.t2t_model.predict(
                embeddings, target_lang=self.target_lang, batch_size=self.batch_size)
            logger.info("Texts decoded successfully.")
            return decoded_texts
        except Exception as e:
            logger.error(f"Error decoding texts: {e}")
            return []

    def compute_bleu(self, original_texts: List[str], reconstructed_texts: List[str]) -> List[float]:
        """Computes the BLEU score between original and reconstructed texts."""
        logger.info("Computing BLEU score...")

        logger.info(f"Original texts: {original_texts}")
        logger.info(f"Reconstructed Texts: {reconstructed_texts}")

        references = [text.split()
                      for text in original_texts]

        logger.info(f"References: {references}")

        predictions = reconstructed_texts

        logger.info(f"Predictions: {predictions}")

        bleu_score = self.bleu_metric.compute(
            predictions=predictions, references=references, smooth=True)
        logger.info(f"BLEU score computed: {bleu_score['bleu']}")
        return [bleu_score['bleu']]

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single batch of data, returning original, reconstructed texts and BLEU score."""
        logger.info("Processing batch...")
        texts = batch['text']
        embeddings = self.encode_texts(texts)
        reconstructed_texts = self.decode_embeddings(embeddings)
        bleu_score = self.compute_bleu(texts, reconstructed_texts)
        logger.info(f"Batch processed. BLEU score: {bleu_score}")
        return {'original': texts, 'reconstructed': reconstructed_texts, 'bleu': bleu_score}

    def process_batches(self):
        """Processes all batches in the dataset and stores the results."""
        try:
            logger.info("Starting to process batches...")
            # Limit to the first 1000 examples
            dataset = self.dataset.select(range(min(len(self.dataset), 1000)))

            # Process the dataset using map
            results = dataset.map(
                lambda batch: self.process_batch(batch),
                batched=True,
                batch_size=self.batch_size,
                remove_columns=dataset.column_names,
                load_from_cache_file=False
            )
            self.results.extend([{k: v[i] for k, v in results.items()}
                                for i in range(len(results[next(iter(results))]))])

            logger.info("All batches processed. Caching results...")
            self.cache_results()
            logger.info("Results cached successfully.")
        except Exception as e:
            logger.error(f"Error processing batches: {e}")

    def cache_results(self):
        """Caches the results to a JSON file."""
        try:
            logger.info("Caching results to results.json...")
            with open('results.json', 'w') as f:
                json.dump(self.results, f)
            logger.info("Results cached successfully.")
        except Exception as e:
            logger.error(f"Error caching results: {e}")

    def analyze_results(self):
        """Analyzes the results to determine the percentage of batches with low BLEU scores."""
        if not self.results:
            logger.warning("No results to analyze.")
            return

        logger.info("Analyzing results...")
        low_bleu_count = sum(
            1 for result in self.results if result['bleu'][0] < self.low_bleu_threshold)
        total_batches = len(self.results)
        low_bleu_percentage = (low_bleu_count / total_batches) * 100

        logger.info(
            f"Percentage of batches with BLEU score below {self.low_bleu_threshold}: {low_bleu_percentage:.2f}%")
        self.report_low_bleu_scores()

    def report_low_bleu_scores(self):
        """Reports batches with BLEU scores below the threshold."""
        for result in self.results:
            if result['bleu'][0] < self.low_bleu_threshold:
                logger.info(f"Low BLEU score detected: {result['bleu'][0]}")
                logger.info(f"Original Text: {result['original']}")
                logger.info(f"Reconstructed Text: {result['reconstructed']}")
