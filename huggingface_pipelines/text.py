import json
import logging
from datasets import load_dataset
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from evaluate import load
from dataclasses import dataclass, field
from typing import List, Dict, Any
from .pipeline_factory import PipelineFactory
from .config import PipelineConfig
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SonarHFTextToTextPipeline:
    """
    A pipeline for encoding text datasets from HuggingFace into embeddings, decoding embeddings back to texts,
    and evaluating the quality using BLEU scores.
    """
    config: PipelineConfig
    results: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """
        Initializes the dataset, models, and BLEU metric after the instance is created.
        """
        logger.info(
            f"Loading dataset {self.config.dataset_name} with split {self.config.dataset_split}...")
        self.dataset = load_dataset(
            self.config.dataset_name, split=self.config.dataset_split)
        logger.info("Dataset loaded. Initializing models...")
        self.bleu_metric = load("bleu")
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

    def compute_bleu(self, original_texts: List[str], reconstructed_texts: List[str]) -> List[float]:
        """
        Computes the BLEU score between original and reconstructed texts.

        Args:
            original_texts (List[str]): A list of original texts.
            reconstructed_texts (List[str]): A list of reconstructed texts.

        Returns:
            List[float]: A list containing the BLEU score.
        """
        logger.info("Computing BLEU score...")

        logger.info(f"Original texts: {original_texts}")
        logger.info(f"Reconstructed Texts: {reconstructed_texts}")

        references = [[text.split()] for text in original_texts]
        logger.info(f"References: {references}")

        predictions = [text.split() for text in reconstructed_texts]
        logger.info(f"Predictions: {predictions}")

        bleu_score = self.bleu_metric.compute(
            predictions=predictions, references=references, smooth=True)
        logger.info(f"BLEU score computed: {bleu_score['bleu']}")
        return [bleu_score['bleu']]

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single batch of data, returning original, reconstructed texts and BLEU score.

        Args:
            batch (Dict[str, Any]): A batch of data containing texts.

        Returns:
            Dict[str, Any]: A dictionary containing original texts, reconstructed texts, and BLEU score.
        """
        logger.info("Processing batch...")
        texts = batch['text']
        embeddings = self.encode_texts(texts)
        reconstructed_texts = self.decode_embeddings(embeddings)
        bleu_score = self.compute_bleu(texts, reconstructed_texts)
        logger.info(f"Batch processed. BLEU score: {bleu_score}")
        return {'original': texts, 'reconstructed': reconstructed_texts, 'bleu': bleu_score}

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
            self.cache_results()
            logger.info("Results cached successfully.")
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

    def analyze_results(self):
        """
        Analyzes the results to determine the percentage of batches with low BLEU scores.
        """
        if not self.results:
            logger.warning("No results to analyze.")
            return

        logger.info("Analyzing results...")
        low_bleu_count = sum(
            1 for result in self.results if result['bleu'][0] < self.config.low_bleu_threshold)
        total_batches = len(self.results)
        low_bleu_percentage = (low_bleu_count / total_batches) * 100

        logger.info(
            f"Percentage of batches with BLEU score below {self.config.low_bleu_threshold}: {low_bleu_percentage:.2f}%")
        self.report_low_bleu_scores()

    def report_low_bleu_scores(self):
        """
        Reports batches with BLEU scores below the threshold.
        """
        for result in self.results:
            if result['bleu'][0] < self.config.low_bleu_threshold:
                logger.info(f"Low BLEU score detected: {result['bleu'][0]}")
                logger.info(f"Original Text: {result['original']}")
                logger.info(f"Reconstructed Text: {result['reconstructed']}")





def main():

    config = PipelineConfig(
        encoder_model="text_sonar_basic_encoder",
        decoder_model="text_sonar_basic_decoder",
        dataset_name="ag_news",
        dataset_split="test",
        source_lang="eng_Latn",
        target_lang="eng_Latn",
        batch_size=5,
        num_shards=1,
        shard_id=0,
        device="cpu"
    )

    pipeline = PipelineFactory.create_pipeline(config)

    pipeline.process_batches()
    pipeline.analyze_results()



if __name__ == "__main__":
    main()