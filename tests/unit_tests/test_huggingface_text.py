import pytest
import time
from unittest.mock import patch, MagicMock
from huggingface_pipelines.text import SonarHFTextToTextPipeline


@pytest.fixture
@patch('datasets.load_dataset')
@patch('sonar.inference_pipelines.text.TextToEmbeddingModelPipeline')
@patch('sonar.inference_pipelines.text.EmbeddingToTextModelPipeline')
@patch('evaluate.load')
def pipeline(mock_load_bleu, mock_EmbeddingToTextModelPipeline, mock_TextToEmbeddingModelPipeline, mock_load_dataset):
    # Mock the dataset
    mock_load_dataset.return_value = MagicMock()

    # Mock the BLEU metric
    mock_bleu_metric = MagicMock()
    mock_bleu_metric.compute.return_value = {'bleu': 0.5}
    mock_load_bleu.return_value = mock_bleu_metric

    # Mock the encoder and decoder models
    mock_encoder = MagicMock()
    mock_decoder = MagicMock()
    mock_TextToEmbeddingModelPipeline.return_value = mock_encoder
    mock_EmbeddingToTextModelPipeline.return_value = mock_decoder

    # Initialize the pipeline
    pipeline = SonarHFTextToTextPipeline(
        encoder_model="text_sonar_basic_encoder",
        decoder_model="text_sonar_basic_decoder",
        dataset_name="ag_news",
        dataset_split="train",
        source_lang="eng_Latn",
        target_lang="eng_Latn",
        batch_size=1
    )
    return pipeline


def test_encode_texts(pipeline):
    # Mock the predict method
    pipeline.t2vec_model.predict.return_value = [
        {'embedding': [0.1, 0.2, 0.3]}]

    texts = ["Hello world"]
    embeddings = pipeline.encode_texts(texts)

    assert len(embeddings) == 1
    assert embeddings[0]['embedding'] == [0.1, 0.2, 0.3]
    pipeline.t2vec_model.predict.assert_called_once_with(
        texts, source_lang="eng_Latn", batch_size=1)


def test_decode_embeddings(pipeline):
    # Mock the predict method
    pipeline.t2t_model.predict.return_value = ["Hello world"]

    embeddings = [{'embedding': [0.1, 0.2, 0.3]}]
    decoded_texts = pipeline.decode_embeddings(embeddings)

    assert decoded_texts == ["Hello world"]
    pipeline.t2t_model.predict.assert_called_once_with(
        embeddings, target_lang="eng_Latn", batch_size=1)


def test_compute_bleu(pipeline):
    original_texts = ["Hello world"]
    reconstructed_texts = ["Hello world"]

    bleu_score = pipeline.compute_bleu(original_texts, reconstructed_texts)

    assert bleu_score == [0.5]
    pipeline.bleu_metric.compute.assert_called_once()


def test_process_batch(pipeline):
    # Mock the methods to return expected results
    pipeline.encode_texts = MagicMock(
        return_value=[{'embedding': [0.1, 0.2, 0.3]}])
    pipeline.decode_embeddings = MagicMock(return_value=["Hello world"])
    pipeline.compute_bleu = MagicMock(return_value=[0.5])

    batch = {'text': ["Hello world"]}
    result = pipeline.process_batch(batch)

    assert result['original'] == ["Hello world"]
    assert result['reconstructed'] == ["Hello world"]
    assert result['bleu'] == [0.5]
    pipeline.encode_texts.assert_called_once_with(["Hello world"])
    pipeline.decode_embeddings.assert_called_once_with(
        [{'embedding': [0.1, 0.2, 0.3]}])
    pipeline.compute_bleu.assert_called_once_with(
        ["Hello world"], ["Hello world"])


@patch('huggingface_pipelines.text.SonarHFTextToTextPipeline.process_batch')
def test_process_batches(mock_process_batch, pipeline):
    # Mock the dataset to return 1000 items
    pipeline.dataset.select = MagicMock(return_value=range(1000))

    # Mock the process_batch method
    mock_process_batch.return_value = {
        'original': ["Hello world"],
        'reconstructed': ["Hello world"],
        'bleu': [0.5]
    }

    pipeline.process_batches()

    assert len(pipeline.results) == 1000
    assert pipeline.results[0]['original'] == ["Hello world"]
    assert pipeline.results[0]['reconstructed'] == ["Hello world"]
    assert pipeline.results[0]['bleu'] == [0.5]


@patch('builtins.open')
@patch('json.dump')
def test_cache_results(mock_json_dump, mock_open, pipeline):
    pipeline.results = [{'original': ["Hello world"],
                         'reconstructed': ["Hello world"], 'bleu': [0.5]}]

    pipeline.cache_results()

    mock_open.assert_called_once_with('results.json', 'w')
    mock_json_dump.assert_called_once_with(pipeline.results, mock_open())


def test_analyze_results(pipeline):
    pipeline.results = [
        {'original': ["Hello world"], 'reconstructed': [
            "Hello world"], 'bleu': [0.4]},
        {'original': ["Goodbye world"], 'reconstructed': [
            "Goodbye world"], 'bleu': [0.6]}
    ]

    with patch.object(pipeline, 'report_low_bleu_scores') as mock_report:
        pipeline.analyze_results()

        assert pipeline.results
        assert mock_report.called
        assert mock_report.call_count == 1
        assert mock_report.call_args_list[0][0] == ()


def test_encode_texts_empty(pipeline):
    # Test with empty input
    texts = [""]
    embeddings = pipeline.encode_texts(texts)

    assert len(embeddings) == 1
    pipeline.t2vec_model.predict.assert_called_once_with(
        texts, source_lang="eng_Latn", batch_size=1)


def test_decode_embeddings_empty(pipeline):
    # Test with empty embedding
    embeddings = [{'embedding': []}]
    decoded_texts = pipeline.decode_embeddings(embeddings)

    assert len(decoded_texts) == 1
    pipeline.t2t_model.predict.assert_called_once_with(
        embeddings, target_lang="eng_Latn", batch_size=1)


def test_compute_bleu_edge_cases(pipeline):
    # Test with empty original and reconstructed texts
    original_texts = [""]
    reconstructed_texts = [""]

    bleu_score = pipeline.compute_bleu(original_texts, reconstructed_texts)

    assert bleu_score == [0.5]
    pipeline.bleu_metric.compute.assert_called_once()


def test_process_batch_performance(pipeline):
    # Test the performance with a large batch
    large_text = " ".join(["Hello world"] * 1000)  # Very long text
    batch = {'text': [large_text]}

    start_time = time.time()
    result = pipeline.process_batch(batch)
    end_time = time.time()

    assert result['original'] == [large_text]
    assert result['reconstructed']  # Check if reconstructed text is not empty
    assert end_time - start_time < 1.0  # Ensure the processing is reasonably fast


def test_process_batch_error_handling(pipeline):
    # Mock the methods to raise an exception
    pipeline.encode_texts = MagicMock(side_effect=Exception("Encoding error"))
    pipeline.decode_embeddings = MagicMock(
        side_effect=Exception("Decoding error"))

    batch = {'text': ["Hello world"]}

    result = pipeline.process_batch(batch)

    assert result == {}
    pipeline.encode_texts.assert_called_once_with(["Hello world"])


@patch('datasets.load_dataset', side_effect=Exception("Dataset load error"))
def test_pipeline_initialization_error(mock_load_dataset):
    with pytest.raises(Exception, match="Dataset load error"):
        SonarHFTextToTextPipeline(
            encoder_model="text_sonar_basic_encoder",
            decoder_model="text_sonar_basic_decoder",
            dataset_name="ag_news",
            dataset_split="train",
            source_lang="eng_Latn",
            target_lang="eng_Latn",
            batch_size=1
        )

