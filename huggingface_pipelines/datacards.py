from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import yaml
import os
from .pipeline import Pipeline
from .text import TextToEmbeddingPipelineConfig, EmbeddingToTextPipelineConfig, HFEmbeddingToTextPipeline, HFTextToEmbeddingPipeline


@dataclass
class BaseDatacard(ABC):
    """
    Abstract base class for all datacards.
    """
    dataset_name: str
    description: str
    license: str
    tags: List[str]
    size_categories: List[str]
    task_categories: List[str]
    paperswithcode_id: Optional[str] = None

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the datacard to a dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseDatacard':
        """Create a datacard instance from a dictionary."""
        pass

    @classmethod
    def load_from_yaml(cls, filename: str) -> 'BaseDatacard':
        """Load a datacard from a YAML file."""
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


@dataclass
class TextDatacard(BaseDatacard):
    """
    Datacard for text datasets.
    """
    languages: List[str]
    text_features: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "languages": self.languages,
            "text_features": self.text_features
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextDatacard':
        return cls(**data)


@dataclass
class AudioDatacard(BaseDatacard):
    """
    Datacard for audio datasets.
    """
    languages: List[str]
    audio_features: List[str]
    sampling_rate: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "languages": self.languages,
            "audio_features": self.audio_features,
            "sampling_rate": self.sampling_rate
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioDatacard':
        return cls(**data)


class PipelineBuilder:
    """
    Factory class for creating pipelines based on dataset and operation type.
    """
    # TODO datacards will live in  huggingface_pipelines/datacards/

    def __init__(self, datacard_dir: str):
        self.datacard_dir = datacard_dir

    def load_datacard(self, dataset_name: str) -> BaseDatacard:
        """Load the appropriate datacard for the given dataset."""
        yaml_file = os.path.join(
            self.datacard_dir, f"{dataset_name}_datacard.yaml")
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(
                f"No datacard found for dataset: {dataset_name}")

        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        if "audio_features" in data:
            return AudioDatacard.from_dict(data)
        else:
            return TextDatacard.from_dict(data)

    def create_pipeline(self, dataset_name: str, operation: str) -> Pipeline:
        """Create and return the appropriate pipeline based on the dataset and operation."""
        datacard = self.load_datacard(dataset_name)

        if isinstance(datacard, TextDatacard):
            if operation == "textToEmbedding":
                config = TextToEmbeddingPipelineConfig(
                    columns=datacard.text_features,
                    output_path=f"output/{dataset_name}",
                    output_column_suffix="embedding",
                    source_lang=datacard.languages[0] if datacard.languages else "eng_Latn"
                )
                return HFTextToEmbeddingPipeline(config)
            elif operation == "embeddingToText":
                config = EmbeddingToTextPipelineConfig(
                    columns=[
                        f"{feature}_embedding" for feature in datacard.text_features],
                    output_path=f"output/{dataset_name}",
                    output_column_suffix="reconstructed",
                    target_lang=datacard.languages[0] if datacard.languages else "eng_Latn"
                )
                return HFEmbeddingToTextPipeline(config)
        elif isinstance(datacard, AudioDatacard):
            # Implement audio pipeline creation here
            pass

        raise ValueError(
            f"Unsupported operation '{operation}' for dataset '{dataset_name}'")
