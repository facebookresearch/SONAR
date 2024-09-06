from abc import ABC
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, TypedDict


class DatasetOverwrites(TypedDict, total=False):
    """
    TypedDict for dataset configuration overwrites.

    Attributes:
        dataset_name (str): Name of the dataset.
        dataset_split (str): Split of the dataset (e.g., 'train', 'test').
        output_dir (str): Directory for output.
        streaming (bool): Whether to use streaming mode.
        config (str): Specific dataset configuration.
        trust_remote_code (bool): Whether to trust remote code.
        world_size (int): Number of shards for distributed processing.
        rank (int): Rank of the current process.
    """

    dataset_name: str
    dataset_split: str
    output_dir: str
    streaming: bool
    config: str
    trust_remote_code: bool
    world_size: int
    rank: int


@dataclass
class DatasetConfig(ABC):
    """
    Abstract base configuration class for loading and sharding datasets.

    This class provides a structured way to configure dataset loading parameters
    and includes methods for loading and sharding datasets from Hugging Face.

    Attributes:
        dataset_name (str): The name or path of the dataset to load.
        dataset_split (str): The split of the dataset to use (e.g., 'train', 'test', 'validation').
        output_dir (str): The directory to store the output datasets.
        streaming (bool): Whether to stream the dataset or load it entirely into memory.
        config (str): The specific configuration of the dataset to load, if applicable.
        world_size (int): The total number of shards to split the dataset into.
        rank (int): The index of the shard to retrieve (0-based).
        trust_remote_code (bool): Whether to trust remote code when loading the dataset.

    Note:
        The `world_size` and `rank` attributes are particularly useful for distributed data processing,
        allowing the dataset to be split across multiple processes or machines.
    """

    dataset_name: str
    dataset_split: str
    output_dir: str
    streaming: bool = False
    config: Optional[str] = None
    world_size: int = 1
    rank: int = 0
    trust_remote_code: bool = False

    def load_dataset(self):
        """
        Loads and optionally shards the dataset based on the configuration settings.

        This method uses the Hugging Face datasets library to load the dataset.
        If `world_size` is greater than 1, it also shards the dataset.

        Returns:
            datasets.Dataset: The loaded and potentially sharded dataset.

        Raises:
            ValueError: If the dataset cannot be loaded with the given configuration.
            ImportError: If the 'datasets' library is not installed.

        Note:
            Ensure that the 'datasets' library is installed before calling this method.
        """
        # We ignore because no offical HF stubs available

        from datasets import load_dataset  # type: ignore

        dataset_kwargs = self.get_dataset_kwargs()
        dataset = load_dataset(**dataset_kwargs)

        self.validate_world_size_and_rank()

        if not self.streaming and self.world_size > 1:
            dataset = dataset.shard(num_shards=self.world_size, index=self.rank)

        return dataset

    def get_dataset_kwargs(self) -> Dict[str, Any]:
        """
        Returns the kwargs for load_dataset function.

        This method prepares the keyword arguments used in the load_dataset function.

        Returns:
            dict[str, Any]: A dictionary of keyword arguments for load_dataset.
        """
        dataset_kwargs = {
            "path": self.dataset_name,
            "name": self.config,
            "split": self.dataset_split,
            "trust_remote_code": self.trust_remote_code,
            "streaming": self.streaming,
        }
        return {k: v for k, v in dataset_kwargs.items() if v is not None}

    def validate_world_size_and_rank(self):
        """
        Validates world_size and rank.

        Raises:
            AssertionError: If world_size or rank are invalid.
        """
        assert (
            self.world_size >= 1
        ), f"Invalid world_size: {self.world_size}. It should be >= 1."
        assert (
            0 <= self.rank < self.world_size
        ), f"Invalid rank: {self.rank}. It should be between 0 and {self.world_size - 1}."

    def with_overwrites(self, overwrites: DatasetOverwrites):
        """
        Creates a new instance with specified overwrites.

        This method allows for the creation of a new configuration object
        with some attributes overwritten, without modifying the original instance.

        Args:
            overwrites (DatasetOverwrites): A dictionary of attributes to overwrite.

        Returns:
            BaseDatasetConfig: A new instance with the specified overwrites applied.

        Example:
            new_config = config.with_overwrites({"dataset_split": "test", "world_size": 4})
        """
        return replace(self, **overwrites)
