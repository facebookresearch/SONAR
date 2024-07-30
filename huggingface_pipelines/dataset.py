from typing import TypedDict
from dataclasses import dataclass, field, replace
import uuid


class DatasetOverwrites(TypedDict, total=False):
    dataset_name: str
    dataset_split: str
    world_size: int
    rank: int
    cache_dir: str


@dataclass
class DatasetConfig:
    """
    Configuration class for loading and sharding datasets.

    This class provides a structured way to configure dataset loading parameters
    and includes methods for loading and sharding datasets.

    Attributes:
        dataset_name (str): The name or path of the dataset to load.
        dataset_split (str): The split of the dataset to use (e.g., 'train', 'test', 'validation').
        output_dir (str): The directory to store the output datasets.
        streaming (bool): Whether to stream the dataset or load it entirely into memory. Defaults to False.
        config (str): The specific configuration of the dataset to load, if applicable. Defaults to None.
        world_size (int): The total number of shards to split the dataset into. Defaults to 1.
        rank (int): The index of the shard to retrieve (0-based). Defaults to 0.
        cache_dir (str): The directory to cache the loaded dataset. If None, uses the default cache. Defaults to None.
        trust_remote_code (bool): Whether to trust remote code when loading the dataset. Use with caution. Defaults to False.
        uuid (str): A unique identifier for this configuration instance. Automatically generated.

    Note:
        The `world_size` and `rank` attributes are particularly useful for distributed data processing,
        allowing the dataset to be split across multiple processes or machines.
    """

    dataset_name: str
    dataset_split: str
    output_dir: str
    streaming: bool = False
    config: str = None
    world_size: int = 1
    rank: int = 0
    cache_dir: str = None
    trust_remote_code: bool = False
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

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
        from datasets import load_dataset

        dataset_kwargs = {
            "path": self.dataset_name,
            "name": self.config,
            "split": self.dataset_split,
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
            "streaming": self.streaming
        }
        dataset_kwargs = {k: v for k,
                          v in dataset_kwargs.items() if v is not None}

        dataset = load_dataset(**dataset_kwargs)

        # Shard the dataset if world_size > 1
        if not self.streaming and self.world_size > 1:
            dataset = dataset.shard(
                num_shards=self.world_size, index=self.rank)

        return dataset

    def with_overwrites(self, overwrites: DatasetOverwrites):
        """
        Creates a new DatasetConfig instance with specified overwrites.

        This method allows for the creation of a new configuration object
        with some attributes overwritten, without modifying the original instance.

        Args:
            overwrites (DatasetOverwrites): A dictionary of attributes to overwrite.

        Returns:
            DatasetConfig: A new instance of DatasetConfig with the specified overwrites applied.

        Example:
            new_config = config.with_overwrites({"dataset_split": "test", "world_size": 4})
        """
        return replace(self, **overwrites)

