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
class DatasetConfig():
    """
    Configuration class for loading and sharding datasets.

    Attributes:
        dataset_name (str): The name of the dataset to load.
        dataset_split (str): The split of the dataset to use (e.g., 'train', 'test', 'validation').
        world_size (int): The number of shards to split the dataset into. Defaults to 1.
        rank (int): The ID of the shard to retrieve. Defaults to 0.
        cache_dir (str): The directory to cache the loaded dataset. Defaults to None.
        trust_remote_code (bool): Whether to trust remote code when loading the dataset. Defaults to False.
    """
    dataset_name: str
    dataset_split: str
    config: str = None
    world_size: int = 1
    rank: int = 0
    cache_dir: str = None
    trust_remote_code: bool = False
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def load_dataset(self):
        """
        Loads and shards the dataset based on the configuration settings.

        Returns:
            datasets.Dataset: The loaded and sharded dataset.
        """
        from datasets import load_dataset

        dataset_kwargs = {
            "path": self.dataset_name,
            "name": self.config,
            "split": self.dataset_split,
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
        }
        dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if v is not None}
        
        dataset = load_dataset(**dataset_kwargs)

        

        # Shard the dataset
        if self.world_size > 1:
            dataset = dataset.shard(
                num_shards=self.world_size, index=self.rank)

        return dataset

    def with_overwrites(self, overwrites: DatasetOverwrites):
        return replace(self, **overwrites)
