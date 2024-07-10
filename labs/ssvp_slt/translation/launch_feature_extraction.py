# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import asyncio
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, List, Tuple, Union

import hydra
import pandas as pd
import ssvp_slt.modeling.sign_hiera as sign_hiera
import ssvp_slt.util.misc as misc
import torch
from einops import rearrange
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from ssvp_slt.data.video_dataset import VideoDataset
from ssvp_slt.modeling.sign_hiera import SignHiera
from stopes.core import Launcher, Requirements, StopesModule
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class LauncherConfig:
    cluster: str = "slurm"
    """Cluster to use by stopes launcher, e.g. 'slurm' or 'local'"""

    partition: str = "gpu"
    """Partition to use in slurm environment"""

    max_jobarray_jobs: int = 128
    """Maximum number of jobs to launch in parallel"""


@dataclass
class FeatureExtractionConfig:
    data_dir: str = MISSING
    """Folder on disk containing manifests and videos"""

    output_dir: str = MISSING
    """Folder on disk where features will be saved"""

    model_name: str = MISSING
    """Hiera model identifier, e.g. hiera_base_128x224"""

    pretrained_model_path: str = MISSING
    """Path to trained feature extractor weights"""

    from_clip: bool = False
    """Whether to initialize weights from a trained CLIP model"""

    split: str = "test"
    """Dataset split, i.e. train, val or test"""

    fp16: bool = False
    """Whether to extract features in fp16 rather than fp32"""

    do_aug: bool = False
    """Whether to perform data augmentation"""

    epochs: int = 1
    """Number of epochs to extract (useful when performing augmentation)"""

    num_frames: int = 128
    """Number of video frames per clip"""

    sampling_rate: int = 2
    """Sample every n-th frame"""

    target_fps: int = 25
    """Target framerate in frames per second"""

    num_items_per_shard: int = 50
    """Number of videos that will be processed sequentially in each shard"""

    max_batch_size: int = 2
    """Maximum batch size (in 128-frame clips). Videos longer than this will be processed in shards."""

    video_backend: str = "pyav"
    """Video backend. Should match the video backend used during pretraining."""

    launcher: LauncherConfig = LauncherConfig()
    """Stopes launcher configuration"""


def shard_generator(data: Any, shard_size: int) -> Generator[Any, None, None]:
    for i in range(0, len(data), shard_size):
        yield data[i : i + shard_size]


class FeatureExtractionModule(StopesModule):
    def __init__(self, config: FeatureExtractionConfig):
        super().__init__(config, FeatureExtractionConfig)

        manifest_file = Path(self.config.data_dir) / "manifests" / f"{self.config.split}.tsv"

        self.num_items = len(
            pd.read_csv(
                manifest_file,
                delimiter="\t",
                names=["video_name", "duration", "caption"],
                quoting=3,
            )
        )

        Path(self.config.output_dir).mkdir(exist_ok=True, parents=True)

    def requirements(self) -> Requirements:
        """Requirements for each submitted job"""
        return Requirements(
            nodes=1,
            mem_gb=6,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=1,
            timeout_min=60 * 72,
        )

    def name(self) -> str:
        name = (
            f"feature_extractor_{self.config.split}_{self.config.model_name}_"
            f"{Path(self.config.pretrained_model_path).stem}"
        )
        return name

    @property
    def num_shards(self) -> int:
        return math.ceil(self.num_items / self.config.num_items_per_shard)

    def array(self) -> List[Any]:
        return [1] * self.num_shards

    def load_model(self, device: Union[torch.device, str] = "cuda") -> SignHiera:
        """
        Loads a pretrained SignHiera model for feature extraction and moves it to specified device
        """

        if self.config.from_clip:
            if "hiera" in self.config.model_name:
                model = SignHiera.from_clip_model(
                    self.config.model_name, self.config.pretrained_model_path
                )
            else:
                raise ValueError(
                    f"Loading `{self.config.model_name}` from a CLIP model is not supported."
                )
        else:
            model = sign_hiera.__dict__[self.config.model_name](pretrained=True, strict=False)
            misc.load_model(model, self.config.pretrained_model_path)

        model.head = nn.Identity()

        print(f"Number of parameters: {sum([p.numel() for p in model.parameters()])}")

        model.eval()
        model.to(device)

        return model

    def get_dataloader(self, indices: Tuple[int, int]) -> DataLoader:
        """
        Creates a dataloader for video feature extraction
        Uses a fixed T//2 stride for the sliding window
        """

        if self.config.video_backend == "cuda":
            torch.multiprocessing.set_start_method("spawn")

        dataset = VideoDataset(
            mode=self.config.split,
            video_backend=self.config.video_backend,
            target_fps=self.config.target_fps,
            data_dir=self.config.data_dir,
            sampling_rate=self.config.sampling_rate,
            num_frames=self.config.num_frames,
            rand_aug=self.config.do_aug,
            train_random_horizontal_flip=self.config.do_aug,
            train_random_crop=self.config.do_aug,
            feature_extraction=True,
            feature_extraction_stride=self.config.num_frames // 2,
            indices=indices,
            gpu=(torch.cuda.current_device() if self.config.video_backend == "cuda" else None),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=1 if self.config.video_backend == "cuda" else 2,
            persistent_workers=True,
            pin_memory=not self.config.video_backend == "cuda",
        )

        return dataloader

    def run(self, iteration_value: Any, iteration_index: int) -> None:
        """
        Extract features for a shard of the data
        """
        output_dir = Path(self.config.output_dir)

        start_id, end_id = (
            self.config.num_items_per_shard * iteration_index,
            self.config.num_items_per_shard * (iteration_index + 1),
        )
        print(f"Indices: {start_id}-{end_id}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device {device}")

        model = self.load_model(device=device)
        dataloader = self.get_dataloader(indices=(start_id, end_id))

        # Split data into folders using prefix based on their video name
        # Helps reduce load when file system is NFS
        fids = [
            Path(dataloader.dataset.path_to_videos[i]).stem for i in range(len(dataloader.dataset))
        ]
        prefixes = [fid[:5] for fid in fids]
        for epoch in tqdm(range(self.config.epochs), desc="Creating folder structure"):
            for prefix in prefixes:
                prefix_path = output_dir / str(epoch) / prefix
                prefix_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            print(f"Epoch: {epoch}")

            prefetcher = misc.Prefetcher(dataloader, device=device)

            start_time = time.time()
            idx = 0
            pbar = tqdm(total=len(dataloader))

            batch = next(prefetcher)
            while batch is not None:
                frames = batch["frames"].float()
                padding = batch["padding"]

                if frames.dim() == 6:
                    frames = rearrange(frames, "b r c t h w -> (b r) c t h w")
                if padding.dim() == 2:
                    padding = rearrange(padding, "b r -> (b r)")

                # Process in chunks
                if len(frames) > self.config.max_batch_size:
                    shard_outputs = []
                    frames_shards = shard_generator(frames, self.config.max_batch_size)
                    padding_shards = shard_generator(padding, self.config.max_batch_size)

                    for frames_shard, padding_shard in zip(frames_shards, padding_shards):

                        with torch.inference_mode(), torch.cuda.amp.autocast(
                            enabled=self.config.fp16
                        ):
                            shard_output = model.extract_features(
                                frames_shard, padding=padding_shard
                            ).cpu()
                        if len(shard_output.shape) == 1:
                            shard_output = shard_output.unsqueeze(0)
                        shard_outputs.append(shard_output)

                    outputs = torch.concatenate(shard_outputs, dim=0)

                else:
                    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.config.fp16):
                        outputs = model.extract_features(frames, padding=padding).detach().cpu()

                fid = fids[idx]
                prefix = prefixes[idx]

                # Save to disk
                output_file = output_dir / str(epoch) / prefix / f"{fid}.pt"
                torch.save(outputs, output_file)

                idx += 1
                pbar.update(1)
                batch = next(prefetcher)

            print(f"Epoch time: {time.time() - start_time:.2f}s")
            pbar.close()


async def run(config: FeatureExtractionConfig):
    launcher = Launcher(
        cluster=config.launcher.cluster,
        partition=config.launcher.partition,
        max_jobarray_jobs=config.launcher.max_jobarray_jobs,
    )
    module = FeatureExtractionModule(config)
    await launcher.schedule(module)


cs = ConfigStore.instance()
cs.store(name="feature_extraction_config", node=FeatureExtractionConfig)


@hydra.main(version_base=None, config_name="feature_extraction_config")
def main(config: DictConfig):
    OmegaConf.resolve(config)

    print(f"Config:\n{'-'*50}\n{OmegaConf.to_yaml(config)}{'-'*50}")

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
