# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# References:
#
# MAE_ST: https://github.com/facebookresearch/mae_st
# Slowfast: https://github.com/facebookresearch/SlowFast
# --------------------------------------------------------
import math
import os
import warnings
from itertools import takewhile
from typing import Dict, Optional, Tuple, Union

import torch
import torchvision
from einops import rearrange
from iopath.common.file_io import g_pathmgr as pathmgr
from omegaconf import OmegaConf
from torchvision import set_video_backend
from torchvision.io._load_gpu_decoder import _HAS_GPU_VIDEO_DECODER
from torchvision.transforms.v2._auto_augment import RandAugment as RandAug

from ssvp_slt.util.video import (get_num_padding_frames, get_start_end_idx,
                                 horizontal_flip, random_crop,
                                 temporal_sampling, tensor_normalize,
                                 uniform_crop)

SUPPORTED_VIDEO_BACKENDS = [
    "video_reader",
    "cuda",
    "pyav",
]  # TODO: support decoding from memory


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mode: str,
        data_dir: Union[str, os.PathLike],
        video_backend: str = "pyav",
        sampling_rate: int = 2,
        num_frames: int = 128,
        target_fps: int = 25,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        # Augmentation
        rand_aug: bool = False,
        repeat_aug: int = 1,
        train_random_crop: bool = True,
        train_crop_size: int = 224,
        train_random_horizontal_flip: bool = True,
        # Normalization
        do_normalize: bool = True,
        mean: Tuple[float, float, float] = (0.45, 0.45, 0.45),
        std: Tuple[float, float, float] = (0.225, 0.225, 0.225),
        # Finetuning
        return_labels: bool = False,
        # Feature extraction
        feature_extraction: bool = False,
        feature_extraction_stride: Optional[int] = None,
        # GPU decoding,
        gpu: Optional[Union[int, torch.device]] = None,
        # Debugging
        max_num_samples: Optional[int] = None,
        indices: Tuple[Optional[int], Optional[int]] = (None, None),
    ):
        if video_backend == "cuda":
            if not _HAS_GPU_VIDEO_DECODER:
                warnings.warn(
                    f"`{video_backend}` backend is not available. Using default `pyav` backend instead."
                )
                self.video_backend = "pyav"
            assert gpu is not None, "`gpu` must be set when using GPU backend for video decoding."
        else:
            assert (
                video_backend in SUPPORTED_VIDEO_BACKENDS
            ), f"Invalid video backend. Supported backends are {SUPPORTED_VIDEO_BACKENDS}"
            print(f"Using video_backend `{video_backend}`")
            self.video_backend = video_backend

        self.mode = mode
        self.data_dir = data_dir
        self.gpu = gpu

        self.max_num_samples = max_num_samples
        self.start_idx, self.end_idx = indices
        self.return_labels = return_labels

        self.min_duration = min_duration or 0
        self.max_duration = max_duration or math.inf

        self.sampling_rate = sampling_rate
        self.num_frames = num_frames
        self.target_fps = target_fps

        self.rand_aug = rand_aug
        self.aug = RandAug(num_ops=4, magnitude=7)
        self.repeat_aug = repeat_aug
        self.train_crop_size = train_crop_size
        self.train_random_crop = train_random_crop
        self.train_random_horizontal_flip = train_random_horizontal_flip

        self.do_normalize = do_normalize
        self.mean = mean
        self.std = std

        self.feature_extraction = feature_extraction
        self.feature_extraction_stride = feature_extraction_stride
        if self.feature_extraction:
            assert (
                self.feature_extraction_stride is not None
            ), "Using dataset for feature extraction requires setting a feature_extraction_stride"
            self.repeat_aug = 1
            self.max_duration = math.inf

        print(
            "Dataset Config:\n"
            f"{OmegaConf.to_yaml(OmegaConf.create({k: v for k, v in locals().items() if k != 'self'}))}"
        )

        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        tsv_file_name = {
            "pretrain": "train",
            "finetune": "train",
            "train": "train",
            "val": "val",
            "test": "test",
        }
        path_to_manifest = os.path.join(
            self.data_dir,
            "manifests",
            f"{tsv_file_name[self.mode]}.tsv",
        )
        path_to_videos = os.path.join(self.data_dir, "videos")

        assert pathmgr.exists(path_to_manifest), f"{path_to_manifest} not found"
        assert pathmgr.exists(path_to_videos), f"{path_to_videos} not found"

        self.path_to_videos = []
        self.labels = []

        invalid = 0
        duration_filtered = 0

        with pathmgr.open(path_to_manifest, "r") as f:
            for video_idx, line in enumerate(f):
                if self.start_idx and video_idx < self.start_idx:
                    continue
                if self.end_idx and video_idx >= self.end_idx:
                    break

                try:
                    video_name, duration, label = line.strip().split("\t")
                    duration = float(duration)
                except Exception:
                    invalid += 1
                    continue

                if self.mode in ["pretrain", "finetune", "train"] and duration < self.min_duration:
                    duration_filtered += 1
                    continue

                prefix = video_name[:5]
                video_filepath = os.path.join(path_to_videos, prefix, video_name)

                self.path_to_videos.append(video_filepath)
                self.labels.append(label)

        if self.max_num_samples is not None and self.max_num_samples < len(self.path_to_videos):
            self.path_to_videos = self.path_to_videos[: self.max_num_samples]
            self.labels = self.labels[: self.max_num_samples]

        print(f"Number of invalid videos skipped: {invalid}")
        print(f"Number of videos filtered due to duration: {duration_filtered}")

        assert (
            len(self.path_to_videos) > 0
        ), f"Failed to load VideoDataset split {self.mode} from {path_to_videos}"

        print(f"Constructing dataloader (size: {len(self.path_to_videos)}) from {path_to_videos}")

    def __getitem__(self, index):
        if self.video_backend == "cuda":
            torch.cuda.set_device(self.gpu)
        set_video_backend(self.video_backend)

        reader = torchvision.io.VideoReader(self.path_to_videos[index])

        fps = reader.get_metadata()["video"]["fps"][0]

        frames = [
            frame["data"]
            for frame in takewhile(lambda frame: frame["pts"] < self.max_duration, reader)
        ]
        frames = torch.stack(frames, dim=0)

        if self.video_backend in {"video_reader", "pyav"}:
            # T C H W -> T H W C
            frames = frames.permute(0, 2, 3, 1)

        if self.feature_extraction:
            return self.sample_frames_for_feature_extraction(frames, fps, index)

        return self.sample_frames(frames, fps, index)

    @torch.no_grad()
    def sample_frames(
        self, frames: torch.Tensor, fps: float, index: int
    ) -> Dict[str, Union[int, torch.Tensor]]:
        """
        Samples `repeat_aug` fixed-size clips from the video at index `index`.
        If the video is shorter than `num_frames`, it will be padded. The method returns the number of
        padding frames.
        """
        frames_list = []
        num_padding_frames_list = []

        crop_size = self.train_crop_size
        sampling_rate = self.sampling_rate
        num_frames = self.num_frames

        # T H W C -> C T H W.
        for i in range(self.repeat_aug):
            clip_sz = sampling_rate * num_frames / self.target_fps * fps
            start_idx, end_idx = get_start_end_idx(
                frames.shape[0],
                clip_sz,
                -1,  # indicates random sampling
                self.repeat_aug,
                use_offset=True,
            )

            new_frames, idx = temporal_sampling(
                frames, start_idx, end_idx, num_frames, return_index=True
            )
            num_padding_frames_list.append(
                get_num_padding_frames(idx, num_frames, sampling_rate, fps, self.target_fps)
            )

            if self.rand_aug:
                new_frames = new_frames.permute(0, 3, 1, 2)
                with torch.no_grad():
                    new_frames = self.aug(new_frames)
                new_frames = new_frames.permute(0, 2, 3, 1)

            if self.do_normalize:
                new_frames = tensor_normalize(new_frames, self.mean, self.std)
            else:
                if new_frames.dtype == torch.uint8:
                    new_frames = new_frames.float()
                    new_frames /= 255.0
            new_frames = new_frames.permute(3, 0, 1, 2)

            # Spatial sampling
            new_frames = (
                random_crop(new_frames, crop_size)
                if self.train_random_crop
                else uniform_crop(
                    new_frames, crop_size, spatial_idx=1
                )  # spatial_idx=1 indicates center crop
            )
            if self.train_random_horizontal_flip:
                new_frames = horizontal_flip(0.5, new_frames)

            frames_list.append(new_frames)

        frames = torch.stack(frames_list, dim=0)
        num_padding_frames = torch.tensor(
            num_padding_frames_list, dtype=torch.long, device=frames.device
        )

        res = {
            "frames": frames,
            "padding": num_padding_frames,
            "index": index,
        }
        if self.return_labels:
            res["labels"] = self.labels[index]
        return res

    @torch.no_grad()
    def sample_frames_for_feature_extraction(
        self, frames: torch.Tensor, fps: float, index: int
    ) -> Dict[str, Union[int, torch.Tensor]]:
        """
        Samples clips with a fixed number of frames in a sliding window for the full video.
        The clips are then stacked in the batch dimension. If the video is shorter than `num_frames`,
        it will be padded. The method returns the number of padding frames.
        This method does not support repeat augmentation.
        """

        frames_list = []
        num_padding_frames_list = []

        stride = self.feature_extraction_stride
        crop_size = self.train_crop_size
        sampling_rate = self.sampling_rate
        num_frames = self.num_frames
        clip_sz = sampling_rate * num_frames / self.target_fps * fps

        for i in range(0, frames.shape[0], stride * sampling_rate):
            start_idx, end_idx = i, i + clip_sz - 1
            new_frames, idx = temporal_sampling(
                frames, start_idx, end_idx, num_frames, return_index=True
            )

            num_padding_frames_list.append(
                get_num_padding_frames(idx, num_frames, sampling_rate, fps, self.target_fps)
            )

            frames_list.append(new_frames)

            if end_idx >= frames.shape[0]:
                break

        new_frames = torch.concatenate(frames_list, dim=0)

        # Perform augmentation on concatenated frames so it is consistent across all clips
        if self.rand_aug:
            new_frames = new_frames.permute(0, 3, 1, 2)
            new_frames = self.aug(new_frames)
            new_frames = new_frames.permute(0, 2, 3, 1)

        if self.do_normalize:
            new_frames = tensor_normalize(new_frames, self.mean, self.std)
        else:
            if new_frames.dtype == torch.uint8:
                new_frames = new_frames.float()
                new_frames /= 255.0
        new_frames = new_frames.permute(3, 0, 1, 2)

        # Spatial sampling
        new_frames = (
            random_crop(new_frames, crop_size)
            if self.train_random_crop
            else uniform_crop(
                new_frames, crop_size, spatial_idx=1
            )  # spatial_idx=1 indicates center crop
        )
        if self.train_random_horizontal_flip:
            new_frames = horizontal_flip(0.5, new_frames)

        frames = rearrange(new_frames, "c (b t) h w -> b c t h w", t=num_frames)

        num_padding_frames = torch.tensor(
            num_padding_frames_list, dtype=torch.long, device=frames.device
        )

        res = {
            "frames": frames,
            "padding": num_padding_frames,
            "index": index,
        }
        if self.return_labels:
            res["labels"] = self.labels[index]
        return res

    def __len__(self):
        return len(self.path_to_videos)
