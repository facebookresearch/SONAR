# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE-ST: https://github.com/facebookresearch/mae_st
# --------------------------------------------------------

import math
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch


def get_num_padding_frames(
    idx: torch.Tensor,
    num_frames: int,
    sampling_rate: int,
    fps: float,
    target_fps: float,
) -> int:
    """
    Get the number of padding frames based on the provided parameters

    Args:
    idx (torch.Tensor): A tensor containing indices.
    num_frames (int): The total number of frames.
    sampling_rate (int): The rate at which frames are sampled.
    fps (float): The original frames per second.
    target_fps (float): The target frames per second.

    Returns:
    int: The number of padding frames.
    """

    num_unique = len(torch.unique(idx))

    # Frames duplicated via interpolation should not count as padding
    if target_fps > (fps * sampling_rate):
        num_non_padding = math.floor(num_unique * target_fps / (fps * sampling_rate))
    else:
        num_non_padding = num_unique
    return num_frames - num_non_padding


def get_start_end_idx(
    video_size: int,
    clip_size: int,
    clip_idx: int,
    num_clips: int,
    use_offset: bool = False,
) -> Tuple[int, int]:
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """

    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips == 1:
                # Take the center clip if num_clips is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(delta / (num_clips - 1))
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1

    return start_idx, end_idx


def temporal_sampling(
    frames: torch.Tensor,
    start_idx: int,
    end_idx: int,
    num_samples: int,
    return_index: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx + 0.5, end_idx + 0.5, num_samples, device=frames.device)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    new_frames = torch.index_select(frames, 0, index)

    if return_index:
        return new_frames, index
    return new_frames


def tensor_normalize(
    tensor: torch.Tensor,
    mean: Union[torch.Tensor, Tuple[float, float, float]],
    std: Union[torch.Tensor, Tuple[float, float, float]],
) -> torch.Tensor:
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if isinstance(mean, tuple):
        mean = torch.tensor(mean, device=tensor.device)
    if isinstance(std, tuple):
        std = torch.tensor(std, device=tensor.device)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def horizontal_flip(prob: float, images: torch.Tensor) -> torch.Tensor:
    """
    Perform horizontal flip on the given images.
    Args:
        prob (float): probility to flip the images.
        images (tensor): images to perform horizontal flip, the dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): images with dimension of
            `num frames` x `channel` x `height` x `width`.
    """
    if np.random.uniform() < prob:
        images = images.flip((-1))
    return images


def random_crop(images: torch.Tensor, size: int) -> torch.Tensor:
    """
    Perform random spatial crop on the given images.
    Args:
        images (tensor): images to perform random crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): the size of height and width to crop on the image.
    Returns:
        cropped (tensor): cropped images with dimension of
            `num frames` x `channel` x `size` x `size`.
    """
    if images.shape[2] == size and images.shape[3] == size:
        return images
    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    return cropped


def uniform_crop(
    images: torch.Tensor, size: int, spatial_idx: int, scale_size: Optional[int] = None
) -> torch.Tensor:
    """
    Perform uniform spatial sampling on the images.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped
