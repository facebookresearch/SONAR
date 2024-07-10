# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os
import random
from typing import Dict, Optional, Tuple, Union

import torch
from iopath.common.file_io import g_pathmgr as pathmgr
from omegaconf import OmegaConf


class SignFeaturesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mode: str,
        path_to_data_dir: Union[str, os.PathLike],
        min_seq_length: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        num_retries: int = 10,
        indices: Tuple[Optional[int], Optional[int]] = (None, None),
    ):
        assert mode in [
            "train",
            "val",
            "test",
        ], f"Split `{mode}` not supported"

        self.mode = mode
        self.path_to_data_dir = path_to_data_dir
        self.start_idx, self.end_idx = indices

        self.min_seq_length = min_seq_length or 0
        self.max_seq_length = max_seq_length or 1e20

        self.num_retries = num_retries

        print(
            "Dataset Config:\n"
            f"{OmegaConf.to_yaml(OmegaConf.create({k: v for k, v in locals().items() if k != 'self'}))}"
        )

        self._construct_loader()

        self.set_epoch(0)

    def _construct_loader(self):
        """
        Construct the feature loader.
        """
        path_to_manifest = os.path.join(
            self.path_to_data_dir, "manifests", f"{self.mode}_features.tsv"
        )
        path_to_features = os.path.join(self.path_to_data_dir, "features")

        assert pathmgr.exists(path_to_manifest), f"{path_to_manifest} not found"
        assert pathmgr.exists(path_to_features), f"{path_to_features} not found"

        self.path_to_features = []
        self.labels = []
        self.sizes = []

        invalid = 0
        truncated = 0
        with pathmgr.open(path_to_manifest, "r") as f:
            for feature_idx, line in enumerate(f):
                if self.start_idx and feature_idx < self.start_idx:
                    continue
                if self.end_idx and feature_idx >= self.end_idx:
                    break
                try:
                    feature_name, length, label = line.strip().split("\t")
                    length = int(length)
                except Exception:
                    invalid += 1
                    continue

                prefix = feature_name[:5]
                feature_name = f"{os.path.splitext(feature_name)[0]}.pt"

                feature_filepath = os.path.join(path_to_features, "{epoch}", prefix, feature_name)

                if length > self.max_seq_length:
                    truncated += 1

                if self.mode == "train" and length < self.min_seq_length:
                    invalid += 1
                    continue

                self.path_to_features.append(feature_filepath)
                self.labels.append(label)
                self.sizes.append(length)

        print(f"Number of invalid features that are skipped: {invalid}")
        print(f"Number of features that are too long and will be truncated: {truncated}")

        assert (
            len(self.path_to_features) > 0
        ), f"Failed to load dataset split {self.mode} from {self.path_to_data_dir}"

        print(
            f"Constructing dataloader (size: {len(self.path_to_features)}) from {self.path_to_data_dir}"
        )

    def set_epoch(self, epoch: int) -> None:
        print(f"Setting epoch {epoch}")
        self.epoch = epoch

    def __getitem__(self, index: int) -> Dict[str, Union[int, torch.tensor, torch.tensor]]:
        """
        Loads and returns a feature array stored as `torch.Tensor` on disk

        Args:
            index (int): the feature index provided by the pytorch sampler.
        Returns:
            id (int): the feature index in the dataset
            feature_id (int): the unique name of the features
            source (tensor): the extracted video features with dimensions (N, D)
            label (tensor): the label string
        """
        feature_id = os.path.splitext(os.path.basename(self.path_to_features[index]))[0]

        for i_try in range(self.num_retries):
            features = None
            try:
                # TODO: Use memory mapping for faster loading
                features = torch.load(self.path_to_features[index].format(epoch=self.epoch))
            except Exception:
                pass
            # Select random features if the current features were unavailable
            if features is None:
                print(
                    f"Failed to meta load features idx {index} from {self.path_to_features[index].format(epoch=self.epoch)}; trial {i_try}"
                )
                if self.mode not in ["test"]:
                    # let's try another one
                    index = random.randint(0, len(self.path_to_features) - 1)
                    print(f"Randomly selected index {index}")
                continue

            # Truncate if necessary
            if len(features) > self.max_seq_length:
                features = features[: self.max_seq_length]

            label = self.labels[index]

            return {
                "id": index,
                "feature_id": feature_id,
                "source": features,
                "label": label,
            }
        else:
            raise RuntimeError(f"Failed to fetch features after {self.num_retries} retries.")

    def get_label(self, index):
        return self.labels[index]

    @property
    def num_features(self) -> int:
        """
        Returns:
            (int): the number of features tensors in the dataset.
        """
        return len(self.path_to_features)

    def __len__(self) -> int:
        """
        Returns:
            (int): the number of features tensors in the dataset.
        """
        return self.num_features
