# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Accelerate: https://github.com/huggingface/accelerate
# --------------------------------------------------------

from torch.utils.data import BatchSampler, DataLoader

"""
Utils for skipping the first batches when resuming training mid-epoch
Reference:
https://github.com/huggingface/accelerate/blob/04825483637a002deed91602878efbc1e4cfc0b4/src/accelerate/data_loader.py#L937
"""

# Kwargs of the DataLoader in min version 1.4.0.
_PYTORCH_DATALOADER_KWARGS = {
    "batch_size": 1,
    "shuffle": False,
    "sampler": None,
    "batch_sampler": None,
    "num_workers": 0,
    "collate_fn": None,
    "pin_memory": False,
    "drop_last": False,
    "timeout": 0,
    "worker_init_fn": None,
    "multiprocessing_context": None,
    "generator": None,
    "prefetch_factor": 2,
    "persistent_workers": False,
}


class SkipBatchSampler(BatchSampler):
    """
    A `torch.utils.data.BatchSampler` that skips the first `n` batches of another `torch.utils.data.BatchSampler`.
    """

    def __init__(self, batch_sampler: BatchSampler, skip_batches: int = 0) -> None:
        self.batch_sampler = batch_sampler
        self.skip_batches = skip_batches

    def __iter__(self):
        for index, samples in enumerate(self.batch_sampler):
            if index >= self.skip_batches:
                yield samples

    @property
    def total_length(self) -> int:
        return len(self.batch_sampler)

    def __len__(self) -> int:
        return len(self.batch_sampler) - self.skip_batches


def skip_first_batches(dataloader: DataLoader, num_batches: int = 0) -> DataLoader:
    """
    Creates a `torch.utils.data.DataLoader` that will efficiently skip the first `num_batches`.
    """
    dataset = dataloader.dataset

    sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
    batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
    new_batch_sampler = SkipBatchSampler(batch_sampler, skip_batches=num_batches)

    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
    ]

    kwargs = {
        k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k])
        for k in _PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    dataloader = DataLoader(dataset, batch_sampler=new_batch_sampler, **kwargs)

    return dataloader
