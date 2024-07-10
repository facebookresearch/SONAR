# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Union

import torch
from tqdm import tqdm

"""
Simple script to extract the lengths of extracted feature tensors and write them to a new manifest file

Example how to run it for a train.tsv manifest:

python get_feature_lengths.py \
    --manifest_path train.tsv \
    --data_dir extracted_features_dailymoth-70h \
    --num_proc 12
"""


def task(filename: Union[str, os.PathLike]) -> Optional[int]:
    """Try to load features tensor and return their length"""
    try:
        features = torch.load(filename)
        return len(features) if len(features.shape) == 2 else 1
    except FileNotFoundError:
        print(f"{filename} does not exist")
        return None


def main():
    filenames = []
    filepaths = []
    labels = []

    parser = ArgumentParser()
    parser.add_argument(
        "--manifest_path",
        type=str,
        help="Manifest tsv file with columns [video_name, duration, label]",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory with nested structure data_dir -> prefix -> video files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory which the new manifest is written to",
    )
    parser.add_argument(
        "--num_procs",
        type=int,
        default=8,
        help="Number of processes for multiprocessing",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    manifest_file_basename, ext = os.path.splitext(os.path.basename(args.manifest_path))
    new_manifest_file = os.path.join(args.output_dir, f"{manifest_file_basename}_features{ext}")

    num_columns = None
    with open(args.manifest_path) as f:
        for line in f:
            line = line.strip().split("\t")

            filename = line[0]
            filepath = os.path.join(args.data_dir, filename[:5], filename.replace(".mp4", ".pt"))

            label = line[-1]
            if num_columns is None:
                num_columns = len(line)

            elif len(line) != num_columns:
                raise RuntimeError("Number of columns in the manifest file is not consistent.")

            filenames.append(filename)
            filepaths.append(filepath)
            labels.append(label)

    print("Reading feature files")

    with ProcessPoolExecutor(args.num_procs) as executor:
        lengths = list(tqdm(executor.map(task, filepaths, chunksize=64), total=len(filepaths)))

    assert len(labels) == len(filenames) == len(lengths)

    with open(new_manifest_file, "w") as f:
        for filename, length, label in zip(filenames, lengths, labels):
            # If length could not be determined, it is set to -1
            length = length if length is not None else -1
            f.write(f"{filename}\t{length}\t{label}\n")


if __name__ == "__main__":
    main()
