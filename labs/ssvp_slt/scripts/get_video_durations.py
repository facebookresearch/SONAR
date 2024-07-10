# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import av
import numpy as np
import pandas as pd

"""
Simple script to extract the durations of segmented video clips and write them to a new manifest file

Example how to run it for a train.tsv manifest:

python get_feature_lengths.py --manifest_path manifest_train.tsv --output_path train.tsv
"""


def format_videoname(row):
    return f"{row['index'] + 1:07d}-{row['video_name']}.mp4"


def calculate_duration(row, data_dir: Path):
    """
    Reads video duration from file
    We don't use the start and end timestamps in the manifest as they might be less accurate
    """
    video_path = str(data_dir / row["video_name"][:5] / row["video_name"])
    container = av.open(video_path)
    duration = container.duration / av.time_base
    return f"{duration:.3f}"


def parallelize(data, func, num_procs=8):
    data_split = np.array_split(data, num_procs)
    pool = Pool(num_procs)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def run_on_subset(func, data_subset, **kwargs):
    return data_subset.apply(func, axis=1, **kwargs)


def parallelize_on_rows(data, func, num_procs=8, **kwargs):
    return parallelize(data, partial(run_on_subset, func, **kwargs), num_procs)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--manifest_path",
        type=str,
        help="Manifest tsv file with columns [video_name, start, end, caption]",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory with nested structure data_dir -> prefix -> video files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path for output manifest",
    )
    parser.add_argument(
        "--num_procs",
        type=int,
        default=8,
        help="Number of processes for multiprocessing",
    )
    args = parser.parse_args()

    df = pd.read_csv(
        args.manifest_path,
        sep="\t",
        quoting=3,
    ).reset_index()

    df["video_name"] = df.apply(format_videoname, axis=1)
    df["duration"] = parallelize_on_rows(
        df, calculate_duration, num_procs=args.num_procs, data_dir=Path(args.data_dir)
    )

    df.to_csv(
        args.output_path,
        sep="\t",
        index=False,
        quoting=3,
        columns=["video_name", "duration", "caption"],
    )


if __name__ == "__main__":
    main()
