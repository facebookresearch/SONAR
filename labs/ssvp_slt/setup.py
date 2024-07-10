# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from setuptools import find_packages, setup

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
assert TORCH_VERSION >= (2, 0), "Requires PyTorch >= 2.0"

setup(
    name="ssvp-slt",
    version="0.0.1",
    author="Phillip Rust",
    description="Self-supervised video pretraining for sign language translation",
    python_requires=">=3.8",
    install_requires=[
        "einops",
        "evaluate",
        "hydra-ax-sweeper",
        "hydra-core",
        "hydra-submitit-launcher",
        "iopath",
        "matplotlib",
        "numpy",
        "omegaconf",
        "opencv-python-headless",
        "pandas",
        "psutil",
        "requests",
        "rouge_score",
        "sacrebleu",
        "scikit-learn",
        "sentencepiece",
        "setuptools",
        "simplejson",
        "stopes",
        "submitit",
        "tensorboard",
        "tensorflow",
        "timm",
        "torch>=2.0.0",
        "torchvision>=0.15.1",
        "tqdm",
        "transformers>=4.32.0",
        "truecase",
        "urllib3",
        "wandb",
    ],
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=True,
    license="CC BY-NC 4.0",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
