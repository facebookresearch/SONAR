#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

BASE_URL="https://dl.fbaipublicfiles.com/dailymoth-70h/"
FILES=(
    "raw_videos.tar.gz"
    "blurred_clips.tar.gz"
    "unblurred_clips.tar.gz"
    "manifests.tar.gz"
)
CHECKSUMS=(
    "875ffe4eeac3a37e50b4202c2b4996d2  raw_videos.tar.gz"
    "a2819c7b06a8b38eb7686e4dc90a7433  blurred_clips.tar.gz"
    "3e69046f6cf415cec89c3544d0523325  unblurred_clips.tar.gz"
    "69e500cc5cfad3133c4b589428865472  manifests.tar.gz"
)

echo "Starting download"
for file in "${FILES[@]}"; do
    wget --continue "${BASE_URL}${file}"
done

echo -e "\nVerifying checksums"
for checksum in "${CHECKSUMS[@]}"; do
    echo "$checksum" | md5sum --check
    if [ $? -ne 0 ]; then
        echo "Checksum verification failed for ${checksum##* }"
        exit 1
    fi
done

echo -e "\nExtracting archives"
for file in "${FILES[@]}"; do
    tar -xvf "$file"
done

echo -e "\nCreating symlinks"
prefix="$(pwd)/dailymoth-70h"
ln -s "$prefix"/manifests "$prefix"/unblurred_clips/manifests
ln -s "$prefix"/manifests "$prefix"/blurred_clips/manifests
