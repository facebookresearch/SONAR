## Scripts

In this folder you will find some useful scripts.

#### 1. `download_dm70.sh`

This script downloads and prepares the DailyMoth-70h dataset in the current working directory.
You can simply execute it via `./download_dm70.sh`

#### 2. `get_video_durations.py`

This script can be used to create a manifest compatible with our [`VideoDataset`](../src/ssvp_slt/data/video_dataset.py). It reads the video durations directly from the video files and writes them to a new manifest. The input manifest needs to be tab-separated with the video name in the first column and the caption/translation in the last column.

Example usage:

```bash
get_video_durations.py \
  --manifest_path dailymoth-70h/manifests/srt_data.tsv \
  --data_dir dailymoth-70h/unblurred/videos \
  --output_path dailymoth-70h/manifests/manifest.tsv \
  --num_procs 12
```

#### 3. `get_feature_lengths.py`

This script can be used to create a manifest compatible with our [`SignFeaturesDataset`](../src/ssvp_slt/data/sign_features_dataset.py). It reads the feature lengths directly from the extracted video features writes them to a new manifest. The input manifest needs to be tab-separated and have the columns `[video_name, duration, caption]` (i.e., the output format of `get_video_lengths.py`).

Example usage:

```bash
get_video_durations.py \
  --manifest_path dailymoth-70h/manifests/srt_data.tsv \
  --data_dir dailymoth-70h/unblurred/videos \
  --output_path dailymoth-70h/manifests/manifest.tsv \
  --num_procs 12
```

#### 4. `train_spm.py`

This script can be used to train a sentencepiece model for tokenization.

Here is an example for training a tokenizer with a vocabulary size of 7000 on the DailyMoth-70h data:

```bash
python train_spm.py \
  --tsv_file dailymoth-70h/manifests/manifest.tsv \
  --vocab_size 7000 \
  --output_dir spm7000
```

To train a lowercase tokenizer (as used for How2Sign), additionally pass `--lowercase`.