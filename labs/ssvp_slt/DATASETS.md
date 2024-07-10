## Data Preparation

### Download

#### DailyMoth-70h

Use our download script under [`scripts/download_dm70.sh`](scripts/download_dm70.sh) to download the full DailyMoth-70h dataset. 

The DailyMoth-70h dataset comes with four folders:
  - **raw_videos**: Contains the unsegmented DailyMoth videos (496 in total) with blurring applied to the burnt-in captions and advertisement breaks and news headline banners
  - **blurred_clips**: Contains the segmented video clips (48386 in total) with facial blurring applied. Each clip comes in its native frame rate (either 24, 29.97 or 30 fps) and as 224x224px region-of-interest (ROI) crops around the signer
  - **unblurred_clips**: Contains the unblurred segmented video clips (48386 in total). Each clip comes in its native frame rate (either 24, 29.97 or 30 fps) and as 224x224px region-of-interest (ROI) crops around the signer
  - **manifests**: Contains the manifest `tsv` files for training, validation, and testing. Also contains a combined manifest file and a `tsv` file with the start and end timestamps used to segment the raw videos

> [!TIP]
> Running the `download_dm70.sh` script will prepare the DailyMoth-70h data in a folder structure ready for pretraining.

#### How2Sign and Youtube-ASL
Download copies of the datasets we used from their original sources:
- **Youtube-ASL**: [https://github.com/google-research/google-research/tree/master/youtube_asl](https://github.com/google-research/google-research/tree/master/youtube_asl).
- **How2Sign**: [https://how2sign.github.io/#download](https://how2sign.github.io/#download). 

> [!IMPORTANT]
> To obtain the correct results with How2Sign, you need to download the full videos "Green Screen RGB videos (frontal view)" and segment them yourself with the timestamps from "English Translation (manually re-aligned)". 
> If you only use "Green Screen RGB clips* (frontal view)", you will use the misaligned data and get **much** worse results.


**Dataset Blurring**

Unfortunately, we cannot open-source our blurring code at this time. As a starting point, [https://github.com/princetonvisualai/imagenet-face-obfuscation/tree/main#face-blurring](https://github.com/princetonvisualai/imagenet-face-obfuscation/tree/main#face-blurring) might work for you.

### Expected Structure

The preprocessed data should follow a structure with three folders:
- `manifests/`, which contains `.tsv` files for the training, validation, and test splits (for YT-ASL, only training). Each file has the columns `["video_name", "duration", "caption"]`, where `video_name` uniquely identifies a video in the dataset, `duration` denotes the video clip duration in seconds, and `caption` is the translation. You can use our [`get_video_durations.py`](scripts/get_video_durations.py) script to obtain these manifest files if you have manifest files in the `["video_name", "start_time", "end_time", "caption"]` format.


- `videos/`, which contains RGB videos, ideally in the form of 224x224px ROIs cropped out around the signer (the video dataloader will random/center crop and resize if necessary). The folder is nested using the first 5 characters of every video as a prefix (this folder structure reduces RPC load when the file system is NFS).

```
BASE_DIR/
└── DATASET_NAME/
  ├── manifests/
    ├── train.tsv
    ├── (val.tsv)
    └── (test.tsv)
  └── videos/
    ├── 00000/
      ├── 0000001-aaaaa.mp4 (example unique video name)
      ├── ...
      ├── 0000099-aaaaa.mp4
    ├── .../
      └── ...
    └── 99999/
      ├── 9999901-aaaaa.mp4
      ├── ...
      └── 9999999-aaaaa.mp4
```
