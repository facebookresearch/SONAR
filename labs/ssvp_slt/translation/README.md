## Translation

### Feature Extraction
Once you have obtained a pretrained model, and your data is prepared as outlined in [DATASETS.md](../DATASETS.md), you are ready for feature extraction. Below is an example for how it can be run.

 `num_items_per_shard` is tuned based on the size of our dataset (fewer items per shard result in larger slurm array).
 `max_batch_size=2` is based on our setup using fp32 and 32GB V100 GPUs. When using a CLIP checkpoint with SignHiera vision tower, pass `from_clip=true`. If you are not using a SLURM cluster, you can pass `launcher.cluster=local` to run feature extraction on the local machine.

For additional parameters, please refer to the `FeatureExtractionConfig` dataclass in [launch_feature_extraction.py](launch_feature_extraction.py).

```bash
python launch_feature_extraction.py \
  data_dir=/path/to/how2sign \
  output_dir=/path/to/empty/folder/for/features \
  from_clip=true \
  video_backend=pyav \
  model_name=hiera_base_128x224 \
  pretrained_model_path=/path/to/pretrained/model/checkpoint.pth \
  epochs=1 \
  split=test \
  num_items_per_shard=50 \
  num_frames=128 \
  sampling_rate=2 \
  target_fps=25 \
  max_batch_size=2 \
  launcher.cluster=slurm
```

After extracting the features, you should make sure they follow the nested folder structure expected by our [`SignFeaturesDataset`](../src/ssvp_slt/data/sign_features_dataset.py) below. The manifest `.tsv` files should have the following columns: `["video_name", "length", "label"]`, where `video_name` uniquely identifies a video in the dataset, `length` denotes the length of the features tensor, and `label` is the translation. You can obtain manifest files with the length column populated based on the extracted features via our provided [get_feature_lengths.py](../scripts/get_feature_lengths.py) script.
```
├── DATASET_NAME/
    ├── manifests/
        ├── train_features.tsv
        ├── val_features.tsv
        └── test_features.tsv
    └── features/
        ├── 0 (epoch)
            ├── 00000 (prefix)
                ├── 00000-abcdef.pt (video_name.pt)
                └── ...
            ├── 00001 
                ├── 00001-cdefgh.pt
                └── ...  
        └── ...  (if extracted more than 1 epoch)
```

### Training
After you have successfully extracted the features, you can simply run the translation pipeline as follows.

For T5:
```bash
# sweep over 5 random seeds
python run_translation.py \
    --config-name slt_t5 \
    --multirun \
    run=64gpu \
    hydra.sweep.dir=sweep_t5 \
    'data.train_data_dirs="/path/to/features/how2sign,/path/to/features/youtube-asl"'  \
    data.val_data_dir=/path/to/features/how2sign \
    optim.train_batch_size=4 \
    optim.gradient_accumulation_steps=1 \
    common.eval_steps=500 \
    common.seed=1,2,3,4,5 \
    wandb.project=t5-sweep
```

For Tarrés et al. (2023):

```bash
# sweep over 5 random seeds
python run_translation.py \
    --config-name slt_tarres_fairseq \
    --multirun \
    run=1gpu \
    hydra.sweep.dir=sweep_tf \
    model.name_or_path=/path/to/folder/w/tarres-tokenizer \
    data.train_data_dirs=/path/to/features/how2sign \
    data.val_data_dir=/path/to/features/how2sign \
    common.seed=1,2,3,4,5 \
    wandb.project=tf-sweep \
    common.fp16=true
```


Adjust the number of GPUs and gradient accumulation steps based on your setup. These setups are based on fp32 training on 32GB V100 GPUs. You can also play with the max sequence length (`data.max_source_positions`) to reduce memory footprint. While we recommend `common.fp16=true` for BART training and the `slt_tarres_fairseq` configuration, `fp16` does not work with T5. We have not tried `bfloat16` training but encourage adding support for it.

For all additional training and evaluation parameters, refer to the dataclasses in [run_translation.py](run_translation.py).

### Evaluation
You can run the translation pipeline in eval mode to only perform evaluation on the validation and test sets.
To do this, simply pass the appropriate overrides, e.g.:

```bash
python run_translation.py \
    --config-name slt_t5 \
    hydra.sweep.dir=sweep_eval_t5 \
    'data.train_data_dirs="/path/to/features/how2sign,/path/to/features/youtube-asl"'  \
    data.val_data_dir=/path/to/features/how2sign \
    common.eval=true \
    common.load_model=/path/to/finetuned-t5/best_model.pth
```


