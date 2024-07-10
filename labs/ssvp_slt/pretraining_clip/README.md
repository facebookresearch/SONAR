## MAE & CLIP/FLIP Pretraining

Ensure your environment is set up following our [INSTALL.md](../INSTALL.md) and your data is prepared following our [DATASETS.md](../DATASETS.md).
You can then launch MAE & CLIP/FLIP pretraining in our default configuration as shown below:

```bash
python run_pretraining_clip.py \
    --config-name pretraining_clip \
    --multirun \
    run=64gpu \
    hydra.sweep.dir=/path/to/experiment/folder \
    optim.batch_size=8 \
    data.base_data_dir=/path/to/base/data/dir \
    data.dataset_names=youtube-asl,how2sign
```
To run only CLIP/FLIP pretraining without MAE, you can pass `model.no_mae=true`.

The example here is to run on A100 GPUs. Depending on your setup, you can tweak the batch size and run-config. For CLIP/FLIP training, we currently do not support gradient accumulation, but we encourage adding it (note that gradient accumulation is not immediately compatible with the CLIP contrastive loss where the loss is computed over all parallel workers but not across batches). When adjusting the batch size without gradient accumulation, make sure to also adjust the learning rate accordingly, e.g. following the linear scaling rule.

You can find all default and configurable parameters in [../configs/pretraining_clip.yaml](../configs/pretraining_clip.yaml) and in the hydra dataclasses in [run_pretraining_clip.py](run_pretraining_clip.py).
