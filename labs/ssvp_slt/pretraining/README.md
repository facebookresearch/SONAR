## MAE Pretraining

Ensure your environment is set up following our [INSTALL.md](../INSTALL.md) and your data is prepared following our [DATASETS.md](../DATASETS.md).
You can then launch MAE pretraining in our default configuration as shown below:

```bash
python run_pretraining.py \
    --config-name pretraining \
    --multirun \
    run=64gpu \
    hydra.sweep.dir=/path/to/experiment/folder \
    optim.batch_size=2 \
    data.repeat_aug=2 \
    data.base_data_dir=/path/to/base/data/dir \
    data.dataset_names=youtube-asl,how2sign
```

The example here is to run in fp32 on A100 GPUs (64 x 2 x 2 = 256 effective batch size). You should be able to fit larger batches on equivalent GPUs; we do not use larger batches primarily due to dataloader bottlenecking. We did not use fp16 amp due to numerical instabilities, but bfloat16 could work well as an alternative to fp32.

Depending on your setup, you can tweak the batch size and run config. You can also add gradient accumulation to reach the same effective batch size on smaller GPUs.

You can find all default and configurable parameters in [../configs/pretraining.yaml](../configs/pretraining.yaml) and in the hydra dataclasses in [run_pretraining.py](run_pretraining.py).
