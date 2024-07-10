## Configs

This folder contains Hydra config files in `yaml` format to configure pretraining and translation experiments.

> [!TIP]
> If you are not yet familiar with Hydra, check out the official [documentation](https://hydra.cc/docs/intro/).


> [!TIP]
> The training configurations here are our preferred default values. However, feel free to tweak them as you prefer. There are many more parameters to configure, and you will find them in the respective python run script, e.g. [`run_translation.py`](../translation/run_translation.py).


> [!IMPORTANT]  
> Before running experiments on SLURM, make sure to adjust the SLURM submitit launcher config in the `yaml` files under [run/](run/) according to your cluster settings.