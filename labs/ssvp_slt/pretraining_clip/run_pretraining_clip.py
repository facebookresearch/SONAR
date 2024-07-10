# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING, DictConfig, OmegaConf

from ssvp_slt.util.misc import reformat_logger

from main_pretraining_clip import main as pretrain

logger = logging.getLogger(__name__)


@dataclass
class CommonConfig:
    output_dir: str = os.path.join(II("hydra:sweep.dir"), II("hydra:sweep.subdir"))
    log_dir: str = os.path.join(II("hydra:sweep.dir"), II("hydra:sweep.subdir"), "logs")
    resume: Optional[str] = field(
        default=None, metadata={"help": "Path to training checkpoint to resume from"}
    )
    load_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model weights to load at the start of the run"},
    )
    seed: int = field(default=42, metadata={"help": "Random seed"})
    device: str = field(default="cuda", metadata={"help": "Device to train and evaluate on"})
    fp16: bool = field(default=False, metadata={"help": "Use fp16 automatic mixed-precision"})
    pin_mem: bool = field(default=True, metadata={"help": "Pin memory"})
    num_workers: int = field(default=10, metadata={"help": "Num dataloader workers"})
    persistent_workers: bool = field(
        default=True, metadata={"help": "Use persistent workers for dataloading"}
    )
    max_checkpoints: Optional[int] = field(
        default=None, metadata={"help": "Only keep the last N checkpoints"}
    )
    print_steps: int = field(
        default=10, metadata={"help": "Print training progress every n steps"}
    )
    logging_steps: int = field(default=50, metadata={"help": "Log progress every n steps"})
    save_steps: int = field(
        default=500, metadata={"help": "Save a training checkpoint every n steps"}
    )


@dataclass
class ModelConfig:
    vision_model_name: str = field(
        default="mae_hiera_base_128x224",
        metadata={"help": "Name of model to train as vision tower"},
    )
    decoder_depth: int = field(default=8, metadata={"help": "Number of layers in the MAE decoder"})
    decoder_num_heads: int = field(
        default=8, metadata={"help": "Number of attention heads in the MAE decoder"}
    )
    drop_path_rate: float = field(default=0.2, metadata={"help": "Drop path rate (default: 0.2)"})
    mask_ratio: float = field(
        default=0.9, metadata={"help": "Masking ratio (percentage of removed patches)."}
    )
    norm_pix_loss: bool = field(
        default=True,
        metadata={"help": "Use (per-patch) normalized pixels as targets for computing loss"},
    )
    vision_model_proj: Optional[str] = field(
        default="mlp",
        metadata={
            "help": "Projection module for the CLIP vision tower, can be 'mlp', 'linear', or None"
        },
    )
    text_model_name_or_path: str = field(
        default="google/t5-v1_1-base",
        metadata={"help": "HF identifier or path to HF text encoder model to use as text tower"},
    )
    text_model_proj: Optional[str] = field(
        default="mlp",
        metadata={
            "help": "Projection module for the CLIP text tower, can be 'mlp', 'linear', or None"
        },
    )
    text_model_pooler: str = field(
        default="mean_pooler",
        metadata={
            "help": "Pooler for the CLIP text tower. Can be 'mean_pooler', 'max_pooler', 'cls_pooler', or "
            "'cls_last_hidden_state_pooler'"
        },
    )
    embed_dim: int = field(default=768, metadata={"help": "Embed dim for CLIP model"})
    gradnorm_alpha: float = field(default=1.0, metadata={"help": "GradNorm alpha hyperparameter"})
    no_mae: bool = field(
        default=False, metadata={"help": "Disable MAE and only run CLIP/FLIP training"}
    )


@dataclass
class DataConfig:
    base_data_dir: str = field(
        default=MISSING, metadata={"help": "Path to train data directories"}
    )
    dataset_names: str = field(
        default=MISSING, metadata={"help": "Names of datasets to pretrain on"}
    )
    num_frames: int = field(
        default=256, metadata={"help": "Number of frames to sample per video clip"}
    )
    sampling_rate: int = field(default=2, metadata={"help": "Take every n-th video frame"})
    target_fps: int = field(default=25, metadata={"help": "Target framerate in frames per second"})
    video_backend: str = field(
        default="pyav",
        metadata={"help": "Backend to use for video decoding. Options are 'cuda' or 'pyav"},
    )
    rand_aug: bool = field(default=True, metadata={"help": "Perform RandAug data augmentation"})
    num_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of training samples to use. Useful for debugging purposes."},
    )
    min_duration: Optional[float] = field(
        default=None,
        metadata={"help": "Minimum video duration in seconds. Shorter videos will be discarded"},
    )
    max_duration: Optional[float] = field(
        default=None, metadata={"help": "Maximum video duration in seconds."}
    )


@dataclass
class OptimizationConfig:
    lr: float = field(default=8e-4, metadata={"help": "Learning rate"})
    min_lr: float = field(default=1e-6, metadata={"help": "Minimum learning rate after decay"})
    weight_decay: float = field(default=0.05, metadata={"help": "Adam weight decay"})
    start_epoch: int = field(default=0, metadata={"help": "Start epoch"})
    epochs: int = field(default=400, metadata={"help": "Number of training epochs"})
    warmup_epochs: int = field(
        default=120, metadata={"help": "Number of epochs to warm up learning rate for"}
    )
    batch_size: int = field(default=32, metadata={"help": "Batch size for training"})
    clip_grad: Optional[float] = field(
        default=None, metadata={"help": "Gradient norm clipping threshold"}
    )
    adam_beta1: float = field(default=0.9, metadata={"help": "Adam beta1 hyperparameter"})
    adam_beta2: float = field(default=0.95, metadata={"help": "Adam beta2 hyperparameter"})
    epoch_offset: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps to skip in current epoch when resuming training."},
    )
    bias_wd: bool = field(
        default=False, metadata={"help": "Apply weight decay to bias parameters"}
    )
    lr_gradnorm: float = field(
        default=1e-3, metadata={"help": "Peak learning rate for the GradNorm optimizer"}
    )
    min_lr_gradnorm: float = field(
        default=1e-5,
        metadata={"help": "Minimum learning rate for the GradNorm optimizer"},
    )
    warmup_epochs_gradnorm: int = field(
        default=1, metadata={"help": "Warmup epochs for the GradNorm optimizer"}
    )


@dataclass
class WandbConfig:
    enabled: bool = field(default=True, metadata={"help": "Whether to log to wandb"})
    project: Optional[str] = field(default=None, metadata={"help": "Wandb project name"})
    entity: Optional[str] = field(default=None, metadata={"help": "Wandb entity name"})
    name: Optional[str] = field(default=None, metadata={"help": "Wandb run name"})
    run_id: Optional[str] = field(
        default=None, metadata={"help": "Wandb run id, mainly used to resume runs"}
    )
    log_code: bool = field(default=True, metadata={"help": "Log code as an artifact to wandb"})


@dataclass
class DistConfig:
    world_size: int = field(default=1, metadata={"help": "Number of distributed processes"})
    port: int = field(default=1, metadata={"help": "Port for dist communication"})
    local_rank: int = field(default=-1, metadata={"help": "Local rank"})
    enabled: bool = field(default=False, metadata={"help": "Use distributed training"})
    rank: Optional[int] = field(default=None, metadata={"help": "Global process rank"})
    dist_url: Optional[str] = field(
        default=None, metadata={"help": "Url for torch.distributed init_method"}
    )
    gpu: Optional[int] = field(default=None, metadata={"help": "Index of GPU device on machine"})
    dist_backend: Optional[str] = field(
        default=None, metadata={"help": "Backend for distributed communication"}
    )


@dataclass
class SweepConfig:
    dir: str = field(default=II("hydra:sweep.dir"), metadata={"help": "Hydra sweep directory"})
    subdir: str = field(
        default=II("hydra:sweep.subdir"), metadata={"help": "Hydra sweep sub-directory"}
    )


@dataclass
class Config:
    common: CommonConfig = CommonConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    optim: OptimizationConfig = OptimizationConfig()
    dist: DistConfig = DistConfig()
    wandb: WandbConfig = WandbConfig()
    sweep: SweepConfig = SweepConfig()

    debug: bool = field(default=False, metadata={"help": "Run in debugging mode"})
    cmd: str = field(
        default=II("oc.env:RUN_CMD"),
        metadata={"help": "Stores the CLI command used for this run"},
    )


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    reformat_logger()

    if cfg.debug:
        print("Running in debug mode")

    Path.mkdir(Path(cfg.common.output_dir), parents=True, exist_ok=True)
    Path.mkdir(Path(cfg.common.log_dir), parents=True, exist_ok=True)

    pretrain(cfg)


if __name__ == "__main__":
    import sys

    os.environ["RUN_CMD"] = f"python {' '.join(sys.argv)}"

    main()
