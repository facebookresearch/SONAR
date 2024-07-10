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

from main_translation import main as translate

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
    fp16: bool = field(
        default=True, metadata={"help": "Whether to use fp16 automatic mixed-precision"}
    )
    eval: bool = field(default=False, metadata={"help": "Perform evaluation only"})
    dist_eval: bool = field(
        default=True,
        metadata={
            "help": "Enabling distributed evaluation (recommended during training for faster monitor"
        },
    )
    pin_mem: bool = field(default=True, metadata={"help": "Whether to pin memory"})
    num_workers: int = field(default=10, metadata={"help": "Num dataloader workers"})
    eval_print_samples: bool = field(
        default=False,
        metadata={"help": "Whether to print hyps and refs during evaluation"},
    )
    max_checkpoints: int = field(default=3, metadata={"help": "Only keep the last N checkpoints"})
    eval_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Interval for evaluation during training. None means evaluation is only performed after every epoch."
        },
    )
    eval_best_model_after_training: bool = field(
        default=True,
        metadata={
            "help": "Whether to run evaluation on the best model (based on validation BLEU-4) after training"
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Whether to overwrite files in the output directory in case it already exists"
        },
    )
    compute_bleurt: bool = field(
        default=False,
        metadata={
            "help": (
                "Compute BLEURT scores as part of the full evaluation. "
                "Requires tensorflow-gpu and sufficient GPU memory for the BLEURT-20 checkpoint."
            )
        },
    )


@dataclass
class ModelConfig:
    name_or_path: str = field(
        default=MISSING,
        metadata={"help": "Path or identifier of pretrained transformer model"},
    )
    feature_dim: int = field(default=MISSING, metadata={"help": "Embedding dimension of features"})
    from_scratch: bool = field(default=False, metadata={"help": "Whether to train from scratch"})
    dropout: float = field(
        default=0.3, metadata={"help": "Dropout probability for the classifier"}
    )
    num_beams: int = field(default=5, metadata={"help": "Num beams for generation"})
    lower_case: bool = field(
        default=False,
        metadata={
            "help": "Whether to train with lowercase labels and apply truecasing in postprocessing"
        },
    )

    # For Fairseq compatibility
    min_source_positions: int = II("data.min_source_positions")
    max_source_positions: int = II("data.max_source_positions")
    max_target_positions: int = II("data.max_target_positions")
    feats_type: Optional[str] = field(default="hiera", metadata={"help": "Feature type"})
    activation_fn: Optional[str] = field(
        default=None, metadata={"help": "Activation function to use"}
    )
    encoder_normalize_before: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to apply layernorm before each encoder block"},
    )
    encoder_embed_dim: Optional[int] = field(
        default=None, metadata={"help": "Embedding dimension of encoder"}
    )
    encoder_ffn_embed_dim: Optional[int] = field(
        default=None, metadata={"help": "Encoder dimension for FFN"}
    )
    encoder_attention_heads: Optional[int] = field(
        default=None, metadata={"help": "Num encoder attention heads"}
    )
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layerdrop rate"}
    )
    encoder_layers: Optional[int] = field(default=None, metadata={"help": "Num encoder layers"})
    decoder_normalize_before: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to apply layernorm before each decoder block"},
    )
    decoder_embed_dim: Optional[int] = field(
        default=None, metadata={"help": "Embedding dimension of decoder"}
    )
    decoder_ffn_embed_dim: Optional[int] = field(
        default=None, metadata={"help": "Decoder dimension for FFN"}
    )
    decoder_attention_heads: Optional[int] = field(
        default=None, metadata={"help": "Num decoder attention heads"}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layerdrop rate"}
    )
    decoder_layers: Optional[int] = field(default=None, metadata={"help": "Num decoder layers"})
    decoder_output_dim: Optional[int] = field(
        default=None, metadata={"help": "Decoder dimension for outputs"}
    )
    classifier_dropout: Optional[float] = field(
        default=None, metadata={"help": "Dropout probability"}
    )
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Dropout probability for attention weights"}
    )
    activation_dropout: Optional[float] = field(
        default=None, metadata={"help": "Dropout probability after activation in FFN."}
    )
    layernorm_embedding: Optional[bool] = field(
        default=None, metadata={"help": "Apply LayerNorm to embeddings"}
    )
    no_scale_embedding: Optional[bool] = field(
        default=None, metadata={"help": "Do not scale embeddings"}
    )
    share_decoder_input_output_embed: Optional[bool] = field(
        default=None, metadata={"help": "share decoder input and output embeddings"}
    )
    num_hidden_layers: Optional[int] = field(
        default=None, metadata={"help": "Number of hidden layers."}
    )


@dataclass
class DataConfig:
    train_data_dirs: str = field(
        default=MISSING,
        metadata={"help": "Comma-separated paths to train data directories"},
    )
    val_data_dir: str = field(
        default=MISSING, metadata={"help": "Path to validation data directory"}
    )
    num_epochs_extracted: int = field(
        default=1,
        metadata={"help": "Number of epochs that (augmented) features have been extracted for"},
    )
    min_source_positions: int = field(
        default=0, metadata={"help": "Min number of tokens in the source sequence"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "Max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "Max number of tokens in the target sequence"}
    )


@dataclass
class CriterionConfig:
    label_smoothing: float = field(default=0.2, metadata={"help": "Label smoothing factor"})


@dataclass
class OptimizationConfig:
    clip_grad: float = field(default=1.0, metadata={"help": "Gradient norm clipping threshold"})
    lr: float = field(default=0.001, metadata={"help": "Learning rate"})
    min_lr: float = field(default=1e-4, metadata={"help": "Minimum learning rate after decay"})
    weight_decay: float = field(default=1e-1, metadata={"help": "Adam weight decay"})
    start_epoch: int = field(default=0, metadata={"help": "Start epoch"})
    epochs: int = field(default=200, metadata={"help": "Number of training epochs"})
    warmup_epochs: int = field(
        default=10, metadata={"help": "Number of epochs to warm up learning rate for"}
    )
    train_batch_size: int = field(default=32, metadata={"help": "Batch size for training"})
    val_batch_size: int = field(default=64, metadata={"help": "Batch size for validation"})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Gradient accumulation steps"}
    )
    early_stopping: bool = field(
        default=True, metadata={"help": "Whether to apply early stopping"}
    )
    patience: int = field(
        default=10,
        metadata={"help": "Early stop after this many epochs without improvement."},
    )
    epoch_offset: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps to skip in current epoch when resuming training."},
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
    criterion: CriterionConfig = CriterionConfig()
    optim: OptimizationConfig = OptimizationConfig()
    dist: DistConfig = DistConfig()
    wandb: WandbConfig = WandbConfig()
    sweep: SweepConfig = SweepConfig()

    debug: bool = field(default=False, metadata={"help": "Run in debugging mode"})
    fairseq: bool = field(default=False, metadata={"help": "Use fairseq functionality"})
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

    if cfg.common.eval:
        if cfg.common.load_model is None:
            raise RuntimeError(
                "Run is in eval mode but no model path has been provided via `common.load_model`."
            )
        cfg.common.output_dir = os.path.dirname(cfg.common.load_model)
        cfg.common.log_dir = os.path.join(cfg.common.output_dir, "logs")
    else:
        Path.mkdir(Path(cfg.common.output_dir), parents=True, exist_ok=True)
        Path.mkdir(Path(cfg.common.log_dir), parents=True, exist_ok=True)

    translate(cfg)


if __name__ == "__main__":
    import sys

    os.environ["RUN_CMD"] = f"python {' '.join(sys.argv)}"

    main()
