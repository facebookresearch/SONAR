# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# Transformers: https://github.com/huggingface/transformers
# Apex: https://github.com/NVIDIA/apex
# --------------------------------------------------------

import builtins
import datetime
import glob
import logging
import os
import random
import time
from collections import OrderedDict, defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import torch.distributed as dist
from iopath.common.file_io import g_pathmgr as pathmgr
from omegaconf import DictConfig, OmegaConf
from torch import inf

try:
    import wandb

    _is_wandb_available = True
except Exception:
    print("Wandb is not available")
    _is_wandb_available = False


logger = logging.getLogger(__name__)


class Prefetcher:
    """
    Prefetcher that records data batches (dictionaries) in a cuda stream
    Reference: https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py#L265
    """

    def __init__(self, loader: torch.utils.data.DataLoader, device: torch.device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(device)
        self.device = device
        self.preload()

    def preload(self) -> None:
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            batch_to_device(self.next_batch, self.device)

    def batch_record_stream(self, batch: Dict[str, Any]) -> None:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v.record_stream(torch.cuda.current_stream())
            elif isinstance(v, dict):
                self.batch_record_stream(v)

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            self.batch_record_stream(batch)
        self.preload()
        return batch


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self) -> Union[float, int]:
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count

    @property
    def max(self) -> Union[float, int]:
        return max(self.deque)

    @property
    def value(self) -> Union[float, int]:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
            total=self.total,
        )


class MetricLogger:
    """
    Creates a MetricLogger that holds `SmoothedValue` meters. The logger can
    synchronize meters across distributed workers and be used for step logging
    in a training or evaluation loop.
    """

    def __init__(self, delimiter: str = "\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr: str) -> Any:
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.meters.items():
            if not name.startswith("_"):
                loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def log_every(self, iterable: Iterable, print_freq: int, header: Optional[str] = None) -> None:
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )

                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def setup_for_distributed(is_main_process: bool) -> None:
    """
    This function disables printing when not in main process
    Overwrites builtin print function with logger (hack to get output from hydra logging)
    """

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_main_process or force:
            logger.info(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized() -> bool:
    """Check if torch.distributed is available and initialized"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Get world size"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get global process rank"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Check if calling process is the main process"""
    return get_rank() == 0


def init_distributed_mode(cfg: DictConfig) -> None:
    """Set up distributed torch"""

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.dist.rank = int(os.environ["RANK"])
        cfg.dist.world_size = int(os.environ["WORLD_SIZE"])
        cfg.dist.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        cfg.dist.rank = int(os.environ["SLURM_PROCID"])
        cfg.dist.gpu = cfg.dist.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_main_process=True)  # hack
        cfg.dist.enabled = False
        return

    cfg.dist.dist_url = "tcp://%s:%s" % (
        os.environ["MASTER_ADDR"],
        os.environ["MASTER_PORT"],
    )

    cfg.dist.enabled = True

    torch.cuda.set_device(cfg.dist.gpu)
    cfg.dist.dist_backend = "nccl"

    print(f"| distributed init (rank {cfg.dist.rank}): {cfg.dist.dist_url}, gpu {cfg.dist.gpu}")
    torch.distributed.init_process_group(
        backend=cfg.dist.dist_backend,
        init_method=cfg.dist.dist_url,
        world_size=cfg.dist.world_size,
        rank=cfg.dist.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(is_main_process=cfg.dist.rank == 0)


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """All-reduce-mean a single tensor"""
    world_size = get_world_size()
    if world_size > 1:
        t_reduce = torch.tensor(t).cuda()
        dist.all_reduce(t_reduce)
        t_reduce /= world_size
        return t_reduce.item()
    else:
        return t


def all_gather_tensor(t: torch.Tensor) -> torch.Tensor:
    """All-gather a single tensor"""
    t_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=t_list, tensor=t.contiguous())
    t_list[dist.get_rank()] = t
    return torch.cat(t_list, 0)


def all_gather(tensors: List[torch.Tensor]):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = get_world_size()
    for tensor in tensors:
        tensor_placeholder = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> None:
    """
    Move potentially nested batch dict to device
    """
    for k, v in batch.items():
        if k == "src_tokens":
            # TODO: is this check still necessary?
            pass
        elif isinstance(v, torch.Tensor):
            batch[k] = batch[k].to(device, non_blocking=True)
        elif isinstance(v, dict):
            batch_to_device(v, device)


def seed_all(seed: int) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NativeScalerWithGradNormCount:
    """
    GradScaler that performs gradient scaling (if fp16 amp) and clipping
    """

    state_dict_key = "amp_scaler"

    def __init__(self, fp32: bool = False):
        self._scaler = torch.cuda.amp.GradScaler(enabled=not fp32)

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad: Optional[float] = None,
        parameters: Optional[Iterable] = None,
        create_graph: bool = False,
        update_grad: bool = True,
    ) -> Optional[torch.Tensor]:
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self) -> OrderedDict:
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters: Iterable, norm_type: float = 2.0) -> torch.Tensor:
    """Compute L_p gradient norm of parameters. Defaults to p=2"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )
    return total_norm


def save_on_main_process(*args, **kwargs):
    """Only save when in main process"""
    if is_main_process():
        torch.save(*args, **kwargs)


def save_checkpoint(
    cfg: DictConfig,
    epoch: int,
    checkpoint: Dict[str, Any],
    max_checkpoints: Optional[int] = None,
    epoch_offset: Optional[int] = None,
    is_best: bool = False,
) -> Union[str, os.PathLike]:
    """
    Save a checkpoint dictionary to disk. Dictionary items should be state dicts.
    """
    checkpoint_path = os.path.join(cfg.common.output_dir, f"checkpoint-{epoch:05d}.pth")

    checkpoint["cfg"] = OmegaConf.to_container(cfg)
    checkpoint["epoch"] = epoch
    if epoch_offset is not None:
        checkpoint["epoch_offset"] = epoch_offset

    save_on_main_process(checkpoint, checkpoint_path)
    if is_best:
        save_on_main_process(checkpoint, os.path.join(cfg.common.output_dir, "best_model.pth"))

    # Clean up old checkpoints
    if max_checkpoints is not None and is_main_process():
        existing_checkpoints = sorted(
            glob.glob(os.path.join(cfg.common.output_dir, "checkpoint-*.pth")),
            reverse=True,
        )
        while len(existing_checkpoints) > max_checkpoints:
            os.remove(existing_checkpoints.pop())

    print(f"Saved checkpoint `{checkpoint_path}`")
    return checkpoint_path


def get_last_checkpoint(cfg: DictConfig) -> Union[str, os.PathLike]:
    """
    Get the last checkpoint from the checkpointing folder.
    """
    names = pathmgr.ls(cfg.common.output_dir) if pathmgr.exists(cfg.common.output_dir) else []
    names = [f for f in names if "checkpoint" in f]
    if len(names) == 0:
        print(f"No checkpoints found in '{cfg.common.output_dir}'.")
        return None
    else:
        # Sort the checkpoints by epoch.
        name = sorted(names)[-1]
        return os.path.join(cfg.common.output_dir, name)


def load_model(
    model: torch.nn.Module,
    checkpoint_path: Union[str, os.PathLike],
    model_key: str = "model"
) -> None:
    """
    Loads only the model from a saved checkpoint
    Model loading is not strict and parameters are evicted from the loaded state_dict if their
    shape does not match the one in passed `model`.
    """

    with pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    print(f"Load pre-trained checkpoint from: {checkpoint_path}")

    # Try all of these if necessary
    for candidate_key in [model_key, "model", "model_state"]:
        if candidate_key in checkpoint.keys():
            checkpoint_model = checkpoint[candidate_key]
            break

    new_checkpoint_model = {}
    for k, v in checkpoint_model.items():
        if "feature_proj" in k and "feature_projection" not in k:
            k = k.replace("feature_proj", "feature_projection.feature_proj")

        if k in model.state_dict() and model.state_dict()[k].shape != v.shape:
            print(f"Pruning {k} due to size mismatch")
        else:
            new_checkpoint_model[k] = v

    msg = model.load_state_dict(new_checkpoint_model, strict=False)
    print(msg)


def load_checkpoint(
    cfg: DictConfig,
    container: Dict[str, Any],
    basename: Optional[str] = None,
) -> None:
    """
    Loads a full saved checkpoint.
    Tries to match every instance in `container` to an item in the loaded checkpoint.
    If matched, loads the respective state_dict.
    """

    if not cfg.common.resume:
        cfg.common.resume = get_last_checkpoint(cfg)
    if cfg.common.resume:
        if cfg.common.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.common.resume, map_location="cpu", check_hash=True
            )
        else:
            if basename and not cfg.common.resume.endswith(".pth"):
                resume = os.path.join(cfg.common.resume, f"{basename}.pth")
            else:
                resume = cfg.common.resume

            with pathmgr.open(resume, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")

        matched_keys = set()
        missing_keys = []

        for k, v in container.items():
            if k in checkpoint:
                v.load_state_dict(checkpoint.pop(k))
                matched_keys.add(k)
            else:
                missing_keys.append(checkpoint.pop(k))

        if "epoch_offset" in checkpoint:
            cfg.optim.epoch_offset = checkpoint.pop("epoch_offset") + 1

        if "epoch" in checkpoint:
            cfg.optim.start_epoch = checkpoint.pop("epoch") + 1

            # Current epoch is unfinished, don't move on to next epoch yet
            if cfg.optim.epoch_offset is not None:
                cfg.optim.start_epoch -= 1

        # Try to read epoch number from filepath
        elif os.path.basename(resume).split("-")[-1].split(".")[0].isnumeric():
            cfg.optim.start_epoch = int(os.path.basename(resume).split("-")[-1].split(".")[0]) + 1
        else:
            print(
                "Warning: Could not find `epoch` in checkpoint. If resuming training, this may not be intended."
            )

        print(f"Resume checkpoint {resume}")

        if "cfg" in checkpoint:
            print(f"Loaded checkpoint config: {OmegaConf.to_yaml(checkpoint.pop('cfg'))}")

        if len(checkpoint) > 0:
            print(
                f"Warning: Found extra keys in checkpoint: {list(checkpoint.keys())}. "
                "Note that this may be expected if loading a checkpoint for evaluation only."
            )

        if "optimizer" in matched_keys:
            print("With optim & sched!")


def gpu_mem_usage() -> float:
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024**3


def cpu_mem_usage() -> Tuple[float, float]:
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024**3
    total = vram.total / 1024**3

    return usage, total


def add_weight_decay(
    model: torch.nn.Module,
    weight_decay: float = 1e-5,
    skip_list: List[str] = (),
    bias_wd: bool = False,
) -> List[Dict[str, Any]]:
    """Return param groups with and without weight decay"""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (not bias_wd) and len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.

    Reference:
    https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def setup_wandb(cfg: DictConfig, model: torch.nn.Module) -> None:
    """Setup wandb logging"""

    if _is_wandb_available and is_main_process():
        skip_wandb_resume = False
        if cfg.common.resume is None or cfg.wandb.run_id is None:
            skip_wandb_resume = True  # can't resume run because run_id has previously not been set
            cfg.wandb.run_id = os.getenv("SLURM_JOB_ID", str(int(time.time())))

        wandb.init(
            name=wandb_run_name(cfg),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            dir=cfg.common.log_dir,
            config=OmegaConf.to_container(cfg),
            id=cfg.wandb.run_id,
            resume=cfg.common.resume is not None and not skip_wandb_resume,
            save_code=False,
        )
        if cfg.wandb.log_code:
            wandb.run.log_code(
                os.path.dirname(os.getcwd()),
                include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"),
            )
        wandb.watch(model, log="all", log_freq=1000)


def wandb_log(data: Dict[str, Any], disable_format: bool = False) -> None:
    """Log a dictionary to wandb"""
    if _is_wandb_available:
        formatted_data = (
            data
            if disable_format
            else {
                k.replace("val_", "val/").replace("train_", "train/"): v for k, v in data.items()
            }
        )
        wandb.log(formatted_data)


def wandb_run_name(cfg: DictConfig) -> str:
    """Create a wandb run name from hydra overrides and slurm environment vars"""
    slurm_str = (
        f"{os.getenv('SLURM_ARRAY_JOB_ID', None) or os.getenv('SLURM_JOB_ID', 0)}"
        f"_{os.getenv('SLURM_ARRAY_TASK_ID', 0)}"
    )
    run_str = cfg.common.output_dir.replace("/", "__")
    return f"{slurm_str}_{run_str}"


def reformat_logger() -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s]  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
