# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import datetime
import time

import ssvp_slt.util.misc as misc
import torch
from omegaconf import DictConfig, OmegaConf

from engine_translation import evaluate, evaluate_full, train_one_epoch
from utils_translation import (create_dataloader, create_model_and_tokenizer,
                               create_optimizer_and_loss_scaler)


def main(cfg: DictConfig):
    misc.init_distributed_mode(cfg)

    misc.seed_all(cfg.common.seed + misc.get_rank())

    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.common.device)
    model, tokenizer = create_model_and_tokenizer(cfg)

    # Load model for finetuning or eval
    if (misc.get_last_checkpoint(cfg) is None or cfg.common.eval) and cfg.common.load_model:
        misc.load_model(model, cfg.common.load_model)

    if cfg.common.eval:
        evaluate_full(cfg, model.to(device), tokenizer, device)
        exit(0)

    dataloader_train = create_dataloader("train", cfg, tokenizer)
    dataloader_val = create_dataloader("val", cfg, tokenizer)

    if cfg.wandb.enabled:
        misc.setup_wandb(cfg, model)

    print(f"Model = {str(model)}")
    print(
        f"Number of params (M): {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1.0e6):.2f}"
    )
    print(f"Learning rate: {cfg.optim.lr:.2e}")
    print(f"Accumulate grad iterations: {cfg.optim.gradient_accumulation_steps}")
    print(
        (
            "Effective batch size: "
            f"{cfg.optim.train_batch_size * cfg.optim.gradient_accumulation_steps * misc.get_world_size()}"
        )
    )

    model.to(device)
    if cfg.dist.enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=(True if cfg.model.from_scratch and cfg.fairseq else False),
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    optimizer, loss_scaler = create_optimizer_and_loss_scaler(cfg, model_without_ddp)

    misc.load_checkpoint(
        cfg=cfg,
        container={
            "model": model_without_ddp,
            "optimizer": optimizer,
            "scaler": loss_scaler,
        },
    )

    checkpoint_path = ""
    print(f"Start training for {cfg.optim.epochs} epochs")
    start_time = time.time()

    best_bleu = 0.0
    patience = cfg.optim.patience

    for epoch in range(cfg.optim.start_epoch, cfg.optim.epochs):
        is_best = False

        if cfg.dist.enabled:
            dataloader_train.sampler.set_epoch(epoch)

        for dataset in dataloader_train.dataset.datasets:
            dataset.set_epoch(epoch % cfg.data.num_epochs_extracted)

        train_stats, best_bleu_inner = train_one_epoch(
            cfg=cfg,
            model=model,
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            tokenizer=tokenizer,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            device=device,
            epoch=epoch,
            best_bleu=best_bleu,
        )
        val_stats, _, _ = evaluate(cfg, dataloader_val, model, tokenizer, device)

        if val_stats["bleu4"] <= best_bleu and best_bleu_inner <= best_bleu:
            patience -= 1
        else:
            patience = cfg.optim.patience
            if val_stats["bleu4"] > best_bleu_inner:
                is_best = True

        best_bleu = max(
            best_bleu,
            best_bleu_inner,
            val_stats["bleu4"],
        )
        print(f"Best BLEU: {best_bleu:.2f}")

        if cfg.common.output_dir:
            checkpoint_path = misc.save_checkpoint(
                cfg=cfg,
                epoch=epoch,
                checkpoint={
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": loss_scaler.state_dict(),
                },
                max_checkpoints=cfg.common.max_checkpoints,
                is_best=is_best,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "val_best_bleu4": best_bleu,
            "epoch": epoch,
        }

        if cfg.wandb.enabled and misc.is_main_process():
            misc.wandb_log(log_stats)

        if cfg.optim.early_stopping and patience == 0:
            print(
                f"Early stopping training after {cfg.optim.patience} epochs with no improvement."
            )
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    if cfg.common.eval_best_model_after_training:
        print("Loading best model")
        cfg.common.resume = cfg.common.output_dir
        misc.load_checkpoint(
            cfg=cfg,
            container={"model": model_without_ddp},
            basename="best_model",
        )

        evaluate_full(cfg, model_without_ddp, tokenizer, device)

    return [checkpoint_path]
