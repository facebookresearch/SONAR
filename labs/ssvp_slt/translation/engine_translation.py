# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json
import math
import os
import sys
from typing import Dict, Iterable, List, Tuple, Union

import evaluate as hf_evaluate
import numpy as np
import sacrebleu
import ssvp_slt.util.lr_sched as lr_sched
import ssvp_slt.util.misc as misc
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from ssvp_slt.modeling.fairseq_model import FairseqTokenizer
from transformers import PreTrainedTokenizerFast

from utils_translation import (compute_accuracy, compute_bleu,
                               create_dataloader, postprocess_text)

EVAL_BLEU_ORDER = 4

rouge_metric = hf_evaluate.load("rouge")
meteor_metric = hf_evaluate.load("meteor")
chrf_metric = hf_evaluate.load("chrf")


def train_one_epoch(
    cfg: DictConfig,
    model: torch.nn.Module,
    dataloader_train: Iterable,
    dataloader_val: Iterable,
    tokenizer: Union[FairseqTokenizer, PreTrainedTokenizerFast],
    optimizer: torch.optim.Optimizer,
    loss_scaler: misc.NativeScalerWithGradNormCount,
    device: torch.device,
    epoch: int,
    best_bleu: float,
) -> Tuple[Dict[str, float], float]:
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = f"Epoch: [{epoch}]"
    print_freq = 50
    gradient_accumulation_steps = cfg.optim.gradient_accumulation_steps

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(dataloader_train, print_freq, header)
    ):
        # We use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % gradient_accumulation_steps == 0:
            lr_sched.adjust_learning_rate(
                cfg, optimizer, data_iter_step / len(dataloader_train) + epoch
            )

        misc.batch_to_device(batch, device)

        try:
            with torch.cuda.amp.autocast(enabled=cfg.common.fp16):
                if cfg.fairseq:
                    loss, logits = model(**batch)
                else:
                    outputs = model(**batch)
                    loss = outputs.loss
                    logits = outputs.logits

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            loss /= gradient_accumulation_steps
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=cfg.optim.clip_grad,
                parameters=model.parameters(),
                create_graph=False,
                update_grad=(data_iter_step + 1) % gradient_accumulation_steps == 0,
            )
        except RuntimeError as e:
            print(f"Runtime error {e}. Exiting")
            sys.exit(1)

        if (data_iter_step + 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())

        batch_size = len(batch["labels"])
        metric_logger.meters["acc"].update(
            compute_accuracy(logits, batch["labels"], pad_idx=tokenizer.pad_token_id),
            n=batch_size,
        )

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        acc_reduce = misc.all_reduce_mean(metric_logger.meters["acc"].value)

        if (
            misc.is_main_process()
            and cfg.wandb.enabled
            and (data_iter_step + 1) % (gradient_accumulation_steps * print_freq) == 0
        ):
            """We use epoch_1000x as the x-axis, which calibrates different curves when batch size changes."""
            epoch_1000x = int((data_iter_step / len(dataloader_train) + epoch) * 1000)
            misc.wandb_log(
                {
                    "train/loss": loss_value_reduce,
                    "train/acc": acc_reduce,
                    "train/lr": max_lr,
                    "train/grad_norm": grad_norm.item(),
                    "step": epoch_1000x,
                }
            )

        if (
            cfg.common.eval_steps is not None
            and (data_iter_step + 1) % (gradient_accumulation_steps * cfg.common.eval_steps) == 0
        ):
            if misc.is_dist_avail_and_initialized():
                dist.barrier()

            print("Running evaluation")
            eval_stats, _, _ = evaluate(cfg, dataloader_val, model, tokenizer, device)
            if eval_stats["bleu4"] > best_bleu:
                best_bleu = eval_stats["bleu4"]

                misc.save_checkpoint(
                    cfg=cfg,
                    epoch=epoch,
                    checkpoint={
                        "model": misc.unwrap_model(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": loss_scaler.state_dict(),
                    },
                    max_checkpoints=cfg.common.max_checkpoints,
                    is_best=True,
                )

            eval_stats.update(
                {
                    "best_bleu4": best_bleu,
                    "step": (epoch * len(dataloader_train)) + (data_iter_step + 1),
                }
            )
            if misc.is_main_process() and cfg.wandb.enabled:
                misc.wandb_log(
                    {f"val_inner/{k}": v for k, v in eval_stats.items()},
                    disable_format=True,
                )
            model.train(True)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, best_bleu


@torch.no_grad()
def evaluate(
    cfg: DictConfig,
    dataloader: Iterable,
    model: torch.nn.Module,
    tokenizer: Union[FairseqTokenizer, PreTrainedTokenizerFast],
    device: torch.device,
) -> Tuple[Dict[str, float], List[str], List[str]]:
    metric_logger = misc.MetricLogger(delimiter="  ")

    header = "Eval:"

    model.eval()

    all_preds = []
    all_refs = []

    samples_seen = 0
    for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, 10, header)):
        refs = [r.strip() for r in batch.pop("labels_text")]

        misc.batch_to_device(batch, device)

        with torch.cuda.amp.autocast(enabled=cfg.common.fp16):
            if cfg.fairseq:
                loss, logits = model(**batch)

                labels = batch.pop("labels")
                generated_tokens = misc.unwrap_model(model).generate(**batch)
            else:
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                labels = batch.pop("labels")
                if "decoder_input_ids" in batch:
                    batch.pop("decoder_input_ids")
                if "decoder_attention_mask" in batch:
                    batch.pop("decoder_attention_mask")
                if "ntokens" in batch:
                    batch.pop("ntokens")

                generated_tokens = misc.unwrap_model(model).generate(
                    **batch,
                    num_beams=cfg.model.num_beams,
                    max_length=cfg.data.max_target_positions,
                )

        preds = postprocess_text(
            tokenizer.batch_decode(generated_tokens, skip_special_tokens=True),
            do_truecase=cfg.model.lower_case,
        )

        # # If we are in a multiprocess environment, the last batch has duplicates
        if misc.get_world_size() > 1:
            if data_iter_step == len(dataloader) - 1:
                preds = preds[: len(dataloader.dataset) - samples_seen]
                refs = refs[: len(dataloader.dataset) - samples_seen]
            else:
                samples_seen += len(refs)
        else:
            samples_seen += len(refs)

        if cfg.common.eval_print_samples:
            for i in range(len(refs)):
                print(f"P-{samples_seen + i}: {preds[i]}")
                print(f"R-{samples_seen + i}: {refs[i]}")

        all_preds.extend(preds)
        all_refs.extend(refs)

        bleu = sacrebleu.corpus_bleu(preds, [refs])
        metric_logger.meters["_bleu_sys_len"].update(bleu.sys_len)
        metric_logger.meters["_bleu_ref_len"].update(bleu.ref_len)

        for i in range(EVAL_BLEU_ORDER):
            metric_logger.meters[f"_bleu_counts_{i}"].update(bleu.counts[i])
            metric_logger.meters[f"_bleu_totals_{i}"].update(bleu.totals[i])

        batch_size = len(generated_tokens)
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc"].update(
            compute_accuracy(logits, labels, pad_idx=tokenizer.pad_token_id),
            n=batch_size,
        )

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # Gather all preds and refs across workers
    if cfg.common.dist_eval and misc.is_dist_avail_and_initialized():
        all_preds_list = [None] * misc.get_world_size()
        all_refs_list = [None] * misc.get_world_size()
        dist.all_gather_object(all_preds_list, all_preds)
        dist.all_gather_object(all_refs_list, all_refs)
        all_preds = [h for preds in all_preds_list for h in preds]
        all_refs = [r for refs in all_refs_list for r in refs]

        print(f"Number of predictions: {len(all_preds)}\nNumber of references: {len(all_refs)}")

    derived_metrics = {}
    for i in range(1, EVAL_BLEU_ORDER + 1):
        derived_metrics[f"bleu{i}"] = compute_bleu(metric_logger.meters, order=i)
    derived_metrics.update(rouge_metric.compute(predictions=all_preds, references=all_refs))
    derived_metrics.update(meteor_metric.compute(predictions=all_preds, references=all_refs))
    derived_metrics.update(
        {"chrf": chrf_metric.compute(predictions=all_preds, references=all_refs)["score"]}
    )

    print(
        "* bleu1 {bleu1} bleu2 {bleu2} bleu 3 {bleu3} bleu4 {bleu4} rouge-l {rougeL:.3f} chrf {chrf:.3f} acc {acc1.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            acc1=metric_logger.meters["acc"],
            losses=metric_logger.loss,
            **derived_metrics,
        )
    )

    print(f"{derived_metrics = }")

    eval_stats = {
        k: meter.global_avg for k, meter in metric_logger.meters.items() if not k.startswith("_")
    }
    eval_stats.update(derived_metrics)

    return eval_stats, all_preds, all_refs


def evaluate_full(
    cfg: DictConfig,
    model: torch.nn.Module,
    tokenizer: Union[FairseqTokenizer, PreTrainedTokenizerFast],
    device: torch.device,
) -> None:
    print("Running full evaluation")

    cfg.common.eval = True
    cfg.common.dist_eval = False

    dataloaders = {}
    stats = {}
    predictions = {}
    references = {}

    # Run evaluation
    for mode in ["val", "test"]:
        print(f"Running evaluation on {mode} split")

        dataloaders[mode] = create_dataloader(mode, cfg, tokenizer)
        stats[mode], predictions[mode], references[mode] = evaluate(
            cfg, dataloaders[mode], model, tokenizer, device
        )

    # Make space on GPU for BLEURT checkpoint
    model.to("cpu")
    del model
    torch.cuda.empty_cache()

    # Optionally compute BLEURT scores and save results
    if cfg.common.compute_bleurt:
        bleurt_metric = hf_evaluate.load("bleurt", module_type="metric", config_name="BLEURT-20")
    for mode in ["val", "test"]:
        if cfg.common.compute_bleurt:
            stats[mode].update(
                {
                    "bleurt": np.array(
                        bleurt_metric.compute(
                            predictions=predictions[mode], references=references[mode]
                        )["scores"]
                    ).mean()
                }
            )

        print(f"{mode} results: {json.dumps(stats[mode], ensure_ascii=False, indent=4)}")

        with open(
            os.path.join(cfg.common.output_dir, f"{mode}_outputs.tsv"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("Prediction\tReference\n")
            for hyp, ref in zip(predictions[mode], references[mode]):
                f.write(f"{hyp}\t{ref}\n")

        with open(os.path.join(cfg.common.output_dir, f"{mode}_results.json"), "w") as f:
            json.dump(stats[mode], f, ensure_ascii=False, indent=4)

    print(f"Wrote outputs to {cfg.common.output_dir}")
