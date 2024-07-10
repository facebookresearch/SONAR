# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from argparse import Namespace
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import ssvp_slt.util.misc as misc
import torch
import truecase
from omegaconf import DictConfig
from ssvp_slt.data.sign_features_dataset import SignFeaturesDataset
from ssvp_slt.modeling.fairseq_model import (FairseqTokenizer,
                                             FairseqTranslationModel)
from ssvp_slt.modeling.sign_bart import (SignBartConfig,
                                         SignBartForConditionalGeneration)
from ssvp_slt.modeling.sign_t5 import (SignT5Config,
                                       SignT5ForConditionalGeneration)
from torch.utils.data import (ConcatDataset, DataLoader, DistributedSampler,
                              RandomSampler, SequentialSampler)
from transformers import AutoTokenizer, PreTrainedTokenizerFast

SUPPORTED_MODES = ["train", "val", "test"]
EVAL_BLEU_ORDER = 4


def create_model_and_tokenizer(
    cfg: DictConfig,
) -> Tuple[torch.nn.Module, Union[FairseqTokenizer, PreTrainedTokenizerFast]]:
    if cfg.fairseq:
        tokenizer = FairseqTokenizer.from_pretrained(
            cfg.model.name_or_path, do_lower_case=cfg.model.lower_case
        )
        model = FairseqTranslationModel(
            cfg.model,
            Namespace(target_dictionary=tokenizer.dictionary),
            label_smoothing=cfg.criterion.label_smoothing,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, use_fast=True)
        if "t5" in cfg.model.name_or_path:
            config = SignT5Config.from_pretrained(
                cfg.model.name_or_path,
                feature_dim=cfg.model.feature_dim,
                label_smoothing=cfg.criterion.label_smoothing,
                dropout_rate=cfg.model.dropout,
            )
            model = SignT5ForConditionalGeneration.from_pretrained(
                cfg.model.name_or_path, config=config
            )
        elif "bart" in cfg.model.name_or_path:
            config = SignBartConfig.from_pretrained(
                cfg.model.name_or_path,
                feature_dim=cfg.model.feature_dim,
                label_smoothing=cfg.criterion.label_smoothing,
                dropout=cfg.model.dropout,
            )
            model = SignBartForConditionalGeneration.from_pretrained(
                cfg.model.name_or_path, config=config
            )
        else:
            raise NotImplementedError(
                f"Model {cfg.model.name_or_path} is not supported. Supported models are `t5` and `bart`"
            )

        if "mbart" in cfg.model.name_or_path.lower():
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("</s>")
            model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids("en_XX")
            tokenizer.tgt_lang = "en_XX"

    if cfg.model.from_scratch:
        model._reinit_weights()

    print(model)

    return model, tokenizer


def get_collator(
    tokenizer: Union[FairseqTokenizer, PreTrainedTokenizerFast],
    max_target_length: int,
    do_lower_case: bool = False,
) -> Callable:
    def maybe_lower(s: str) -> str:
        return s.lower() if do_lower_case else s

    def collate_fn(samples: Dict[str, Union[str, torch.Tensor]]) -> Dict[str, Any]:
        batch_size = len(samples)

        if batch_size == 0:
            return {}

        # Get maximum sequence length in batch to pad to
        source_lengths = [len(sample["source"]) for sample in samples]
        max_length = max(source_lengths)
        source_shape = list(samples[0]["source"].size())[1:]

        # Create padded source container
        source = samples[0]["source"].new_zeros([batch_size, max_length] + source_shape)

        # Create padding mask container
        attention_mask = torch.ones(batch_size, max_length)

        # Fill padded source container and padding mask container
        for i in range(batch_size):
            source[i, : source_lengths[i]] = samples[i]["source"]
            attention_mask[i, source_lengths[i] :] = 0

        # Process labels
        if isinstance(tokenizer, FairseqTokenizer):
            padding_mask = ~attention_mask.bool()
            batch = tokenizer(
                [maybe_lower(sample["label"]) for sample in samples],
                source,
                padding_mask,
            )
            batch["labels_text"] = [sample["label"] for sample in samples]
            batch["labels"] = batch["target"]
            return batch
        else:
            labels = tokenizer(
                text_target=[maybe_lower(sample["label"]) for sample in samples],
                max_length=max_target_length,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            labels_lengths = torch.LongTensor([len(label) for label in labels])
            ntokens = labels_lengths.sum().item()

            result = {
                "ntokens": ntokens,
                "inputs_embeds": source,
                "attention_mask": attention_mask,
                "labels": labels,
                "labels_text": [sample["label"] for sample in samples],
            }
            return result

    return collate_fn


def create_dataloader(
    mode: str,
    cfg: DictConfig,
    tokenizer: Union[FairseqTokenizer, PreTrainedTokenizerFast],
) -> DataLoader:
    assert (
        mode in SUPPORTED_MODES
    ), f"Mode {mode} is not supported. Supported modes are {SUPPORTED_MODES}"

    data_dirs = cfg.data.train_data_dirs.split(",") if mode == "train" else [cfg.data.val_data_dir]
    dataset = ConcatDataset(
        [
            SignFeaturesDataset(
                mode=mode,
                path_to_data_dir=data_dir,
                min_seq_length=cfg.data.min_source_positions,
                max_seq_length=cfg.data.max_source_positions,
                indices=(0, 100) if cfg.debug else (None, None),
            )
            for data_dir in data_dirs
        ]
    )

    if cfg.dist.enabled and not cfg.common.eval:
        world_size = misc.get_world_size()
        global_rank = misc.get_rank()
        if mode == "train":
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=global_rank, shuffle=True
            )
        elif cfg.common.dist_eval:
            if len(dataset) % world_size != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler = SequentialSampler(dataset)
    else:
        sampler = RandomSampler(dataset) if mode == "train" else SequentialSampler(dataset)

    collate_fn = get_collator(
        tokenizer=tokenizer,
        max_target_length=cfg.data.max_target_positions,
        do_lower_case=cfg.model.lower_case,
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        batch_size=(cfg.optim.train_batch_size if mode == "train" else cfg.optim.val_batch_size),
        num_workers=cfg.common.num_workers,
        pin_memory=cfg.common.pin_mem,
        drop_last=(mode == "train"),
    )

    return dataloader


def create_optimizer_and_loss_scaler(
    cfg: DictConfig, model_without_ddp: torch.nn.Module
) -> Tuple[torch.optim.Optimizer, misc.NativeScalerWithGradNormCount]:
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    param_groups = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.optim.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=cfg.optim.lr)
    loss_scaler = misc.NativeScalerWithGradNormCount(fp32=not cfg.common.fp16)

    return optimizer, loss_scaler


def postprocess_text(text: List[str], do_truecase: bool = False) -> List[str]:
    # Truecasing follows Tarrés et al. 2023 (https://arxiv.org/abs/2304.06371)
    if do_truecase:
        return [truecase.get_true_case(t.strip()) for t in text]
    return [t.strip() for t in text]


def compute_bleu(meters: Dict[str, misc.SmoothedValue], order: int = EVAL_BLEU_ORDER) -> float:
    import inspect

    try:
        from sacrebleu.metrics import BLEU

        comp_bleu = BLEU.compute_bleu
    except ImportError:
        # compatibility API for sacrebleu 1.x
        import sacrebleu

        comp_bleu = sacrebleu.compute_bleu

    fn_sig = inspect.getfullargspec(comp_bleu)[0]
    if "smooth_method" in fn_sig:
        smooth = {"smooth_method": "exp"}
    else:
        smooth = {"smooth": "exp"}
    bleu = comp_bleu(
        correct=np.array([int(meters[f"_bleu_counts_{i}"].total) for i in range(order)]),
        total=np.array([int(meters[f"_bleu_totals_{i}"].total) for i in range(order)]),
        sys_len=int(meters["_bleu_sys_len"].total),
        ref_len=int(meters["_bleu_ref_len"].total),
        max_ngram_order=order,
        **smooth,
    )
    return round(bleu.score, 2)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int = 1) -> float:
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    mask = targets.ne(pad_idx)
    n_correct = torch.sum(logits.argmax(dim=1).masked_select(mask).eq(targets.masked_select(mask)))
    total = torch.sum(mask)
    return (100 * n_correct / total).item()
