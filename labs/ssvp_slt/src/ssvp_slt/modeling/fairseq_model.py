# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# References:
#
# Fairseq: https://github.com/facebookresearch/fairseq
# --------------------------------------------------------

import os
from argparse import Namespace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from fairseq import search
from fairseq.data import Dictionary, data_utils, encoders
from fairseq.models.sign_language import Sign2TextTransformerModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params


class FairseqTokenizer:
    """
    Wrapper around Fairseq tokenization utils that is consisent with the main functionality of the HF tokenizer API
    """

    def __init__(
        self, dictionary_path: str, spm_path: str, do_lower_case: bool = False, **kwargs
    ) -> None:
        self.do_lower_case = do_lower_case
        print(f"[FairseqTokenizer]: lowercase={self.do_lower_case}")

        self.dictionary = Dictionary.load(dictionary_path)

        if not os.path.isfile(spm_path):
            print("Using GPT-2 BPE model")
            bpe = "gpt2"
        else:
            bpe = Namespace(**{"bpe": "sentencepiece", "sentencepiece_model": spm_path})

        self.bpe_tokenizer = encoders.build_bpe(bpe)

    def __call__(
        self,
        labels: List[str],
        source: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[str]:
        targets = []
        for label in labels:
            if self.do_lower_case:
                label = label.lower()
            label = self.bpe_tokenizer.encode(label)
            label = self.dictionary.encode_line(
                label,
                append_eos=True,
                add_if_not_exist=False,
            ).long()
            targets.append(label)

        target_lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = target_lengths.sum().item()
        pad, eos = self.dictionary.pad(), self.dictionary.eos()
        targets_ = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False)
        prev_output_tokens = data_utils.collate_tokens(
            targets,
            pad_idx=pad,
            eos_idx=eos,
            left_pad=False,
            move_eos_to_beginning=True,
        )

        net_input = {
            "prev_output_tokens": prev_output_tokens,
            "utt_id": [-1],
        }
        if source is not None:
            net_input["source"] = source
        if padding_mask is not None:
            net_input["padding_mask"] = padding_mask
        batch = {
            "utt_id": [-1],
            "id": [-1],
            "net_input": net_input,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "target": targets_,
        }

        return batch

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        print(f"Loading pretrained `FairseqTokenizer` from `{path}` with {kwargs = }")
        dictionary_path = os.path.join(path, "dictionary.txt")
        spm_path = os.path.join(path, "sentencepiece.model")
        return cls(dictionary_path, spm_path, **kwargs)

    def decode(self, tok: Union[torch.Tensor, List[int]], symbols_ignore=None):
        tok = self.dictionary.string(tok, extra_symbols_to_ignore=symbols_ignore)
        if self.bpe_tokenizer:
            tok = self.bpe_tokenizer.decode(tok)
        return tok

    def batch_decode(
        self,
        batch: Union[torch.Tensor, List[torch.Tensor], List[List[int]]],
        skip_special_tokens: bool = True,
        **kwargs,
    ):
        symbols_ignore = (
            {self.dictionary.eos(), self.dictionary.pad()} if skip_special_tokens else None
        )
        result = []
        for tok in batch:
            tok = self.dictionary.string(tok, extra_symbols_to_ignore=symbols_ignore)
            if self.bpe_tokenizer:
                tok = self.bpe_tokenizer.decode(tok)
                result.append(tok)
        return result

    def __len__(self):
        return len(self.dictionary)

    @property
    def pad_token_id(self):
        return self.dictionary.pad()

    @property
    def eos_token_id(self):
        return self.dictionary.eos()

    @property
    def bos_token_id(self):
        return self.dictionary.bos()


class FairseqTranslationModel(nn.Module):
    def __init__(self, cfg, task, label_smoothing: float = 0.1):
        super().__init__()

        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, ignore_index=task.target_dictionary.pad()
        )
        self.transformer = Sign2TextTransformerModel.build_model(cfg, task)
        self.gen_args = {"beam": 5, "lenpen": 1.0}

        self.generator = build_generator(
            task.target_dictionary, [self.transformer], Namespace(**self.gen_args)
        )

    @torch.no_grad()
    def generate(self, **kwargs) -> torch.Tensor:
        sample = kwargs
        gen_out = self.generator.generate(
            [self.transformer], sample, prefix_tokens=None, constraints=None
        )
        generated_tokens = [gen_out[i][0]["tokens"].int() for i in range(len(gen_out))]
        return generated_tokens

    def forward(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = kwargs
        decoder_output = self.transformer(**sample["net_input"])[0]
        loss = self.criterion(
            decoder_output.view(-1, decoder_output.size(-1)), sample["target"].view(-1)
        )
        return loss, decoder_output

    def _reinit_weights(self) -> None:
        print("Reinitializing weights")

        self.apply(init_bert_params)

        def init_layernorm_params(module):
            if isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

        self.apply(init_layernorm_params)


def build_generator(
    target_dictionary,
    models,
    args,
    seq_gen_cls=None,
    extra_gen_cls_kwargs=None,
    prefix_allowed_tokens_fn=None,
):
    """
    Build a :class:`~fairseq.SequenceGenerator` instance for this
    task.

    Args:
        models (List[~fairseq.models.FairseqModel]): ensemble of models
        args (fairseq.dataclass.configs.GenerationConfig):
            configuration object (dataclass) for generation
        extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
            through to SequenceGenerator
        prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
            If provided, this function constrains the beam search to
            allowed tokens only at each step. The provided function
            should take 2 arguments: the batch ID (`batch_id: int`)
            and a unidimensional tensor of token ids (`inputs_ids:
            torch.Tensor`). It has to return a `List[int]` with the
            allowed tokens for the next generation step conditioned
            on the previously generated tokens (`inputs_ids`) and
            the batch ID (`batch_id`). This argument is useful for
            constrained generation conditioned on the prefix, as
            described in "Autoregressive Entity Retrieval"
            (https://arxiv.org/abs/2010.00904) and
            https://github.com/facebookresearch/GENRE.
    """
    if getattr(args, "score_reference", False):
        from fairseq.sequence_scorer import SequenceScorer

        return SequenceScorer(
            target_dictionary,
            compute_alignment=getattr(args, "print_alignment", False),
        )

    from fairseq.sequence_generator import (SequenceGenerator,
                                            SequenceGeneratorWithAlignment)

    # Choose search strategy. Defaults to Beam Search.
    sampling = getattr(args, "sampling", False)
    sampling_topk = getattr(args, "sampling_topk", -1)
    sampling_topp = getattr(args, "sampling_topp", -1.0)
    diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
    diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
    match_source_len = getattr(args, "match_source_len", False)
    diversity_rate = getattr(args, "diversity_rate", -1)
    constrained = getattr(args, "constraints", False)
    if prefix_allowed_tokens_fn is None:
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
    if (
        sum(
            int(cond)
            for cond in [
                sampling,
                diverse_beam_groups > 0,
                match_source_len,
                diversity_rate > 0,
            ]
        )
        > 1
    ):
        raise ValueError("Provided Search parameters are mutually exclusive.")
    assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
    assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

    if sampling:
        search_strategy = search.Sampling(target_dictionary, sampling_topk, sampling_topp)
    elif diverse_beam_groups > 0:
        search_strategy = search.DiverseBeamSearch(
            target_dictionary, diverse_beam_groups, diverse_beam_strength
        )
    elif match_source_len:
        # this is useful for tagging applications where the output
        # length should match the input length, so we hardcode the
        # length constraints for simplicity
        search_strategy = search.LengthConstrainedBeamSearch(
            target_dictionary,
            min_len_a=1,
            min_len_b=0,
            max_len_a=1,
            max_len_b=0,
        )
    elif diversity_rate > -1:
        search_strategy = search.DiverseSiblingsSearch(target_dictionary, diversity_rate)
    elif constrained:
        search_strategy = search.LexicallyConstrainedBeamSearch(
            target_dictionary, args.constraints
        )
    elif prefix_allowed_tokens_fn:
        search_strategy = search.PrefixConstrainedBeamSearch(
            target_dictionary, prefix_allowed_tokens_fn
        )
    else:
        search_strategy = search.BeamSearch(target_dictionary)

    extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
    if seq_gen_cls is None:
        if getattr(args, "print_alignment", False):
            seq_gen_cls = SequenceGeneratorWithAlignment
            extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
        else:
            seq_gen_cls = SequenceGenerator

    return seq_gen_cls(
        models,
        target_dictionary,
        beam_size=getattr(args, "beam", 5),
        max_len_a=getattr(args, "max_len_a", 0),
        max_len_b=getattr(args, "max_len_b", 200),
        min_len=getattr(args, "min_len", 1),
        normalize_scores=(not getattr(args, "unnormalized", False)),
        len_penalty=getattr(args, "lenpen", 1),
        unk_penalty=getattr(args, "unkpen", 0),
        temperature=getattr(args, "temperature", 1.0),
        match_source_len=getattr(args, "match_source_len", False),
        no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
        search_strategy=search_strategy,
        **extra_gen_cls_kwargs,
    )
