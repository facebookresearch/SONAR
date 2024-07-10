# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import math
from argparse import Namespace
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.data import Dictionary
from fairseq.data.audio.data_cfg import MultitaskConfig, S2SDataConfig
from fairseq.data.audio.speech_to_speech_dataset import SpeechToSpeechDatasetCreator
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    TextTargetMultitaskData,
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.speech_to_text import DummyMultiTask
from fairseq.tasks.text_to_speech import batch_mel_cepstral_distortion

logger = logging.getLogger(__name__)


class StackUnitSequenceGenerator(nn.Module):
    def __init__(self, tgt_dict, vocab_size):
        super().__init__()
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.unk = tgt_dict.unk()
        self.offset = len(tgt_dict) - vocab_size
        self.vocab_size = vocab_size

    def pack_units(self, input: torch.Tensor, n_frames_per_step) -> torch.Tensor:
        if n_frames_per_step <= 1:
            return input

        bsz, _, n = input.shape
        assert n == n_frames_per_step

        scale = [
            pow(self.vocab_size, n_frames_per_step - 1 - i)
            for i in range(n_frames_per_step)
        ]
        scale = torch.LongTensor(scale).squeeze(0).to(input.device)
        mask = input >= self.offset
        res = ((input - self.offset) * scale * mask).sum(dim=2) + self.offset
        return res

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        # currently only support viterbi search for stacked units
        model = models[0]
        model.eval()

        max_len = model.max_decoder_positions()
        # TODO: incorporate max_len_a and max_len_b

        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len, _ = src_tokens.size()
        n_frames_per_step = model.decoder.n_frames_per_step

        # initialize
        encoder_out = model.forward_encoder(
            src_tokens, src_lengths, speaker=sample["speaker"]
        )
        incremental_state = {}
        pred_out, attn, scores = [], [], []
        finished = src_tokens.new_zeros((bsz,)).bool()

        prev_output_tokens = src_lengths.new_zeros((bsz, 1)).long().fill_(self.eos)
        for _ in range(max_len):
            cur_out, cur_extra = model.forward_decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
            )

            lprobs = model.get_normalized_probs([cur_out], log_probs=True)
            # never select pad, unk
            lprobs[:, :, self.pad] = -math.inf
            lprobs[:, :, self.unk] = -math.inf

            cur_pred_lprob, cur_pred_out = torch.max(lprobs, dim=2)
            scores.append(cur_pred_lprob)
            pred_out.append(cur_pred_out)

            prev_output_tokens = torch.cat(
                (
                    prev_output_tokens,
                    self.pack_units(
                        cur_pred_out.view(bsz, 1, n_frames_per_step), n_frames_per_step
                    ),
                ),
                dim=1,
            )

            attn.append(cur_extra["attn"][0])

            cur_finished = torch.any(cur_pred_out.squeeze(1) == self.eos, dim=1)
            finished = finished | cur_finished
            if finished.sum().item() == bsz:
                break

        pred_out = torch.cat(pred_out, dim=1).view(bsz, -1)
        attn = torch.cat(attn, dim=2)
        alignment = attn.max(dim=1)[1]
        attn = attn.repeat_interleave(n_frames_per_step, dim=2)
        alignment = alignment.repeat_interleave(n_frames_per_step, dim=1)
        scores = torch.cat(scores, dim=1)
        eos_idx = (pred_out == self.eos).nonzero(as_tuple=True)
        out_lens = src_lengths.new_zeros((bsz,)).long().fill_(max_len)
        for b, l in zip(eos_idx[0], eos_idx[1]):
            out_lens[b] = min(l, out_lens[b])

        hypos = [
            [
                {
                    "tokens": pred_out[b, :out_len],
                    "attn": attn[b, :, :out_len],
                    "alignment": alignment[b, :out_len],
                    "positional_scores": scores[b, :out_len],
                    "score": utils.item(scores[b, :out_len].sum().data),
                }
            ]
            for b, out_len in zip(range(bsz), out_lens)
        ]

        return hypos


@register_task("speech_to_speech")
class SpeechToSpeechTask(LegacyFairseqTask):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--multitask-config-yaml",
            type=str,
            default=None,
            help="Configuration YAML filename for the multitasks (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--target-is-code",
            action="store_true",
            help="set if target is discrete unit instead of spectrogram",
        )
        parser.add_argument(
            "--target-code-size", type=int, default=None, help="# discrete units"
        )
        parser.add_argument(
            "--n-frames-per-step",
            type=int,
            default=1,
            help="# stacked frames, use 0 for reduced discrete unit sequence",
        )
        parser.add_argument("--eval-inference", action="store_true")
        parser.add_argument(
            "--eval-args",
            type=str,
            default="{}",
            help='generation args for speech-to-unit model , e.g., \'{"beam": 5, "max_len_a": 1}\', as JSON string',
        )
        parser.add_argument("--eos-prob-threshold", type=float, default=0.5)
        parser.add_argument(
            "--mcd-normalize-type",
            type=str,
            default="targ",
            choices=["targ", "pred", "path"],
        )
        parser.add_argument(
            "--vocoder",
            type=str,
            default="griffin_lim",
            choices=["griffin_lim", "hifigan", "code_hifigan"],
        )
        parser.add_argument("--spec-bwd-max-iter", type=int, default=8)
        parser.add_argument(
            "--infer-target-lang",
            type=str,
            default="",
            help="target language for inference",
        )

    def __init__(self, args, tgt_dict, infer_tgt_lang_id=None):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.data_cfg = S2SDataConfig(Path(args.data) / args.config_yaml)

        self.multitask_tasks = {}
        self.tgt_dict_mt = None
        self.eos_token_mt = None
        if getattr(args, "multitask_config_yaml", None) is not None:
            multitask_cfg = MultitaskConfig(
                Path(args.data) / args.multitask_config_yaml
            )
            first_pass_task_idx = multitask_cfg.first_pass_decoder_task_index
            for i, (task_name, task_config) in enumerate(
                multitask_cfg.get_all_tasks().items()
            ):
                task_obj = DummyMultiTask(
                    task_config,
                    task_config.tgt_dict,
                    first_pass=i == first_pass_task_idx,
                )
                self.multitask_tasks[task_name] = task_obj
                if task_obj.is_first_pass_decoder:
                    self.tgt_dict_mt = task_obj.target_dictionary
                    if task_config.prepend_bos_and_append_tgt_lang_tag:
                        self.eos_token_mt = task_config.eos_token
                        assert not isinstance(self.eos_token_mt, List)

                        if not self.eos_token_mt:
                            raise Warning(
                                "Please provide eos_token in --multitask-config-yaml to replace eos in sequence generator"
                            )

        self._infer_tgt_lang_id = infer_tgt_lang_id

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = data_cfg = S2SDataConfig(Path(args.data) / args.config_yaml)
        tgt_dict = None
        infer_tgt_lang_id = None
        if args.target_is_code:
            if data_cfg.prepend_tgt_lang_tag_as_bos:
                # dictionary with language tags
                dict_path = Path(args.data) / data_cfg.vocab_filename
                if not dict_path.is_file():
                    raise FileNotFoundError(
                        f"Dict has to be provided when setting prepend_tgt_lang_tag_as_bos: true, but dict not found: {dict_path}"
                    )
                tgt_dict = Dictionary.load(dict_path.as_posix())

                # target langauge for inference
                if args.infer_target_lang != "":
                    tgt_lang_tag = SpeechToTextDataset.LANG_TAG_TEMPLATE.format(
                        args.infer_target_lang
                    )
                    infer_tgt_lang_id = tgt_dict.index(tgt_lang_tag)
                    assert infer_tgt_lang_id != tgt_dict.unk()
            else:
                assert args.target_code_size is not None

                tgt_dict = Dictionary()
                for i in range(args.target_code_size):
                    tgt_dict.add_symbol(str(i))
            logger.info(f"dictionary size: " f"{len(tgt_dict):,}")

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')

        assert args.n_frames_per_step >= 1
        assert (
            not args.eval_inference
            or (args.target_is_code and args.vocoder == "code_hifigan")
            or (not args.target_is_code and args.vocoder != "code_hifigan")
        )

        return cls(args, tgt_dict, infer_tgt_lang_id=infer_tgt_lang_id)

    def build_criterion(self, args):
        from fairseq import criterions

        if len(self.multitask_tasks) > 0:
            if self.args.target_is_code and not args._name.startswith("speech_to_unit"):
                raise ValueError(
                    "set --criterion speech_to_unit for speech-to-unit loss with multitask"
                )
            elif not self.args.target_is_code and not args._name.startswith(
                "speech_to_spectrogram"
            ):
                raise ValueError(
                    "set --criterion speech_to_spectrogram for speech-to-spectrogram loss with multitask"
                )

        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = SpeechToSpeechDatasetCreator.from_tsv(
            root=self.args.data,
            data_cfg=self.data_cfg,
            splits=split,
            is_train_split=split.startswith("train"),
            epoch=epoch,
            seed=self.args.seed,
            target_is_code=self.args.target_is_code,
            tgt_dict=self.target_dictionary,
            n_frames_per_step=self.args.n_frames_per_step,
            multitask=self.multitask_tasks,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def target_dictionary_mt(self):
        return self.tgt_dict_mt

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args, from_checkpoint=False):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_transformed_channels
        args.target_speaker_embed = self.data_cfg.target_speaker_embed is not None
        args.n_frames_per_step = self.args.n_frames_per_step

        model = super().build_model(args, from_checkpoint)

        if len(self.multitask_tasks) > 0:
            from fairseq.models.speech_to_speech.s2s_transformer import (
                S2STransformerMultitaskModelBase,
            )

            assert isinstance(model, S2STransformerMultitaskModelBase)

        if self.args.eval_inference:
            self.eval_gen_args = json.loads(self.args.eval_args)
            self.generator = self.build_generator(
                [model], Namespace(**self.eval_gen_args)
            )

        return model

    def build_generator_dual_decoder(
        self,
        models,
        args,
        extra_gen_cls_kwargs=None,
    ):
        from examples.speech_to_speech.unity.sequence_generator_multi_decoder import (
            MultiDecoderSequenceGenerator,
        )

        return MultiDecoderSequenceGenerator(
            models,
            self.target_dictionary,
            self.target_dictionary_mt,
            beam_size=max(1, getattr(args, "beam", 1)),
            beam_size_mt=max(1, getattr(args, "beam_mt", 1)),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            max_len_a_mt=getattr(args, "max_len_a_mt", 0),
            max_len_b_mt=getattr(args, "max_len_b_mt", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            **extra_gen_cls_kwargs,
        )

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):

        if not self.args.target_is_code or self.args.eval_inference:
            from fairseq.models.text_to_speech.vocoder import get_vocoder

            self.vocoder = get_vocoder(self.args, self.data_cfg)
            self.vocoder = (
                self.vocoder.cuda()
                if torch.cuda.is_available() and not self.args.cpu
                else self.vocoder.cpu()
            )

        has_dual_decoder = getattr(models[0], "mt_task_name", None) is not None

        if self.args.target_is_code:
            if self.args.n_frames_per_step == 1:
                if has_dual_decoder:
                    seq_generator = self.build_generator_dual_decoder(
                        models,
                        args,
                        extra_gen_cls_kwargs=extra_gen_cls_kwargs,
                    )
                else:
                    seq_generator = super().build_generator(
                        models,
                        args,
                        seq_gen_cls=None,
                        extra_gen_cls_kwargs=extra_gen_cls_kwargs,
                    )
            else:
                assert (
                    getattr(args, "beam", 1) == 1 and getattr(args, "nbest", 1) == 1
                ), "only support viterbi search for stacked units"
                seq_generator = StackUnitSequenceGenerator(
                    self.tgt_dict,
                    self.args.target_code_size,
                )
        else:
            if has_dual_decoder:
                if getattr(args, "teacher_forcing", False):
                    raise NotImplementedError
                else:
                    from fairseq.speech_generator import MultiDecoderSpeechGenerator

                    generator = MultiDecoderSpeechGenerator

                lang_token_ids_aux = {
                    i
                    for s, i in self.tgt_dict_mt.indices.items()
                    if TextTargetMultitaskData.is_lang_tag(s)
                }

                if extra_gen_cls_kwargs is None:
                    extra_gen_cls_kwargs = {}
                extra_gen_cls_kwargs[
                    "symbols_to_strip_from_output"
                ] = lang_token_ids_aux

                eos_id_mt = (
                    self.tgt_dict_mt.index(self.eos_token_mt)
                    if self.eos_token_mt
                    else None
                )
                assert eos_id_mt != self.tgt_dict_mt.unk()
                extra_gen_cls_kwargs["eos_mt"] = eos_id_mt

                seq_generator = generator(
                    models,
                    args,
                    self.vocoder,
                    self.data_cfg,
                    self.target_dictionary_mt,
                    max_iter=self.args.max_target_positions,
                    eos_prob_threshold=self.args.eos_prob_threshold,
                    **extra_gen_cls_kwargs,
                )
            else:
                if getattr(args, "teacher_forcing", False):
                    from fairseq.speech_generator import (
                        TeacherForcingAutoRegressiveSpeechGenerator,
                    )

                    generator = TeacherForcingAutoRegressiveSpeechGenerator
                    logger.info("Teacher forcing mode for generation")
                else:
                    from fairseq.speech_generator import AutoRegressiveSpeechGenerator

                    generator = AutoRegressiveSpeechGenerator

                seq_generator = generator(
                    models[0],
                    self.vocoder,
                    self.data_cfg,
                    max_iter=self.args.max_target_positions,
                    eos_prob_threshold=self.args.eos_prob_threshold,
                )

        return seq_generator

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        for task_name, task_obj in self.multitask_tasks.items():
            criterion.set_multitask_loss_weight(
                task_name, task_obj.args.get_loss_weight(update_num)
            )
            if task_name in model.multitask_decoders:
                model.multitask_decoders[task_name].train()

        loss, sample_size, logging_output = super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        for task_name in self.multitask_tasks.keys():
            if task_name in model.multitask_decoders:
                model.multitask_decoders[task_name].eval()
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if self.args.eval_inference:
            hypos, inference_losses = self.valid_step_with_inference(
                sample, model, self.generator
            )
            for k, v in inference_losses.items():
                assert k not in logging_output
                logging_output[k] = v

        return loss, sample_size, logging_output

    def valid_step_with_inference(self, sample, model, generator):
        if self.args.target_is_code:
            hypos = generator.generate([model], sample)
            tgt_lens = (
                sample["target_lengths"] - 1
            ) * self.args.n_frames_per_step  # strip <eos>
            for b, (f, l) in enumerate(zip(sample["target"], tgt_lens)):
                hypos[b][0]["targ_waveform"] = self.vocoder(
                    {"code": f[:l] - 4},  # remove <bos>, <pad>, <eos>, <unk>
                    dur_prediction=self.eval_gen_args.get("dur_prediction", False),
                )
                if len(hypos[b][0]["tokens"]) > 0:
                    hypos[b][0]["waveform"] = self.vocoder(
                        {"code": hypos[b][0]["tokens"] - 4},
                        dur_prediction=self.eval_gen_args.get("dur_prediction", False),
                    )
                else:
                    hypos[b][0]["waveform"] = torch.flip(
                        hypos[b][0]["targ_waveform"], dims=[0]
                    )
        else:
            hypos = [
                [hypo] for hypo in generator.generate(model, sample, has_targ=True)
            ]

        losses = {
            "mcd_loss": 0.0,
            "targ_frames": 0.0,
            "pred_frames": 0.0,
            "path_frames": 0.0,
            "nins": 0.0,
            "ndel": 0.0,
        }
        rets = batch_mel_cepstral_distortion(
            [hypo[0]["targ_waveform"] for hypo in hypos],
            [hypo[0]["waveform"] for hypo in hypos],
            self.data_cfg.output_sample_rate,
            normalize_type=None,
        )
        for d, extra in rets:
            pathmap = extra[-1]
            losses["mcd_loss"] += d.item()
            losses["targ_frames"] += pathmap.size(0)
            losses["pred_frames"] += pathmap.size(1)
            losses["path_frames"] += pathmap.sum().item()
            losses["nins"] += (pathmap.sum(dim=1) - 1).sum().item()
            losses["ndel"] += (pathmap.sum(dim=0) - 1).sum().item()
        losses["norm_frames"] = losses[
            f"{getattr(self.args, 'mcd_normalize_type', 'targ')}_frames"
        ]

        return hypos, losses

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            if self._infer_tgt_lang_id is not None:
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    constraints=constraints,
                    bos_token=self._infer_tgt_lang_id,
                )
            else:
                return super().inference_step(
                    generator,
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    constraints=constraints,
                )
