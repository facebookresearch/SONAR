# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from argparse import Namespace
from pathlib import Path
from typing import List

from fairseq.data import Dictionary, encoders
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data.audio.data_cfg import MultitaskConfig
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    TextTargetMultitaskData,
)
from fairseq.tasks import LegacyFairseqTask, register_task

logger = logging.getLogger(__name__)


@register_task("speech_to_text")
class SpeechToTextTask(LegacyFairseqTask):
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

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        self.speaker_to_id = self._get_speaker_to_id()
        if (
            self.data_cfg.prepend_tgt_lang_tag
            and self.data_cfg.prepend_bos_and_append_tgt_lang_tag
        ):
            raise ValueError(
                "Please set only one of the two options to avoid adding target token multiple times"
            )

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

    def _get_speaker_to_id(self):
        speaker_to_id = None
        speaker_set_filename = self.data_cfg.config.get("speaker_set_filename")
        if speaker_set_filename is not None:
            speaker_set_path = Path(self.args.data) / speaker_set_filename
            with open(speaker_set_path) as f:
                speaker_to_id = {r.strip(): i for i, r in enumerate(f)}
        return speaker_to_id

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        dict_path = Path(args.data) / data_cfg.vocab_filename
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        tgt_dict = Dictionary.load(dict_path.as_posix())
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            root=self.args.data,
            cfg=self.data_cfg,
            splits=split,
            tgt_dict=self.tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            speaker_to_id=self.speaker_to_id,
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
        args.input_channels = self.data_cfg.input_channels
        args.speaker_to_id = self.speaker_to_id
        return super(SpeechToTextTask, self).build_model(args, from_checkpoint)

    def build_generator_dual_decoder(
        self,
        models,
        args,
        extra_gen_cls_kwargs,
    ):
        from examples.speech_to_speech.unity.sequence_generator_multi_decoder import (
            MultiDecoderSequenceGenerator,
        )

        lang_token_ids_aux = {
            i
            for s, i in self.tgt_dict_mt.indices.items()
            if TextTargetMultitaskData.is_lang_tag(s)
        }

        extra_gen_cls_kwargs["symbols_to_strip_from_output"].update(lang_token_ids_aux)

        eos_id_mt = (
            self.tgt_dict_mt.index(self.eos_token_mt) if self.eos_token_mt else None
        )
        assert eos_id_mt != self.tgt_dict_mt.unk()
        extra_gen_cls_kwargs["eos_mt"] = eos_id_mt

        return MultiDecoderSequenceGenerator(
            models,
            self.target_dictionary,
            self.target_dictionary_mt,
            beam_size=max(1, getattr(args, "beam", 1)),
            beam_size_mt=max(1, getattr(args, "beam_mt", 1)),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            max_len_a_mt=getattr(args, "max_len_a_mt", 0),
            max_len_b_mt=getattr(args, "max_len_b_mt", 0),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            len_penalty_mt=getattr(args, "lenpen_mt", 1),
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
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }

        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}
        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids

        eos_token = (
            args.eos_token
            if "eos_token" in args and args.eos_token is not None
            else self.data_cfg.config.get("eos_token", None)
        )

        if self.data_cfg.prepend_bos_and_append_tgt_lang_tag and not eos_token:
            raise Warning(
                "Please provide --eos_token to replace eos in sequence generator"
            )

        eos_id = self.tgt_dict.index(eos_token) if eos_token else None
        extra_gen_cls_kwargs["eos"] = eos_id

        has_dual_decoder = getattr(models[0], "mt_task_name", None) is not None

        if has_dual_decoder:
            return self.build_generator_dual_decoder(
                models,
                args,
                extra_gen_cls_kwargs=extra_gen_cls_kwargs,
            )
        else:
            return super().build_generator(
                models,
                args,
                seq_gen_cls=None,
                extra_gen_cls_kwargs=extra_gen_cls_kwargs,
            )

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
        for task_name, task_obj in self.multitask_tasks.items():
            if task_name in model.multitask_decoders:
                model.multitask_decoders[task_name].eval()
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        return loss, sample_size, logging_output

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )


class DummyMultiTask(LegacyFairseqTask):
    def __init__(self, args, tgt_dict, first_pass=False):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.first_pass = first_pass

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def is_first_pass_decoder(self):
        return self.first_pass

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        if self.args.decoder_type == "ctc":
            model = models[0]  # only support single model
            encoder_out = model(**sample)
            if hasattr(model, "get_logits"):
                emissions = model.get_logits(
                    encoder_out
                )  # no need to normalize emissions
            else:
                emissions = model.get_normalized_probs(encoder_out, log_probs=True)
            return generator.decode(
                emissions.transpose(0, 1).float().cpu().contiguous()
            )
        else:
            raise NotImplementedError("only ctc decoder is supported at the moment")

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if self.args.decoder_type == "ctc":
            from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder

            return W2lViterbiDecoder(args, self.tgt_dict)
        else:
            raise NotImplementedError("only ctc decoder is supported at the moment")
