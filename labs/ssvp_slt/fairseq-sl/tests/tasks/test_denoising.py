# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from tempfile import TemporaryDirectory

from fairseq import options
from fairseq.binarizer import FileBinarizer, VocabularyDatasetBinarizer
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks.denoising import DenoisingTask
from tests.utils import build_vocab, make_data


class TestDenoising(unittest.TestCase):
    def test_denoising(self):
        with TemporaryDirectory() as dirname:

            # prep input file
            raw_file = os.path.join(dirname, "raw")
            data = make_data(out_file=raw_file)
            vocab = build_vocab(data)

            # binarize
            binarizer = VocabularyDatasetBinarizer(vocab, append_eos=False)
            split = "train"
            bin_file = os.path.join(dirname, split)
            dataset_impl = "mmap"
            FileBinarizer.multiprocess_dataset(
                input_file=raw_file,
                binarizer=binarizer,
                dataset_impl=dataset_impl,
                vocab_size=len(vocab),
                output_prefix=bin_file,
            )

            # setup task
            train_args = options.parse_args_and_arch(
                options.get_training_parser(),
                [
                    "--task",
                    "denoising",
                    "--arch",
                    "bart_base",
                    "--seed",
                    "42",
                    "--mask-length",
                    "word",
                    "--permute-sentences",
                    "1",
                    "--rotate",
                    "0",
                    "--replace-length",
                    "-1",
                    "--mask",
                    "0.2",
                    dirname,
                ],
            )
            cfg = convert_namespace_to_omegaconf(train_args)
            task = DenoisingTask(cfg.task, binarizer.dict)

            # load datasets
            original_dataset = task._load_dataset_split(bin_file, 1, False)
            task.load_dataset(split)
            masked_dataset = task.dataset(split)

            iterator = task.get_batch_iterator(
                dataset=masked_dataset,
                max_tokens=65_536,
                max_positions=4_096,
            ).next_epoch_itr(shuffle=False)
            mask_index = task.source_dictionary.index("<mask>")
            for batch in iterator:
                for sample in range(len(batch)):
                    net_input = batch["net_input"]
                    masked_src_tokens = net_input["src_tokens"][sample]
                    masked_src_length = net_input["src_lengths"][sample]
                    masked_tgt_tokens = batch["target"][sample]

                    sample_id = batch["id"][sample]
                    original_tokens = original_dataset[sample_id]
                    original_tokens = original_tokens.masked_select(
                        masked_src_tokens[:masked_src_length] == mask_index
                    )
                    masked_tokens = masked_tgt_tokens.masked_select(
                        masked_src_tokens == mask_index
                    )

                    assert masked_tokens.equal(original_tokens)


if __name__ == "__main__":
    unittest.main()
