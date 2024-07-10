# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import argparse
import logging
from multiprocessing import cpu_count
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

import pandas as pd
import sentencepiece as sp

UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1

logger = logging.getLogger(__name__)


def gen_vocab(
    input_path: Path,
    output_dir: Path,
    model_type="bpe",
    vocab_size=7000,
    special_symbols: Optional[List[str]] = None,
):
    # Train SentencePiece Model
    output_path_prefix = output_dir / "sentencepiece"
    arguments = [
        f"--input={input_path.as_posix()}",
        f"--model_prefix={output_path_prefix.as_posix()}",
        f"--model_type={model_type}",
        f"--vocab_size={vocab_size}",
        "--character_coverage=1.0",
        f"--num_threads={cpu_count()}",
        f"--unk_id={UNK_TOKEN_ID}",
        f"--bos_id={BOS_TOKEN_ID}",
        f"--eos_id={EOS_TOKEN_ID}",
        f"--pad_id={PAD_TOKEN_ID}",
    ]
    if special_symbols is not None:
        _special_symbols = ",".join(special_symbols)
        arguments.append(f"--user_defined_symbols={_special_symbols}")
    sp.SentencePieceTrainer.Train(" ".join(arguments))

    prefix = output_path_prefix.as_posix()

    spm = sp.SentencePieceProcessor()
    spm.Load(f"{prefix}.model")
    vocab = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}
    assert (
        vocab.get(UNK_TOKEN_ID) == UNK_TOKEN
        and vocab.get(PAD_TOKEN_ID) == PAD_TOKEN
        and vocab.get(BOS_TOKEN_ID) == BOS_TOKEN
        and vocab.get(EOS_TOKEN_ID) == EOS_TOKEN
    )
    vocab = {
        i: s for i, s in vocab.items() if s not in {UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    }
    with open(output_dir / "dictionary.txt", "w") as f_out:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            f_out.write(f"{s} 1\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_file", required=True, nargs="+", type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--vocab_size", required=True, type=int)
    parser.add_argument(
        "--vocab_type",
        default="unigram",
        type=str,
        choices=["bpe", "unigram", "char", "word"],
    )
    parser.add_argument("--column", default="caption", type=str)
    parser.add_argument("--lowercase", action="store_true", default=False)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = []
    for tsv_file in args.tsv_file:
        tsv_file = Path(tsv_file).expanduser().resolve()
        df = pd.read_csv(
            tsv_file, sep="\t", quoting=3, names=["video_name", "duration", "caption"]
        )
        sentences.extend(df[args.column].to_list())

    with NamedTemporaryFile(mode="w") as f:
        for sent in sentences:
            if args.lowercase:
                sent = sent.lower()
            f.write(str(sent) + "\n")

        gen_vocab(
            Path(f.name),
            output_dir,
            args.vocab_type,
            args.vocab_size,
        )


if __name__ == "__main__":
    main()
