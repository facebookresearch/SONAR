# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from fairseq2.models.sequence import SequenceBatch
from torch.testing import assert_close  # type: ignore

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    TextToTextModelPipeline,
)


class TestSonarTextClass:
    eng_sentences = ["Hello, my name is Paul", "I'm working as a teacher"]
    fr_sentences = ["Bonjour, mon nom est Paul", "Je travaille comme professeur."]

    text2vec = TextToEmbeddingModelPipeline(
        "text_sonar_basic_encoder", "text_sonar_basic_encoder"
    )
    text2text = TextToTextModelPipeline(
        "text_sonar_basic_encoder",
        "text_sonar_basic_decoder",
        "text_sonar_basic_encoder",
    )

    def get_normalized_embeddings(
        self, sentences: List[str], lang: str
    ) -> torch.Tensor:
        output = self.text2vec.predict(sentences, source_lang=lang)
        normalized_embed_sentence = torch.nn.functional.normalize(output, dim=-1)
        return normalized_embed_sentence

    @torch.inference_mode()
    def test_text_encoder_sonar_basic(self) -> None:
        encoded_eng = self.get_normalized_embeddings(
            self.eng_sentences, lang="eng_Latn"
        )
        encoded_fr = self.get_normalized_embeddings(self.fr_sentences, lang="fra_Latn")
        actual_sim = torch.matmul(encoded_eng, encoded_fr.T)
        expected_sim = torch.Tensor([[0.9367, 0.3658], [0.3787, 0.8596]])
        assert_close(actual_sim, expected_sim, rtol=1e-4, atol=1e-4)

    @torch.inference_mode()
    def test_text_decoder_sonar(self) -> None:
        eng_tokenizer_encoder = self.text2text.tokenizer.create_encoder(lang="eng_Latn")
        tokenized_seq = eng_tokenizer_encoder(self.eng_sentences[0]).unsqueeze(0)
        batch = SequenceBatch(tokenized_seq, None)
        encoded_vec = self.text2text.model.encoder(batch)

        decoder = self.text2text.model.decoder
        dummy_prev_output_tokens = torch.Tensor([[3, 333]]).int()
        seqs, padding_mask = decoder.decoder_frontend(
            dummy_prev_output_tokens, seq_lens=None
        )

        decoder_output, decoder_padding_mask = decoder.decoder(
            seqs,
            padding_mask,
            encoder_output=encoded_vec.sentence_embeddings.unsqueeze(1),
        )
        decoder_output = decoder.project(decoder_output, decoder_padding_mask)
        out = decoder_output.logits
        assert_close(
            out[0, 0, :4],
            torch.Tensor([-1.4572, -2.7325, -1.0546, 0.7818]),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_close(
            out[0, 0, -3:],
            torch.Tensor([0.8982, 0.4996, -0.1487]),
            rtol=1e-4,
            atol=1e-4,
        )

        assert_close(
            out[0, 1, :4],
            torch.Tensor([2.4092, 6.9624, 3.6308, 9.4825]),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_close(
            out[0, 1, -4:],
            torch.Tensor([3.8826, 3.8777, 3.2820, 3.3275]),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_encoder_decoder_translate(self) -> None:
        french_translated_sentences = self.text2text.predict(
            self.eng_sentences, source_lang="eng_Latn", target_lang="fra_Latn"
        )
        actual = french_translated_sentences
        assert actual == self.fr_sentences
