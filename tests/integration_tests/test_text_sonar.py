# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import pytest
import torch
from fairseq2.nn import BatchLayout
from torch.testing import assert_close  # type: ignore

from sonar.inference_pipelines.text import (
    EmbeddingToTextModelPipeline,
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
        "text_sonar_basic_encoder",  # name of the tokenizer
    )
    vec2text = EmbeddingToTextModelPipeline(
        "text_sonar_basic_decoder",  # name of the decoder
        "text_sonar_basic_encoder",  # name of the tokenizer
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
    def test_encode_long_text(self) -> None:
        """Text that the encoder does not fail with long texts"""
        with pytest.warns():
            self.get_normalized_embeddings(["Hello! " * 1000], lang="eng_Latn")

    @torch.inference_mode()
    def test_text_decoder_sonar(self) -> None:
        eng_tokenizer_encoder = self.text2text.decoder_tokenizer.create_encoder(
            lang="eng_Latn"
        )
        tokenized_seq = eng_tokenizer_encoder(self.eng_sentences[0]).unsqueeze(0)
        tokenized_seq_layout = BatchLayout.of(tokenized_seq)
        encoded_vec = self.text2text.model.encoder(tokenized_seq, tokenized_seq_layout)

        decoder = self.text2text.model.decoder
        dummy_prev_output_tokens = torch.Tensor([[3, 333]]).int()
        dummy_prev_output_layout = BatchLayout.of(dummy_prev_output_tokens)
        seqs, seqs_layout = decoder.decoder_frontend(
            dummy_prev_output_tokens, dummy_prev_output_layout
        )
        encoder_output = encoded_vec.sentence_embeddings.unsqueeze(1)
        encoder_output_layout = BatchLayout.of(encoder_output)
        decoder_output = decoder.decoder(
            seqs,
            seqs_layout,
            encoder_output=encoder_output,
            encoder_output_layout=encoder_output_layout,
        )

        decoder_output = decoder.project(decoder_output)

        assert_close(
            decoder_output[0, 0, :4],
            torch.Tensor([-1.4572, -2.7325, -1.0546, 0.7818]),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_close(
            decoder_output[0, 0, -3:],
            torch.Tensor([0.8982, 0.4996, -0.1487]),
            rtol=1e-4,
            atol=1e-4,
        )

        assert_close(
            decoder_output[0, 1, :4],
            torch.Tensor([2.4092, 6.9624, 3.6308, 9.4825]),
            rtol=1e-4,
            atol=1e-4,
        )
        assert_close(
            decoder_output[0, 1, -4:],
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

    @torch.inference_mode()
    def test_vec2text_decode(self) -> None:
        encoded_eng = self.text2vec.predict(self.eng_sentences, source_lang="eng_Latn")
        decoded_fra = self.vec2text.predict(encoded_eng, target_lang="fra_Latn")
        assert decoded_fra == self.fr_sentences

    def test_order_preserving(self) -> None:
        random_sentneces = [
            "xwz",
            "qazwsxedcrfvtg",
            "rtyuio",
            "asdfghjklmnbv",
            "zxcvb",
            "poiuytrewq",
            "mnbvcxzasdfg",
            "lkjhgfdsaq",
            "qwertyuiopk",
            "asdfgh",
        ]
        output1 = self.text2vec.predict(
            random_sentneces, source_lang="eng_Latn", batch_size=2
        )
        output2 = self.text2vec.predict(
            random_sentneces, source_lang="eng_Latn", batch_size=1
        )
        output3 = self.text2vec.predict(
            random_sentneces,
            source_lang="eng_Latn",
            batch_size=None,
            batch_max_tokens=5,
        )
        output4 = self.text2vec.predict(
            random_sentneces,
            source_lang="eng_Latn",
            batch_size=None,
            batch_max_tokens=30,
        )
        output5 = torch.cat(
            [
                self.text2vec.predict([x], source_lang="eng_Latn")
                for x in random_sentneces
            ],
            dim=0,
        )
        assert_close(output1, output2)
        assert_close(output2, output3)
        assert_close(output3, output4)
        assert_close(output4, output5)
