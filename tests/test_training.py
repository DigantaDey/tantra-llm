import tempfile
import unittest
from pathlib import Path

import train_corrector


class TrainingUtilityTests(unittest.TestCase):
    def test_sentence_splitting_respects_model_length(self):
        samples = train_corrector._split_samples("সাধনা মন্ত্র। " + "কুণ্ডলিনী " * 20, max_chars=30)
        self.assertTrue(samples)
        self.assertTrue(all(len(sample) <= 30 for sample in samples))
        self.assertTrue(all(sample.strip() for sample in samples))

    def test_collect_samples_reads_clean_bengali_corpus_and_terms(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clean.txt"
            path.write_text("সাধনা মন্ত্র।\nতন্ত্রের কথা।", encoding="utf-8")
            samples = train_corrector.collect_samples(
                [path], max_length=64, max_samples=100, terms={"বীজ মন্ত্র"}
            )
        self.assertIn("সাধনা মন্ত্র।", samples)
        self.assertIn("বীজ মন্ত্র", samples)

    def test_dictionary_sample_is_reserved_with_a_small_sample_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clean.txt"
            path.write_text("সাধনা মন্ত্র।\nতন্ত্রের কথা।", encoding="utf-8")
            samples = train_corrector.collect_samples(
                [path], max_length=64, max_samples=2, terms={"বিশেষবীজ"}
            )
        self.assertIn("বিশেষবীজ", samples)
        self.assertLessEqual(len(samples), 2)

    def test_synthetic_pair_dataset_is_deterministic_per_epoch(self):
        vocab_chars = set("ক্ষ সাধনা মন্ত্র")
        vocab = list(train_corrector.SPECIAL_TOKENS) + sorted(vocab_chars - set(train_corrector.SPECIAL_TOKENS))
        dataset = train_corrector.SyntheticPairDataset(
            ["ক্ষ সাধনা"],
            vocab=vocab,
            terms={"সাধনা"},
            max_length=32,
            seed=19,
            noise_scale=1.0,
            anchor_weight=2.5,
        )
        self.assertEqual(dataset[0], dataset[0])
        self.assertTrue(all(value in (1.0, 2.5) for value in dataset[0]["weights"]))
        dataset.set_epoch(1)
        self.assertEqual(dataset[0], dataset[0])


if __name__ == "__main__":
    unittest.main()
