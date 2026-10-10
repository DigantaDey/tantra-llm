import math
import random
import tempfile
import unittest
from pathlib import Path

import cleaner


class NormalizationTests(unittest.TestCase):
    def test_unicode_and_wrapped_lines_are_normalized(self):
        source = "অক্ষর  সাধনা\n   তন্ত্র\n\n মন্ত্র।  "
        self.assertEqual(cleaner.normalize_ocr_text(source), "অক্ষর সাধনা তন্ত্র\n\nমন্ত্র।")

    def test_preserve_lines_and_explicit_hasant_repair(self):
        source = "ক্ ষ\nতন্ত্র"
        self.assertEqual(cleaner.normalize_ocr_text(source, preserve_lines=True), "ক্ষ\nতন্ত্র")

    def test_form_feed_and_invisible_characters(self):
        source = "মন্ত্র\ufeff\u200b\x0cসাধনা"
        self.assertEqual(cleaner.normalize_ocr_text(source), "মন্ত্র\n\nসাধনা")

    def test_lookalike_groups_keep_overlapping_alternatives(self):
        self.assertTrue({"র", "ক", "ভ"}.issubset(set(cleaner.CONFUSIONS["ব"])))
        self.assertIn("ব", cleaner.CONFUSIONS["র"])

    def test_synthetic_corruption_is_seeded_and_can_split_ligatures(self):
        first = cleaner.synthetic_corrupt(
            "ক্ষ",
            rng=random.Random(7),
            conjunct_rate=1,
            confusion_rate=0,
            whitespace_rate=0,
        )
        second = cleaner.synthetic_corrupt(
            "ক্ষ",
            rng=random.Random(7),
            conjunct_rate=1,
            confusion_rate=0,
            whitespace_rate=0,
        )
        self.assertEqual(first, second)
        self.assertEqual(first, "ক ষ")


class LexiconTests(unittest.TestCase):
    def test_visual_confusion_is_only_repaired_with_a_domain_anchor(self):
        unanchored = cleaner.LexiconCorrector()
        self.assertEqual(unanchored.correct("ন্যাশ"), "ন্যাশ")

        anchored = cleaner.LexiconCorrector(terms={"ন্যাস"})
        self.assertEqual(anchored.correct("ন্যাশ"), "ন্যাস")

    def test_exact_domain_spelling_is_protected(self):
        corrector = cleaner.LexiconCorrector(terms={"ন্যাশ"})
        self.assertEqual(corrector.correct("ন্যাশ"), "ন্যাশ")

    def test_broken_conjunct_merge_requires_a_known_word(self):
        unknown = cleaner.LexiconCorrector()
        self.assertEqual(unknown.correct("অক ষর"), "অক ষর")
        anchored = cleaner.LexiconCorrector(terms={"অক্ষর"})
        self.assertEqual(anchored.correct("অক ষর"), "অক্ষর")

    def test_char_language_model_scores_both_context_directions(self):
        model = cleaner.BidirectionalCharLanguageModel("সাধনা মন্ত্র সাধনা মন্ত্র")
        self.assertTrue(math.isfinite(model.score("সাধনা মন্ত্র")))
        self.assertGreater(model.score("সাধনা মন্ত্র"), model.score("কখগ ঘঙচ"))

    def test_clean_loop_keeps_terms_and_normalizes_layout(self):
        corrector = cleaner.LexiconCorrector(terms={"ন্যাস"})
        output = cleaner.clean_ocr_text("ন্যাশ\n মন্ত্র।", corrector=corrector)
        self.assertEqual(output, "ন্যাস মন্ত্র।")


class OutputTests(unittest.TestCase):
    def test_markdown_frontmatter_is_json_quoted(self):
        rendered = cleaner.markdown_document(
            "মন্ত্র।",
            title='A "quoted" title',
            source="archive source",
            identifier="item_1",
        )
        self.assertIn('title: "A \\"quoted\\" title"', rendered)
        self.assertIn("archive_identifier: \"item_1\"", rendered)
        self.assertTrue(rendered.endswith("মন্ত্র।\n"))

    def test_edit_ratio(self):
        self.assertEqual(cleaner._edit_ratio("ক্ষ", "ক্ষ"), 0.0)
        self.assertEqual(cleaner._edit_ratio("ক", "খ"), 1.0)
        self.assertGreater(cleaner._edit_ratio("সাধনা", "সাধন"), 0.0)

    def test_corpus_loader_counts_bengali_words(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline.txt"
            path.write_text("সাধনা মন্ত্র সাধনা", encoding="utf-8")
            text, words = cleaner.read_corpus([path])
        self.assertIn("সাধনা", text)
        self.assertEqual(words["সাধনা"], 2)
        self.assertEqual(words["মন্ত্র"], 1)


if __name__ == "__main__":
    unittest.main()
