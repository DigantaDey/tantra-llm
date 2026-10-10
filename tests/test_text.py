from __future__ import annotations

import random

import pytest

from tantra.alignment import DELETE, KEEP, align_replacements, apply_labels, to_labels
from tantra.bengali import bengali_ratio, normalize, word_spans
from tantra.dataset import clean_windows, make_windows
from tantra.noise import CONFUSABLE, NoiseProfile, corrupt
from tantra.textmetrics import cer, levenshtein

BENGALI_ALPHABET = list("কখগঘচছজঝটঠডঢতথদধনপফবভমযরলশষসহ") + [
    "্",
    "া",
    "ি",
    "ী",
    "ু",
    "ে",
    "ো",
    " ",
    "।",
    "ং",
    "ঁ",
    "ড়",
]


def _random_text(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(BENGALI_ALPHABET + list("ab1 ")) for _ in range(n))


# ----------------------------------------------------------------- normalisation
def test_nukta_sequences_are_composed_although_nfc_misses_them():
    # NFC does not compose these (composition exclusions); we must.
    assert normalize("\u09a1\u09bc") == "\u09dc\n"  # ড়
    assert normalize("\u09a2\u09bc") == "\u09dd\n"  # ঢ়
    assert normalize("\u09af\u09bc") == "\u09df\n"  # য়


def test_zero_width_joiners_are_kept_but_bom_and_zwsp_removed():
    text = "ক\u200dষ\ufeff\u200b"
    assert normalize(text) == "ক\u200dষ\n"


def test_whitespace_is_collapsed_and_blank_runs_limited():
    assert normalize("  একটি\t\tদুটি  \r\n\r\n\r\n\r\nতিন") == "একটি দুটি\n\nতিন\n"


def test_normalize_is_idempotent_and_handles_empty():
    rng = random.Random(1)
    for _ in range(50):
        t = _random_text(rng, 40) + "\n\n" + _random_text(rng, 10)
        once = normalize(t)
        assert normalize(once) == once
    assert normalize("   \n\n ") == ""


def test_bengali_ratio():
    assert bengali_ratio("সাধনা") == 1.0
    assert bengali_ratio("abc") == 0.0
    assert bengali_ratio("") == 0.0
    assert 0.4 < bengali_ratio("সাধনা abcd") < 0.6


def test_word_spans_exclude_danda_and_punctuation():
    text = "বীজমন্ত্র, ন্যাস। মুদ্রা"
    spans = word_spans(text)
    assert [text[s:e] for s, e in spans] == ["বীজমন্ত্র", "ন্যাস", "মুদ্রা"]


# ----------------------------------------------------------------- alignment
def test_conjunct_break_is_labelled_as_hasanta_insertion():
    # "ক ষ" (space instead of hasanta) must be repaired by writing the hasanta over the space.
    assert to_labels("ক ষ", "ক্ষ") == [KEEP, "\u09cd", KEEP]


def test_spurious_space_is_labelled_as_delete():
    assert to_labels("সাধ না", "সাধনা") == [KEEP, KEEP, KEEP, DELETE, KEEP, KEEP]


def test_empty_source_cases():
    assert align_replacements("", "") == []
    assert align_replacements("", "x") is None


def test_apply_labels_checks_length():
    with pytest.raises(ValueError):
        apply_labels("abc", [KEEP])


@pytest.mark.parametrize("seed", range(300))
def test_alignment_round_trip_property(seed):
    """For any source/target pair, applying the labels must reproduce the target exactly."""
    rng = random.Random(seed)
    src = _random_text(rng, rng.randint(1, 30))
    tgt = _random_text(rng, rng.randint(0, 30))
    labels = to_labels(src, tgt)
    assert labels is not None
    assert apply_labels(src, labels) == tgt


# ----------------------------------------------------------------- noise
def test_noise_is_deterministic_for_a_seed():
    text = "তন্ত্রসাধনায় বীজমন্ত্র ও ন্যাস এবং মুদ্রা\nদ্বিতীয় লাইন"
    a = corrupt(text, random.Random(5), NoiseProfile.scaled(3))
    b = corrupt(text, random.Random(5), NoiseProfile.scaled(3))
    assert a == b


def test_zero_noise_is_identity():
    text = "তন্ত্রসাধনায় বীজমন্ত্র\nদ্বিতীয়"
    zero = NoiseProfile(conjunct_break=0, confusion=0, space_split=0, fade=0, drop=0, bleed=0)
    assert corrupt(text, random.Random(0), zero) == text


def test_noise_preserves_line_structure_and_changes_text():
    text = "\n".join(["ক্ষমা তন্ত্রসাধনায় সাধক ভৈরব কালী শক্তি মন্ত্র"] * 20)
    out = corrupt(text, random.Random(2), NoiseProfile.scaled(4))
    assert out.count("\n") == text.count("\n")
    assert out != text


def test_confusion_table_is_symmetric():
    for a, options in CONFUSABLE.items():
        for b in options:
            assert a in CONFUSABLE[b]
    assert "ব" in CONFUSABLE["র"] and "ক" in CONFUSABLE["ব"]


def test_profile_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        NoiseProfile(drop=1.5)


# ----------------------------------------------------------------- windows & datasets
@pytest.mark.parametrize("max_len", [2, 5, 17, 160])
def test_make_windows_reconstructs_the_line(max_len):
    line = "তন্ত্র সাধনায় বীজমন্ত্র ও ন্যাস এবং মুদ্রা অতি প্রাচীন ঐতিহ্য"
    windows = make_windows(line, max_len)
    assert "".join(windows) == line
    assert all(len(w) <= max_len for w in windows)


def test_make_windows_rejects_tiny_max_len():
    with pytest.raises(ValueError):
        make_windows("abc", 1)


def test_clean_windows_skip_latin_and_short_lines():
    lines = [
        "This is an English sentence that is long enough to pass the length check." * 2,
        "সাধন",
        "তন্ত্রসাধনায় বীজমন্ত্র ও ন্যাস এবং মুদ্রা সম্পর্কে একটি দীর্ঘ বাক্য লেখা হলো।",
    ]
    out = list(clean_windows(lines, max_len=60))
    assert out and all(bengali_ratio(w) > 0.6 for w in out)


def test_dataset_labels_always_reconstruct_clean_text():
    from tantra.dataset import build_examples

    windows = [
        "তন্ত্রসাধনায় বীজমন্ত্র ও ন্যাস এবং মুদ্রা সম্পর্কে একটি দীর্ঘ বাক্য।",
        "কুণ্ডলিনী শক্তি ষট্চক্র ভেদ করে সহস্রারে পৌঁছায় বলে বিশ্বাস করা হয়।",
    ]
    examples = build_examples(windows * 25, copies=2, profile=NoiseProfile.scaled(2), max_len=160, seed=3)
    assert len(examples) > 50
    for ex in examples:
        assert apply_labels(ex.source, list(ex.labels)) == ex.clean
        assert len(ex.labels) == len(ex.source)


# ----------------------------------------------------------------- metrics
def test_levenshtein_known_values():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("ক্ষ", "কষ") == 1
    assert levenshtein("same", "same") == 0


@pytest.mark.parametrize("seed", range(100))
def test_levenshtein_matches_naive_dp(seed):
    rng = random.Random(seed)
    a = "".join(rng.choice("abc") for _ in range(rng.randint(0, 12)))
    b = "".join(rng.choice("abc") for _ in range(rng.randint(0, 12)))
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    assert levenshtein(a, b) == dp[m][n]
    capped = levenshtein(a, b, max_dist=2)
    assert capped == min(dp[m][n], 3)


def test_cer():
    assert cer("abcd", "abcd") == 0
    assert cer("abcd", "abce") == 0.25
    assert cer("", "") == 0
