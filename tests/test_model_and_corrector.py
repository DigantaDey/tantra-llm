from __future__ import annotations

import math
import random

import pytest
import torch

from tantra.alignment import DELETE, KEEP
from tantra.corrector import Corrector
from tantra.dataset import Example, build_examples
from tantra.dictionary import EsotericDictionary
from tantra.model import CharVocab, ModelConfig, TagDenoiser, TagVocab
from tantra.noise import NoiseProfile
from tantra.training import (
    TrainConfig,
    edit_scores,
    encode_examples,
    label_accuracy,
    learning_rate,
    train_denoiser,
)
from tests.fakes import TINY_CONFIG as TINY
from tests.fakes import controlled_corrector

SENTENCES = [
    "তন্ত্রসাধনায় বীজমন্ত্র ও ন্যাস এবং মুদ্রা সম্পর্কে একটি দীর্ঘ বাক্য লেখা হলো।",
    "কুণ্ডলিনী শক্তি ষট্চক্র ভেদ করে সহস্রারে পৌঁছায় বলে বিশ্বাস করা হয়।",
    "শ্মশানে বসে সাধক জপ করেন এবং ভৈরবীর আশীর্বাদ প্রার্থনা করেন।",
]


def _tiny_setup(labels: list[str] | None = None):
    chars = CharVocab.from_counts({c: 3 for s in SENTENCES for c in s}, min_count=1)
    tags = TagVocab(labels or ["ক্", "স"])
    model = TagDenoiser(len(chars), len(tags), TINY)
    return chars, tags, model


# ----------------------------------------------------------------- vocab & model
def test_char_vocab_reserves_pad_and_unk():
    vocab = CharVocab.from_counts({"ক": 5, "x": 1}, min_count=2)
    ids = vocab.encode("কx")
    assert ids[0] >= 2 and ids[1] == 1  # 'x' below min_count -> UNK
    assert "\n" in vocab.to_list()
    assert CharVocab.from_list(vocab.to_list()).encode("ক") == vocab.encode("ক")


def test_tag_vocab_keep_and_delete_have_fixed_ids():
    tags = TagVocab(["স", KEEP, "ক্"])
    assert tags.id_of(KEEP) == 0 and tags.id_of(DELETE) == 1
    assert tags.id_of("স") == 2 and tags.id_of("ক্") == 3
    assert tags.id_of("unseen") is None


def test_model_shapes_and_length_guard():
    chars, tags, model = _tiny_setup()
    ids = torch.tensor([chars.encode(SENTENCES[0][:40])])
    assert model(ids).shape == (1, 40, len(tags))
    with pytest.raises(ValueError):
        model(torch.zeros((1, 65), dtype=torch.long))


def test_model_config_validation():
    chars, tags, _ = _tiny_setup()
    with pytest.raises(ValueError):
        TagDenoiser(len(chars), len(tags), ModelConfig(d_model=30, n_heads=4))


def test_padding_does_not_change_predictions():
    """A window must get the same logits whether or not it is padded in a batch (mask correctness)."""
    chars, tags, model = _tiny_setup()
    model.eval()
    a = chars.encode(SENTENCES[0][:30])
    b = chars.encode(SENTENCES[1][:50])
    with torch.inference_mode():
        alone = model(torch.tensor([a]))[0]
        padded = torch.zeros((2, 50), dtype=torch.long)
        padded[0, :30] = torch.tensor(a)
        padded[1] = torch.tensor(b)
        batched = model(padded)[0, :30]
    assert torch.allclose(alone, batched, atol=1e-5)


# ----------------------------------------------------------------- learning-rate schedule
def test_learning_rate_warms_up_then_decays():
    cfg = TrainConfig(steps=100, lr=1e-3, warmup_steps=10)
    assert learning_rate(0, cfg) < learning_rate(9, cfg) <= cfg.lr
    values = [learning_rate(s, cfg) for s in range(10, 100)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))
    assert math.isclose(learning_rate(99, cfg), cfg.lr * 0.1, rel_tol=0.05)


# ----------------------------------------------------------------- training & checkpoints
def test_training_writes_a_loadable_checkpoint_that_reproduces_predictions(tmp_path):
    chars, tags, _ = _tiny_setup(["ক্", "স", "ন্"])
    examples = build_examples(SENTENCES * 10, copies=2, profile=NoiseProfile.scaled(3), max_len=64, seed=1)
    valid = examples[:10]
    ckpt = tmp_path / "ck.pt"
    report = train_denoiser(
        examples[10:],
        valid,
        chars=chars,
        tags=tags,
        model_config=TINY,
        train_config=TrainConfig(steps=20, batch_size=8, warmup_steps=2, threads=1, eval_every=1000, eval_examples=5),
        checkpoint_path=ckpt,
        log=lambda _m: None,
    )
    assert ckpt.exists() and report.steps == 20
    assert 0.0 <= report.valid_label_accuracy <= 1.0
    reloaded = Corrector.load(ckpt)
    assert reloaded.chars.to_list() == chars.to_list()
    assert reloaded.tags.to_list() == tags.to_list()
    text = SENTENCES[2]
    out_a, _ = reloaded.correct_lines([text], use_dictionary=False)
    out_b, _ = reloaded.correct_lines([text], use_dictionary=False)
    assert out_a == out_b  # deterministic inference


def test_training_rejects_empty_input(tmp_path):
    chars, tags, _ = _tiny_setup()
    with pytest.raises(ValueError):
        train_denoiser(
            [],
            [],
            chars=chars,
            tags=tags,
            model_config=TINY,
            train_config=TrainConfig(steps=1),
            checkpoint_path=tmp_path / "x.pt",
            log=lambda _m: None,
        )


def test_load_rejects_foreign_checkpoints(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"format": "something-else"}, path)
    with pytest.raises(ValueError):
        Corrector.load(path)


# ----------------------------------------------------------------- dictionary anchor
def test_without_bonus_the_model_keeps_its_own_reading():
    dictionary = EsotericDictionary.default()
    out, stats = controlled_corrector(0.0, dictionary).correct_lines(["ন্যাশ"])
    assert out == ["ন্যাশ"] and stats.dictionary_anchors == 0


def test_dictionary_bonus_restores_the_esoteric_term():
    dictionary = EsotericDictionary.default()
    out, stats = controlled_corrector(3.0, dictionary).correct_lines(["ন্যাশ"])
    assert out == ["ন্যাস"]
    assert stats.dictionary_anchors == 1 and stats.anchored_terms == {"ন্যাস": 1}


def test_dictionary_can_be_switched_off_per_call():
    dictionary = EsotericDictionary.default()
    out, _ = controlled_corrector(3.0, dictionary).correct_lines(["ন্যাশ"], use_dictionary=False)
    assert out == ["ন্যাশ"]


def test_words_far_from_the_lexicon_are_never_touched():
    dictionary = EsotericDictionary.default()
    corrector = controlled_corrector(3.0, dictionary)
    out, stats = corrector.correct_lines(["শশ নশ"])
    assert stats.dictionary_anchors == 0
    assert out == ["শশ নশ"]


def test_correct_lines_keeps_line_structure_and_spacing():
    chars, tags, model = _tiny_setup()
    corrector = Corrector(model, chars, tags, None)
    lines = ["", SENTENCES[0], "  ", SENTENCES[1] + " "]
    out, stats = corrector.correct_lines(lines, use_dictionary=False)
    assert len(out) == len(lines)
    assert out[0] == "" and stats.lines == 4


def test_random_inputs_never_crash_and_keep_length_for_identity_model():
    chars, tags, model = _tiny_setup()
    corrector = Corrector(model, chars, tags, EsotericDictionary.default())
    rng = random.Random(0)
    lines = ["".join(rng.choice("কখগমন্যাসশ ") for _ in range(rng.randint(1, 120))) for _ in range(20)]
    out, _ = corrector.correct_lines(lines)
    assert len(out) == len(lines)


# ----------------------------------------------------------------- edit recall (non-KEEP)
class _ScriptedTagger(torch.nn.Module):
    """Predicts a fixed tag per position: row b of the batch gets rows[b] (one batch only)."""

    def __init__(self, rows: list[list[int]], n_tags: int) -> None:
        super().__init__()
        self.rows = rows
        self.n_tags = n_tags

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out = torch.full((ids.shape[0], ids.shape[1], self.n_tags), -1e4)
        for b, row in enumerate(self.rows):
            for t, tag in enumerate(row):
                out[b, t, tag] = 0.0
        return out


def _edit_examples() -> tuple[CharVocab, TagVocab, list[Example]]:
    """Two examples, 8 labels in total: 3 gold edits and 5 KEEP labels. Tags: KEEP=0, DELETE=1, স=2, ক্=3."""
    tags = TagVocab(["স", "ক্"])
    chars = CharVocab.from_counts({c: 1 for c in "কখগঘঙচ"}, min_count=1)
    examples = [
        Example(source="কখগঘ", clean="কখগঘ", labels=(KEEP, "স", KEEP, "ক্")),  # 2 gold edits
        Example(source="কখঙচ", clean="কখঙচ", labels=(KEEP, KEEP, KEEP, "স")),  # 1 gold edit
    ]
    return chars, tags, examples


def test_edit_scores_count_gold_edits_and_predicted_edits():
    chars, tags, examples = _edit_examples()
    encoded = encode_examples(examples, chars, tags)
    assert [labels for _, labels in encoded] == [[0, 2, 0, 3], [0, 0, 0, 2]]

    all_keep = edit_scores(_ScriptedTagger([[0, 0, 0, 0], [0, 0, 0, 0]], len(tags)), encoded, tags)
    assert (all_keep.recall, all_keep.gold_edits, all_keep.predicted_edits) == (0.0, 3, 0)
    assert math.isnan(all_keep.precision)  # no edits made, so precision is undefined

    one_missed = edit_scores(_ScriptedTagger([[0, 2, 0, 0], [0, 0, 0, 2]], len(tags)), encoded, tags)
    assert one_missed.recall == pytest.approx(2 / 3) and one_missed.precision == 1.0

    perfect = edit_scores(_ScriptedTagger([[0, 2, 0, 3], [0, 0, 0, 2]], len(tags)), encoded, tags)
    assert (perfect.recall, perfect.precision) == (1.0, 1.0)


def test_over_editing_shows_up_as_low_precision_not_high_recall():
    """Recall alone rewards editing everywhere; precision exposes it."""
    chars, tags, examples = _edit_examples()
    encoded = encode_examples(examples, chars, tags)
    everywhere = edit_scores(_ScriptedTagger([[2, 2, 2, 2], [2, 2, 2, 2]], len(tags)), encoded, tags)
    assert everywhere.recall == pytest.approx(2 / 3)  # hits: the 'স' edit in ex1, and the 'স' edit in ex2
    assert everywhere.predicted_edits == 8 and everywhere.precision == pytest.approx(2 / 8)


def test_all_keep_model_scores_well_on_accuracy_but_zero_on_edit_recall():
    """The reason this metric exists: plain label accuracy rewards doing nothing."""
    chars, tags, examples = _edit_examples()
    encoded = encode_examples(examples, chars, tags)
    all_keep = _ScriptedTagger([[0, 0, 0, 0], [0, 0]], len(tags))
    # 8 labels: the 5 KEEP labels are right, the 3 edits are missed.
    assert label_accuracy(all_keep, encoded) == pytest.approx(5 / 8)
    assert edit_scores(all_keep, encoded, tags).recall == 0.0


def test_edit_weight_is_recorded_in_the_checkpoint(tmp_path):
    """A checkpoint must say how it was trained, so results can be traced back to the loss setting."""
    chars, tags, _ = _tiny_setup(["ক্", "স"])
    examples = build_examples(SENTENCES * 4, copies=1, profile=NoiseProfile.scaled(3), max_len=64, seed=2)
    ckpt = tmp_path / "weighted.pt"
    train_denoiser(
        examples,
        examples[:4],
        chars=chars,
        tags=tags,
        model_config=TINY,
        train_config=TrainConfig(
            steps=4, batch_size=8, warmup_steps=1, threads=1, eval_every=1000, eval_examples=4, edit_weight=5.0
        ),
        checkpoint_path=ckpt,
        log=lambda _m: None,
    )
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved["train_config"]["edit_weight"] == 5.0


# ------------------------------------------------------- train/eval mode regressions
DROPOUT = ModelConfig(d_model=32, n_heads=2, n_layers=1, max_len=64, dropout=0.5)


def test_scoring_helpers_leave_the_model_in_the_mode_they_found_it():
    chars, tags, _ = _tiny_setup(["ক্", "স"])
    model = TagDenoiser(len(chars), len(tags), DROPOUT)
    examples = build_examples(SENTENCES * 2, copies=1, profile=NoiseProfile.scaled(3), max_len=64, seed=3)
    encoded = encode_examples(examples, chars, tags)
    model.train()
    label_accuracy(model, encoded)
    edit_scores(model, encoded, tags)
    assert model.training
    model.eval()
    label_accuracy(model, encoded)
    edit_scores(model, encoded, tags)
    assert not model.training


def test_inference_is_unchanged_by_scoring_that_runs_first():
    """Regression: scoring used to leave dropout on, so later corrections drifted from run to run."""
    chars, tags, _ = _tiny_setup(["ক্", "স"])
    model = TagDenoiser(len(chars), len(tags), DROPOUT)
    examples = build_examples(SENTENCES * 2, copies=1, profile=NoiseProfile.scaled(3), max_len=64, seed=3)
    encoded = encode_examples(examples, chars, tags)
    ids = torch.tensor([chars.encode(SENTENCES[0])[:32]])
    Corrector(model, chars, tags, None)  # puts the model in eval mode, as inference requires
    with torch.inference_mode():
        before = model(ids)
    edit_scores(model, encoded, tags)
    with torch.inference_mode():
        after = model(ids)
    assert torch.equal(before, after)


def test_training_keeps_dropout_on_after_interim_checkpoints(tmp_path, monkeypatch):
    """Regression: an interim checkpoint used to switch training to eval mode for the rest of the run."""
    chars, tags, _ = _tiny_setup(["ক্", "স"])
    examples = build_examples(SENTENCES * 10, copies=2, profile=NoiseProfile.scaled(3), max_len=64, seed=4)
    training_modes: list[bool] = []
    original_forward = TagDenoiser.forward

    def recording_forward(self, ids):
        if torch.is_grad_enabled():
            training_modes.append(self.training)
        return original_forward(self, ids)

    monkeypatch.setattr(TagDenoiser, "forward", recording_forward)
    train_denoiser(
        examples[10:],
        examples[:10],
        chars=chars,
        tags=tags,
        model_config=DROPOUT,
        train_config=TrainConfig(steps=8, batch_size=8, warmup_steps=1, threads=1, eval_every=2, eval_examples=4),
        checkpoint_path=tmp_path / "interim.pt",
        log=lambda _m: None,
    )
    assert len(training_modes) == 8
    assert all(training_modes)
