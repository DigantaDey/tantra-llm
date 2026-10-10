"""Training loop for :class:`tantra.model.TagDenoiser` (CPU-friendly, deterministic, checkpointed)."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from tantra.alignment import KEEP, is_trainable_label
from tantra.corrector import Corrector
from tantra.dataset import Example
from tantra.model import IGNORE_INDEX, PAD_ID, CharVocab, ModelConfig, TagDenoiser, TagVocab
from tantra.textmetrics import levenshtein

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    batch_size: int = 64
    lr: float = 1e-3
    warmup_steps: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 13
    threads: int = 2
    eval_every: int = 500
    eval_examples: int = 200
    # Loss weight for every non-KEEP label. Edits are ~3% of labels, so unweighted cross-entropy
    # learns the KEEP prior and copies its input (see README, "Results"). 1.0 disables weighting.
    edit_weight: float = 1.0


@dataclass(frozen=True)
class TrainReport:
    steps: int
    train_examples: int
    valid_examples: int
    final_loss: float
    valid_label_accuracy: float
    valid_cer_noisy: float
    valid_cer_corrected: float
    seconds: float
    parameters: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def learning_rate(step: int, cfg: TrainConfig) -> float:
    """Linear warm-up, then cosine decay to 10% of the peak rate."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps))
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def encode_examples(examples: Sequence[Example], chars: CharVocab, tags: TagVocab) -> list[tuple[list[int], list[int]]]:
    """Character ids and label ids per example. Labels outside the tag vocabulary become IGNORE_INDEX."""
    encoded: list[tuple[list[int], list[int]]] = []
    for ex in examples:
        ids = chars.encode(ex.source)
        labels = []
        for label in ex.labels:
            tid = tags.id_of(label) if is_trainable_label(label) else None
            labels.append(IGNORE_INDEX if tid is None else tid)
        encoded.append((ids, labels))
    return encoded


def _collate(items: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(ids) for ids, _ in items)
    ids_t = torch.full((len(items), width), PAD_ID, dtype=torch.long)
    lab_t = torch.full((len(items), width), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, labels) in enumerate(items):
        ids_t[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        lab_t[row, : len(labels)] = torch.tensor(labels, dtype=torch.long)
    return ids_t, lab_t


def label_accuracy(model: TagDenoiser, encoded: list[tuple[list[int], list[int]]], batch: int = 64) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoded), batch):
            ids, labels = _collate(encoded[start : start + batch])
            pred = model(ids).argmax(dim=-1)
            mask = labels.ne(IGNORE_INDEX)
            correct += int((pred.eq(labels) & mask).sum())
            total += int(mask.sum())
    model.train()
    return correct / total if total else 0.0


@dataclass(frozen=True)
class EditScores:
    """How well the tagger finds and makes the edits that are needed.

    Positions are counted only where the gold label is in the tag vocabulary. ``recall`` is over gold edits,
    ``precision`` is over predicted edits. A model that edits everywhere has high recall and low precision.
    """

    recall: float
    precision: float
    gold_edits: int
    predicted_edits: int


def edit_scores(
    model: TagDenoiser, encoded: list[tuple[list[int], list[int]]], tags: TagVocab, batch: int = 64
) -> EditScores:
    keep_id = tags.id_of(KEEP)
    hits = gold = predicted = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoded), batch):
            ids, labels = _collate(encoded[start : start + batch])
            pred = model(ids).argmax(dim=-1)
            known = labels.ne(IGNORE_INDEX)
            gold_edit = known & labels.ne(keep_id)
            pred_edit = known & pred.ne(keep_id)
            hits += int((pred.eq(labels) & gold_edit).sum())
            gold += int(gold_edit.sum())
            predicted += int(pred_edit.sum())
    model.train()
    recall = hits / gold if gold else float("nan")
    precision = hits / predicted if predicted else float("nan")
    return EditScores(recall=recall, precision=precision, gold_edits=gold, predicted_edits=predicted)


def evaluate_cer(corrector: Corrector, examples: Sequence[Example]) -> tuple[float, float]:
    """Mean CER of the noisy input and of the model's greedy correction, vs. clean text."""
    if not examples:
        return 0.0, 0.0
    sources = [ex.source for ex in examples]
    cleans = [ex.clean for ex in examples]
    corrected, _ = corrector.correct_lines(sources, use_dictionary=False)
    total = sum(len(c) for c in cleans)
    noisy_err = sum(levenshtein(c, s) for c, s in zip(cleans, sources, strict=True))
    fixed_err = sum(levenshtein(c, h) for c, h in zip(cleans, corrected, strict=True))
    return noisy_err / total, fixed_err / total


def train_denoiser(
    train_examples: Sequence[Example],
    valid_examples: Sequence[Example],
    *,
    chars: CharVocab,
    tags: TagVocab,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_path: str | Path,
    log: LogFn = print,
) -> TrainReport:
    """Train, evaluate and write a checkpoint. Returns a summary report."""
    if not train_examples:
        raise ValueError("no training examples: is the corpus Bengali enough?")
    torch.set_num_threads(train_config.threads)
    torch.manual_seed(train_config.seed)
    rng = random.Random(train_config.seed)

    train_enc = encode_examples(train_examples, chars, tags)
    valid_enc = encode_examples(valid_examples, chars, tags)
    model = TagDenoiser(len(chars), len(tags), model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    class_weight = torch.ones(len(tags))
    class_weight[tags.id_of(KEEP)] = 1.0
    class_weight[torch.arange(len(tags)) != tags.id_of(KEEP)] = float(train_config.edit_weight)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight, ignore_index=IGNORE_INDEX)
    log(f"model: {model.num_parameters() / 1e6:.2f}M parameters, {len(tags)} edit labels, {len(chars)} characters")

    started = time.time()
    running = float("nan")
    model.train()
    for step in range(train_config.steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, train_config)
        batch = [train_enc[rng.randrange(len(train_enc))] for _ in range(train_config.batch_size)]
        ids, labels = _collate(batch)
        logits = model(ids)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        value = float(loss.detach())
        running = value if math.isnan(running) else 0.98 * running + 0.02 * value

        if (step + 1) % 50 == 0 or step == 0:
            elapsed = time.time() - started
            lr_now = optimizer.param_groups[0]["lr"]
            log(f"step {step + 1:>6}/{train_config.steps}  loss {running:.4f}  lr {lr_now:.2e}  {elapsed:6.1f}s")
        if valid_enc and (step + 1) % train_config.eval_every == 0 and step + 1 < train_config.steps:
            log(f"  validation label accuracy: {label_accuracy(model, valid_enc):.4f}")
            # Interim checkpoint: a long run should not be lost to an interruption.
            _save(Corrector(model, chars, tags, None), checkpoint_path, {"interim_step": step + 1})

    seconds = time.time() - started
    model.eval()
    acc = label_accuracy(model, valid_enc) if valid_enc else 0.0
    probe = list(valid_examples[: train_config.eval_examples])
    noisy_cer, fixed_cer = evaluate_cer(Corrector(model, chars, tags, None), probe)
    report = TrainReport(
        steps=train_config.steps,
        train_examples=len(train_examples),
        valid_examples=len(valid_examples),
        final_loss=running,
        valid_label_accuracy=acc,
        valid_cer_noisy=noisy_cer,
        valid_cer_corrected=fixed_cer,
        seconds=seconds,
        parameters=model.num_parameters(),
    )
    log(f"validation: label accuracy {acc:.4f} | CER noisy {noisy_cer:.4f} -> corrected {fixed_cer:.4f}")
    _save(
        Corrector(model, chars, tags, None),
        checkpoint_path,
        {"train_config": asdict(train_config), "report": report.as_dict()},
    )
    log(f"checkpoint written: {checkpoint_path}")
    return report


def _save(corrector: Corrector, path: str | Path, extra: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(corrector.checkpoint(extra=extra), tmp)
    tmp.replace(target)  # atomic: a crash never leaves a half-written checkpoint
