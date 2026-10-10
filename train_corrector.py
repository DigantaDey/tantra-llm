#!/usr/bin/env python3
"""Train a compact Bengali character denoiser from clean text + synthetic OCR noise.

This optional CPU training utility writes a local checkpoint for cleaner.py; it
never downloads a pretrained model or sends corpus text to a service.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from cleaner import (
    BENGALI_PRESENT_RE,
    BROKEN_CONJUNCTS,
    CONFUSION_GROUPS,
    _strip_markdown_metadata,
    corpus_files,
    default_terms_path,
    load_terms,
    normalize_ocr_text,
    synthetic_corrupt,
)
from neural_corrector import SPECIAL_TOKENS, build_model


def _split_samples(text: str, *, max_chars: int) -> list[str]:
    samples: list[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = re.split(r"(?<=[।॥!?])\s+", paragraph)
        for sentence in sentences:
            remainder = sentence.strip()
            while len(remainder) > max_chars:
                cut = remainder.rfind(" ", 0, max_chars + 1)
                if cut < max_chars // 2:
                    cut = max_chars
                sample = remainder[:cut].strip()
                if sample:
                    samples.append(sample)
                remainder = remainder[cut:].strip()
            if len(remainder) >= 2:
                samples.append(remainder)
    return samples


def collect_samples(
    paths: Iterable[Path],
    *,
    max_length: int,
    max_samples: int,
    terms: Iterable[str] = (),
) -> list[str]:
    """Read clean Bengali text and make bounded, context-rich training examples."""
    max_chars = max(8, max_length - 2)
    samples: list[str] = []
    for path in corpus_files(paths):
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            print(f"[warn] Unable to read {path}: {exc}", file=sys.stderr)
            continue
        text = normalize_ocr_text(_strip_markdown_metadata(text), preserve_lines=True)
        samples.extend(sample for sample in _split_samples(text, max_chars=max_chars) if BENGALI_PRESENT_RE.search(sample))
        if max_samples and len(samples) >= max_samples:
            break

    term_samples: list[str] = []
    for term in sorted(terms):
        normalized = normalize_ocr_text(term)
        if BENGALI_PRESENT_RE.search(normalized):
            term_samples.extend(_split_samples(normalized, max_chars=max_chars))

    # Reserve room for lexicon examples so a large baseline cannot crowd out
    # the domain anchors that distinguish this project from generic cleanup.
    if max_samples:
        corpus_limit = max(0, max_samples - len(term_samples))
        samples = samples[:corpus_limit]
        term_samples = term_samples[:max_samples]
    samples.extend(term_samples)

    # Deduplicate while keeping source order; repeated OCR pages should not
    # dominate training simply because an item was scanned more than once.
    samples = list(dict.fromkeys(sample for sample in samples if len(sample) >= 2))
    if max_samples:
        samples = samples[:max_samples]
    return samples


class SyntheticPairDataset:
    """Small torch-compatible dataset whose corruption changes by epoch."""

    def __init__(
        self,
        samples: Sequence[str],
        *,
        vocab: Sequence[str],
        terms: Iterable[str],
        max_length: int,
        seed: int,
        noise_scale: float,
        anchor_weight: float,
    ):
        self.samples = list(samples)
        self.char_to_id = {char: index for index, char in enumerate(vocab)}
        self.terms = tuple(sorted({term for term in terms if term}, key=len, reverse=True))
        self.max_length = max_length
        self.seed = seed
        self.epoch = 0
        self.noise_scale = max(0.0, noise_scale)
        self.anchor_weight = max(1.0, anchor_weight)

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _anchor_weights(self, clean: str) -> list[float]:
        weights = [1.0] * len(clean)
        for term in self.terms:
            start = 0
            while True:
                found = clean.find(term, start)
                if found < 0:
                    break
                for index in range(found, min(len(clean), found + len(term))):
                    weights[index] = self.anchor_weight
                start = found + max(1, len(term))
        return weights

    def __getitem__(self, index: int) -> dict[str, list[int] | list[float]]:
        clean = self.samples[index][: self.max_length - 1]
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index * 9176)
        noisy = synthetic_corrupt(
            clean,
            rng=rng,
            conjunct_rate=min(0.75, 0.08 * self.noise_scale),
            confusion_rate=min(0.25, 0.015 * self.noise_scale),
            whitespace_rate=min(0.25, 0.02 * self.noise_scale),
        )
        char_to_id = self.char_to_id
        unk_id = SPECIAL_TOKENS.index("<unk>")
        source_ids = [char_to_id.get(char, unk_id) for char in noisy[: self.max_length]]
        target_ids = [char_to_id.get(char, unk_id) for char in clean]
        weights = self._anchor_weights(clean)[: len(target_ids)]
        return {"source": source_ids, "target": target_ids, "weights": weights}


def collate_pairs(
    examples: Sequence[dict[str, list[int] | list[float]]],
    *,
    torch: Any,
    pad_id: int,
    bos_id: int,
    eos_id: int,
):
    source_values = [list(example["source"]) or [pad_id] for example in examples]
    target_values = [list(example["target"]) for example in examples]
    weight_values = [list(example["weights"]) for example in examples]
    source_length = max(map(len, source_values))
    target_length = max(len(values) + 1 for values in target_values)
    batch_size = len(examples)

    source = torch.full((batch_size, source_length), pad_id, dtype=torch.long)
    decoder_input = torch.full((batch_size, target_length), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, target_length), pad_id, dtype=torch.long)
    weights = torch.zeros((batch_size, target_length), dtype=torch.float32)

    for row, (source_ids, target_ids, target_weights) in enumerate(
        zip(source_values, target_values, weight_values)
    ):
        source[row, : len(source_ids)] = torch.tensor(source_ids, dtype=torch.long)
        decoder_values = [bos_id] + target_ids
        label_values = target_ids + [eos_id]
        row_weights = target_weights + [1.0]
        decoder_input[row, : len(decoder_values)] = torch.tensor(decoder_values, dtype=torch.long)
        labels[row, : len(label_values)] = torch.tensor(label_values, dtype=torch.long)
        weights[row, : len(row_weights)] = torch.tensor(row_weights, dtype=torch.float32)

    return source, decoder_input, labels, weights


def _read_checkpoint_safe(torch: Any, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def train(args: argparse.Namespace) -> int:
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except ImportError:
        print("PyTorch is needed to train the corrector. Install `pip install -e '.[train]'`.", file=sys.stderr)
        return 2

    if args.max_length < 16:
        print("--max-length must be at least 16", file=sys.stderr)
        return 2
    if args.model_dim < 1 or args.heads < 1 or args.model_dim % args.heads:
        print("--model-dim and --heads must be positive, and model-dim must be divisible by heads", file=sys.stderr)
        return 2
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.threads not in (2, 4)
        or args.max_samples < 0
        or args.layers < 1
        or not 0.0 <= args.dropout < 1.0
        or args.learning_rate <= 0
    ):
        print("Invalid training limits: epochs/batch/layers/heads/lr must be positive, max-samples non-negative, dropout in [0,1), and threads 2 or 4.", file=sys.stderr)
        return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; use --device cpu.", file=sys.stderr)
        return 2

    terms = load_terms(args.terms)
    samples = collect_samples(
        args.corpus,
        max_length=args.max_length,
        max_samples=args.max_samples,
        terms=terms,
    )
    if len(samples) < 2:
        print("Need at least two Bengali training samples. Supply clean UTF-8 .txt/.md files with --corpus.", file=sys.stderr)
        return 2

    vocab_chars = set("".join(samples))
    vocab_chars.update("".join(terms))
    vocab_chars.update("".join(char for group in CONFUSION_GROUPS for char in group))
    vocab_chars.update("".join(left + right + joined for left, right, joined in BROKEN_CONJUNCTS))
    vocab_chars.update(" \t\n\u09cd\u09c7\u09c8\u09cb\u09be\u09bf\u09c0\u09c1\u09c2\u09c3\u09c4\u0964\u0965")
    vocab = list(SPECIAL_TOKENS) + sorted(vocab_chars - set(SPECIAL_TOKENS))
    pad_id, bos_id, eos_id, _ = range(len(SPECIAL_TOKENS))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True

    shuffled = list(samples)
    random.shuffle(shuffled)
    if len(shuffled) >= 12:
        validation_count = max(1, min(len(shuffled) // 10, 128))
        train_samples = shuffled[:-validation_count]
        validation_samples = shuffled[-validation_count:]
    else:
        train_samples = shuffled
        validation_samples = shuffled

    train_set = SyntheticPairDataset(
        train_samples,
        vocab=vocab,
        terms=terms,
        max_length=args.max_length,
        seed=args.seed,
        noise_scale=args.noise_scale,
        anchor_weight=args.anchor_weight,
    )
    validation_set = SyntheticPairDataset(
        validation_samples,
        vocab=vocab,
        terms=terms,
        max_length=args.max_length,
        seed=args.seed + 40_009,
        noise_scale=args.noise_scale,
        anchor_weight=args.anchor_weight,
    )
    def collate(batch):
        return collate_pairs(
            batch, torch=torch, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    config = {
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "unk_id": 3,
        "max_length": args.max_length,
        "d_model": args.model_dim,
        "nhead": args.heads,
        "layers": args.layers,
        "feedforward": args.model_dim * 3,
        "dropout": args.dropout,
    }
    model = build_model(len(vocab), config).to(torch.device(args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_validation = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Training {len(train_samples):,} samples (+ {len(terms)} lexicon anchors), "
        f"vocab={len(vocab)}, device={args.device}, threads={args.threads}"
    )

    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        model.train()
        train_total = 0.0
        train_batches = 0
        for source, decoder_input, labels, weights in train_loader:
            source = source.to(args.device)
            decoder_input = decoder_input.to(args.device)
            labels = labels.to(args.device)
            weights = weights.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, decoder_input)
            losses = F.cross_entropy(
                logits.reshape(-1, len(vocab)),
                labels.reshape(-1),
                ignore_index=pad_id,
                reduction="none",
            )
            active = labels.reshape(-1).ne(pad_id)
            active_weights = weights.reshape(-1) * active
            loss = (losses * active_weights).sum() / active_weights.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += float(loss.detach().cpu())
            train_batches += 1

        model.eval()
        validation_set.set_epoch(epoch)
        validation_total = 0.0
        validation_batches = 0
        with torch.inference_mode():
            for source, decoder_input, labels, weights in validation_loader:
                source = source.to(args.device)
                decoder_input = decoder_input.to(args.device)
                labels = labels.to(args.device)
                weights = weights.to(args.device)
                logits = model(source, decoder_input)
                losses = F.cross_entropy(
                    logits.reshape(-1, len(vocab)),
                    labels.reshape(-1),
                    ignore_index=pad_id,
                    reduction="none",
                )
                active = labels.reshape(-1).ne(pad_id)
                active_weights = weights.reshape(-1) * active
                loss = (losses * active_weights).sum() / active_weights.sum().clamp_min(1.0)
                validation_total += float(loss.cpu())
                validation_batches += 1

        train_loss = train_total / max(1, train_batches)
        validation_loss = validation_total / max(1, validation_batches)
        print(
            f"epoch {epoch + 1:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}"
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            torch.save(
                {
                    "format": "tantra-char-denoiser-v1",
                    "vocab": vocab,
                    "config": config,
                    "state_dict": model.state_dict(),
                    "training": {
                        "epochs_completed": epoch + 1,
                        "samples": len(samples),
                        "seed": args.seed,
                        "best_validation_loss": best_validation,
                    },
                },
                temporary,
            )
            temporary.replace(args.output)

    print(f"Checkpoint saved: {args.output} (best validation loss {best_validation:.4f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small character-level Bengali OCR denoiser.")
    parser.add_argument("--corpus", type=Path, nargs="+", required=True, help="Clean Bengali .txt/.md file(s) or directories; modern open data is suitable as a baseline.")
    parser.add_argument("--terms", type=Path, default=default_terms_path(), help="Domain lexicon whose characters receive extra target weight.")
    parser.add_argument("--output", type=Path, default=Path("models/tantra-char-denoiser.pt"), help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-samples", type=int, default=20_000, help="Maximum number of source examples; 0 means no explicit cap.")
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--anchor-weight", type=float, default=2.5)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=2, choices=(2, 4))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    return train(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
