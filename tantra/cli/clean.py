"""``cleaner.py`` command line: train the Bengali OCR denoiser and correct archive texts."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from rich import box
from rich.table import Table

from tantra import ui
from tantra.bengali import bengali_ratio, word_spans
from tantra.config import (
    BASELINE_DIR,
    CALIBRATION_DIR,
    DENOISER_CHECKPOINT,
    PROCESSED_DIR,
    RAW_DIR,
)
from tantra.corrector import Corrector
from tantra.dataset import build_examples, char_counts, clean_windows, label_counts, split_is_validation
from tantra.dictionary import EsotericDictionary
from tantra.markdown import DocumentMeta
from tantra.model import CharVocab, ModelConfig, TagVocab
from tantra.noise import NoiseProfile
from tantra.pipeline import correct_document
from tantra.textmetrics import levenshtein
from tantra.training import TrainConfig, edit_scores, encode_examples, train_denoiser


# ----------------------------------------------------------------- helpers
def iter_corpus_lines(paths: Iterable[Path]) -> Iterator[str]:
    for path in paths:
        files = sorted(path.rglob("*.txt")) if path.is_dir() else [path]
        for file in files:
            with file.open(encoding="utf-8", errors="replace") as fh:
                yield from fh


def _corpus_paths(values: list[str]) -> list[Path]:
    paths = [Path(v) for v in values] or [BASELINE_DIR]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            f"corpus not found: {', '.join(missing)}\n"
            "  put clean modern Bengali .txt files in data/baseline/ (see README: 'Clean baseline')"
        )
    return paths


def _load_dictionary(path: str | None) -> EsotericDictionary:
    return EsotericDictionary.from_tsv(path) if path else EsotericDictionary.default()


def _meta_for(raw_dir: Path, fallback_id: str) -> DocumentMeta:
    meta_path = raw_dir / "metadata.json"
    data: dict = {}
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    ident = str(data.get("identifier", fallback_id))
    return DocumentMeta(
        identifier=ident,
        title=str(data.get("title", ident)),
        year=data.get("year"),
        languages=tuple(data.get("languages", [])),
        licenseurl=str(data.get("licenseurl", "")),
        source_url=str(data.get("source_url", f"https://archive.org/details/{ident}")),
    )


def _text_jobs(input_path: Path) -> list[tuple[Path, str, DocumentMeta]]:
    """Resolve ``--input`` to ``(text file, identifier, metadata)`` triples."""
    if input_path.is_file():
        return [(input_path, input_path.stem, _meta_for(input_path.parent, input_path.stem))]
    jobs = []
    for txt in sorted(input_path.glob("*/*_djvu.txt")):
        jobs.append((txt, txt.parent.name, _meta_for(txt.parent, txt.parent.name)))
    if not jobs:
        raise SystemExit(f"no *_djvu.txt files found under {input_path}; run scraper.py download first")
    return jobs


# ----------------------------------------------------------------- commands
def cmd_train(args: argparse.Namespace) -> int:
    ui.banner("Train the Bengali OCR denoiser")
    corpus_paths = _corpus_paths(args.corpus)
    dictionary = _load_dictionary(args.dictionary)
    profile = NoiseProfile.scaled(args.noise)
    model_cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        max_len=args.max_len,
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=min(args.warmup, max(1, args.steps // 4)),
        threads=args.threads,
        seed=args.seed,
        eval_every=max(1, args.steps // 4),
        edit_weight=args.edit_weight,
    )

    ui.step("Clean corpus")
    windows = list(clean_windows(iter_corpus_lines(corpus_paths), args.max_len))
    if not windows:
        raise SystemExit("no usable Bengali windows in the corpus (need Bengali lines of 24+ characters)")
    train_w = [w for w in windows if not split_is_validation(w, args.valid_fraction)]
    valid_w = [w for w in windows if split_is_validation(w, args.valid_fraction)]
    ui.ok(f"{len(windows):,} clean windows ({len(train_w):,} train / {len(valid_w):,} validation)")

    ui.step("Synthetic degradation")
    boost = frozenset(term.text for term in dictionary)
    train_ex = build_examples(
        train_w,
        copies=args.copies,
        profile=profile,
        max_len=args.max_len,
        seed=args.seed,
        boost_terms=boost if args.boost_dictionary else frozenset(),
    )
    valid_ex = build_examples(valid_w, copies=1, profile=profile, max_len=args.max_len, seed=args.seed + 1)
    ui.ok(f"{len(train_ex):,} training pairs · {len(valid_ex):,} validation pairs (noise ×{args.noise:g})")

    ui.step("Vocabularies")
    chars = CharVocab.from_counts(char_counts(windows), min_count=2)
    tags = TagVocab([label for label, _ in label_counts(train_ex).most_common(args.max_tags)])
    ui.ok(f"{len(chars)} characters · {len(tags)} edit labels")

    ui.step("Training")
    report = train_denoiser(
        train_ex,
        valid_ex,
        chars=chars,
        tags=tags,
        model_config=model_cfg,
        train_config=train_cfg,
        checkpoint_path=args.out,
        log=lambda m: ui.console.print(f"  [dim]{m}[/]"),
    )
    table = ui.kv_table(
        [
            ("parameters", f"{report.parameters / 1e6:.2f} M"),
            ("steps", f"{report.steps:,}"),
            ("training time", f"{report.seconds / 60:.1f} min"),
            ("validation label accuracy", f"{report.valid_label_accuracy:.4f}"),
            ("CER on noisy validation", f"{report.valid_cer_noisy:.4f}"),
            ("CER after correction", f"{report.valid_cer_corrected:.4f}"),
            ("checkpoint", str(args.out)),
        ]
    )
    ui.step("Result")
    ui.console.print(table)
    return 0


def _corrector(args: argparse.Namespace) -> Corrector:
    dictionary = None if getattr(args, "no_dictionary", False) else _load_dictionary(args.dictionary)
    return Corrector.load(args.checkpoint, dictionary, dict_bonus=args.dict_bonus)


def cmd_correct(args: argparse.Namespace) -> int:
    ui.banner("Correct archive OCR into clean Markdown")
    corrector = _corrector(args)
    jobs = _text_jobs(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ui.step(f"{len(jobs)} document(s) → {out_dir}")
    table = Table(box=box.SIMPLE_HEAD, header_style=f"bold {ui.ACCENT}", expand=True)
    for col in ("Document", "Pages", "Lines fixed", "Windows changed", "Dictionary anchors", "Seconds"):
        table.add_column(col, justify="left" if col == "Document" else "right")
    for txt_path, ident, meta in jobs:
        started = time.time()
        raw = txt_path.read_text(encoding="utf-8", errors="replace")
        if bengali_ratio(raw) < 0.2:
            ui.warn(f"{ident}: text is not mostly Bengali, skipped")
            continue
        result = correct_document(raw, corrector, meta, use_dictionary=not args.no_dictionary)
        (out_dir / f"{ident}.md").write_text(result.markdown, encoding="utf-8")
        (out_dir / f"{ident}.txt").write_text(result.plain_text, encoding="utf-8")
        table.add_row(
            ident,
            str(result.pages),
            f"{result.lines_corrected:,}",
            f"{result.stats.changed_windows:,}",
            f"{result.stats.dictionary_anchors:,}",
            f"{time.time() - started:.1f}",
        )
    ui.console.print(table)
    return 0


def cmd_calibration(args: argparse.Namespace) -> int:
    ui.banner("Build the calibration corpus for i-matrix quantization")
    src = Path(args.processed)
    files = sorted(src.glob("*.txt"))
    if not files:
        raise SystemExit(f"no processed .txt files in {src}; run `cleaner.py correct` first")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    chars = 0
    with out.open("w", encoding="utf-8") as fh:
        for f in files:
            text = f.read_text(encoding="utf-8")
            fh.write(text.strip() + "\n\n")
            chars += len(text)
    ui.ok(f"{len(files)} document(s), {chars:,} characters → {out}")
    if chars < 50_000:
        ui.warn("calibration set is small; i-matrix quality improves with more text (aim for >100k chars)")
    return 0


def _finite_or_none(value: float) -> float | None:
    """JSON has no NaN: report 'not measurable' (e.g. no dictionary terms at this noise level) as null."""
    return None if math.isnan(value) else round(value, 6)


def cmd_evaluate(args: argparse.Namespace) -> int:
    ui.banner("Evaluate correction quality on held-out synthetic OCR")
    corrector_on = _corrector(args)
    corrector_off = Corrector(corrector_on.model, corrector_on.chars, corrector_on.tags, None)
    dictionary = _load_dictionary(args.dictionary)
    windows = list(clean_windows(iter_corpus_lines(_corpus_paths(args.corpus)), corrector_on.max_len))
    held = [w for w in windows if split_is_validation(w, args.valid_fraction)] or windows
    held = held[: args.max_windows]
    rng = random.Random(args.seed)
    records: list[dict[str, float | int | None]] = []
    for scale in args.noise:
        profile = NoiseProfile.scaled(scale)
        examples = build_examples(
            held, copies=1, profile=profile, max_len=corrector_on.max_len, seed=rng.randrange(1 << 30)
        )
        sources = [e.source for e in examples]
        cleans = [e.clean for e in examples]
        total = sum(len(c) for c in cleans)
        noisy_err = sum(levenshtein(c, s) for c, s in zip(cleans, sources, strict=True))
        off_out, _ = corrector_off.correct_lines(sources, use_dictionary=False)
        on_out, _ = corrector_on.correct_lines(sources, use_dictionary=True)
        off_err = sum(levenshtein(c, h) for c, h in zip(cleans, off_out, strict=True))
        on_err = sum(levenshtein(c, h) for c, h in zip(cleans, on_out, strict=True))
        encoded = encode_examples(examples, corrector_on.chars, corrector_on.tags)
        scores = edit_scores(corrector_on.model, encoded, corrector_on.tags)
        records.append(
            {
                "noise": scale,
                "windows": len(examples),
                "cer_noisy": _finite_or_none(noisy_err / total),
                "cer_model": _finite_or_none(off_err / total),
                "cer_dictionary": _finite_or_none(on_err / total),
                "edit_recall_model": _finite_or_none(scores.recall),
                "edit_precision_model": _finite_or_none(scores.precision),
                "gold_edit_positions": scores.gold_edits,
                "predicted_edit_positions": scores.predicted_edits,
                "term_recall_model": _finite_or_none(_term_recall(cleans, off_out, dictionary)),
                "term_recall_dictionary": _finite_or_none(_term_recall(cleans, on_out, dictionary)),
            }
        )

    def fmt(value: float | None, digits: int) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    table = Table(box=box.SIMPLE_HEAD, header_style=f"bold {ui.ACCENT}", expand=True)
    for col in (
        "Noise ×",
        "CER noisy",
        "CER model",
        "CER +dictionary",
        "Edit recall",
        "Edit precision",
        "Term recall model",
        "Term recall +dict",
    ):
        table.add_column(col, justify="right")
    for r in records:
        table.add_row(
            f"{r['noise']:g}",
            fmt(r["cer_noisy"], 4),
            fmt(r["cer_model"], 4),
            fmt(r["cer_dictionary"], 4),
            fmt(r["edit_recall_model"], 3),
            fmt(r["edit_precision_model"], 3),
            fmt(r["term_recall_model"], 3),
            fmt(r["term_recall_dictionary"], 3),
        )
    ui.console.print(table)
    ui.info(
        "Edit recall: share of needed edits made exactly. Edit precision: share of edits made that are right. "
        "Term recall: share of lexicon words restored exactly."
    )
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        ui.info(f"metrics written to {args.json}")
    return 0


def _term_recall(cleans: list[str], outputs: list[str], dictionary: EsotericDictionary) -> float:
    """Share of dictionary-term occurrences in the clean text that the corrector restores exactly."""
    expected = 0
    restored = 0
    for clean, out in zip(cleans, outputs, strict=True):
        clean_words = [clean[s:e] for s, e in word_spans(clean)]
        out_words = [out[s:e] for s, e in word_spans(out)]
        out_set = set(out_words)
        for w in clean_words:
            if w in dictionary:
                expected += 1
                restored += w in out_set
    return restored / expected if expected else float("nan")


# ----------------------------------------------------------------- parser
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cleaner.py", description="Bengali OCR correction for Tantric archive texts.")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train the denoiser on a clean Bengali corpus")
    t.add_argument(
        "--corpus", nargs="*", default=[], help=f"clean .txt file(s) or directories (default: {BASELINE_DIR})"
    )
    t.add_argument("--out", type=Path, default=DENOISER_CHECKPOINT)
    t.add_argument("--dictionary", help="custom TSV lexicon (default: built-in Tantra lexicon)")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch-size", type=int, default=48)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--seed", type=int, default=13)
    t.add_argument("--copies", type=int, default=2, help="corrupted copies per clean window")
    t.add_argument("--noise", type=float, default=1.0, help="noise intensity multiplier (1.0 = default profile)")
    t.add_argument(
        "--edit-weight",
        type=float,
        default=1.0,
        help="loss weight on non-KEEP labels (1.0 = unweighted; ~5 counters the 97%% KEEP prior)",
    )
    t.add_argument("--valid-fraction", type=float, default=0.05)
    t.add_argument("--max-len", type=int, default=160, help="window length in characters")
    t.add_argument("--max-tags", type=int, default=1024)
    t.add_argument("--d-model", type=int, default=128)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument(
        "--no-boost-dictionary",
        dest="boost_dictionary",
        action="store_false",
        help="disable term oversampling",
    )
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("correct", help="correct archive OCR text(s) into Markdown")
    c.add_argument("--checkpoint", type=Path, default=DENOISER_CHECKPOINT)
    c.add_argument("--input", default=str(RAW_DIR), help="a *_djvu.txt file or the raw data directory")
    c.add_argument("--out-dir", default=str(PROCESSED_DIR))
    c.add_argument("--dictionary")
    c.add_argument("--dict-bonus", type=float, default=3.0, help="log-probability bonus for lexicon terms (nats)")
    c.add_argument("--no-dictionary", action="store_true")
    c.set_defaults(func=cmd_correct)

    k = sub.add_parser("calibration", help="concatenate processed text for llama-imatrix")
    k.add_argument("--processed", default=str(PROCESSED_DIR))
    k.add_argument("--out", default=str(CALIBRATION_DIR / "calibration.txt"))
    k.set_defaults(func=cmd_calibration)

    e = sub.add_parser("evaluate", help="measure CER and term recall on synthetic held-out OCR")
    e.add_argument("--checkpoint", type=Path, default=DENOISER_CHECKPOINT)
    e.add_argument("--corpus", nargs="*", default=[], help=f"clean corpus (default: {BASELINE_DIR})")
    e.add_argument("--dictionary")
    e.add_argument("--dict-bonus", type=float, default=3.0)
    e.add_argument("--noise", nargs="*", type=float, default=[0.5, 1.0, 2.0])
    e.add_argument("--valid-fraction", type=float, default=0.05)
    e.add_argument("--max-windows", type=int, default=400)
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--json", help="also write metrics to this JSON file")
    e.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        ui.warn("interrupted")
        return 130
    except FileNotFoundError as exc:
        ui.fail(f"missing file: {exc.filename or exc}")
        ui.info("run `python cleaner.py train` first, or pass --checkpoint")
        return 1
    except SystemExit:
        raise
    except Exception as exc:
        ui.fail(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
