#!/usr/bin/env python3
"""Conservative Bengali OCR cleanup with optional contextual neural correction.

Default mode only normalizes Unicode/layout and applies dictionary-supported
repairs. It never substitutes a visually similar Bengali letter unless the
result is attested by a supplied clean corpus or the editable Tantra lexicon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

BENGALI_WORD_RE = re.compile(r"[\u0980-\u09ff]+")
BENGALI_PRESENT_RE = re.compile(r"[\u0980-\u09ff]")

# The examples intentionally cover common shapes, not every regional or archaic
# spelling. Add project-specific forms to data/tantra_terms.txt.
CONFUSION_GROUPS = (
    ("র", "ব", "ক"),
    ("স", "শ"),
    ("ন", "ণ"),
    ("ব", "ভ"),
    ("য", "য়"),
    ("দ", "ধ"),
    ("ত", "ট"),
)
CONFUSIONS: dict[str, tuple[str, ...]] = {}
for _group in CONFUSION_GROUPS:
    for _char in _group:
        alternatives = set(CONFUSIONS.get(_char, ()))
        alternatives.update(char for char in _group if char != _char)
        CONFUSIONS[_char] = tuple(sorted(alternatives))

# (left fragment, right fragment, repaired conjunct). Include both a visible
# hasant and a dropped-hasant OCR variant for the most common examples.
BROKEN_CONJUNCTS = (
    ("ক্", "ষ", "ক্ষ"),
    ("ক", "ষ", "ক্ষ"),
    ("জ্", "ঞ", "জ্ঞ"),
    ("জ", "ঞ", "জ্ঞ"),
    ("ত্", "র", "ত্র"),
    ("ত", "র", "ত্র"),
    ("শ্", "র", "শ্র"),
    ("শ", "র", "শ্র"),
    ("দ্", "ব", "দ্ব"),
    ("দ", "ব", "দ্ব"),
    ("ন্", "দ", "ন্দ"),
    ("ন", "দ", "ন্দ"),
    ("ন্", "ত", "ন্ত"),
    ("ন", "ত", "ন্ত"),
    ("ষ্", "ট", "ষ্ট"),
    ("ষ", "ট", "ষ্ট"),
    ("ষ্", "ঠ", "ষ্ঠ"),
    ("ষ", "ঠ", "ষ্ঠ"),
    ("ম্", "ব", "ম্ব"),
    ("ম", "ব", "ম্ব"),
    ("প্", "র", "প্র"),
    ("প", "র", "প্র"),
    ("ক্", "র", "ক্র"),
    ("ক", "র", "ক্র"),
    ("গ্", "র", "গ্র"),
    ("গ", "র", "গ্র"),
    ("ত্", "ত", "ত্ত"),
    ("ত", "ত", "ত্ত"),
)


def _is_bengali_letter(char: str) -> bool:
    return bool(char) and "\u0980" <= char <= "\u09ff" and unicodedata.category(char).startswith("L")


def normalize_ocr_text(text: str, *, preserve_lines: bool = False) -> str:
    """Normalize Unicode and physical line wrapping without rewriting words."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x0c", "\n\n").replace("\x00", "")
    for invisible in ("\ufeff", "\u200b", "\u00ad"):
        text = text.replace(invisible, "")

    # OCR often leaves a space after an explicit hasant/virama. This repair is
    # safer than joining arbitrary adjacent Bengali words because the conjunct
    # marker is still present in the source.
    text = re.sub(
        r"([\u0985-\u09b9]\u09cd)\s+([\u0985-\u09b9])",
        r"\1\2",
        text,
    )

    normalized_blocks: list[str] = []
    for block in re.split(r"\n\s*\n+", text):
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in block.split("\n")]
        lines = [line for line in lines if line]
        if not lines:
            continue
        rendered = "\n".join(lines) if preserve_lines else " ".join(lines)
        rendered = re.sub(r"[ \t]+", " ", rendered)
        rendered = re.sub(r"[ \t]+([,;:!?।॥])", r"\1", rendered)
        normalized_blocks.append(rendered.strip())
    return "\n\n".join(normalized_blocks).strip()


def synthetic_corrupt(
    text: str,
    *,
    rng: random.Random | None = None,
    conjunct_rate: float = 0.08,
    confusion_rate: float = 0.015,
    whitespace_rate: float = 0.02,
) -> str:
    """Inject reproducible, configurable Bengali scanner-like noise.

    The operation is for making paired training examples; it is not a cleaner.
    Lower probabilities are intentionally used for look-alike substitutions so
    the model does not learn to change every uncommon word.
    """
    rng = rng or random.Random()
    noisy = unicodedata.normalize("NFC", text)

    unique_conjuncts = sorted({joined for _, _, joined in BROKEN_CONJUNCTS}, key=len, reverse=True)
    for conjunct in unique_conjuncts:
        pattern = re.compile(re.escape(conjunct))

        def break_match(match: re.Match[str], current: str = conjunct) -> str:
            if rng.random() >= conjunct_rate:
                return match.group(0)
            for left, right, joined in BROKEN_CONJUNCTS:
                if joined == current:
                    return f"{left.rstrip('্')} {right}"
            return match.group(0)

        noisy = pattern.sub(break_match, noisy)

    confused: list[str] = []
    for char in noisy:
        alternatives = CONFUSIONS.get(char, ())
        if alternatives and rng.random() < confusion_rate:
            confused.append(rng.choice(alternatives))
        else:
            confused.append(char)

    # Insert whitespace only between letters, never in a Bengali combining mark
    # or between a base character and its dependent vowel sign.
    spaced: list[str] = []
    for index, char in enumerate(confused):
        spaced.append(char)
        if (
            index + 1 < len(confused)
            and _is_bengali_letter(char)
            and _is_bengali_letter(confused[index + 1])
            and rng.random() < whitespace_rate
        ):
            spaced.append(" ")
    return "".join(spaced)


def default_terms_path() -> Path:
    local = Path(__file__).resolve().parent / "data" / "tantra_terms.txt"
    installed = Path(sys.prefix) / "share" / "tantra-llm" / "data" / "tantra_terms.txt"
    return local if local.is_file() else installed


def load_terms(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    terms: set[str] = set()
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            terms.add(normalize_ocr_text(value))
    return terms


def corpus_files(paths: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            if path.suffix.casefold() in (".txt", ".md"):
                files.add(path.resolve())
        elif path.is_dir():
            files.update(
                child.resolve()
                for child in path.rglob("*")
                if child.is_file() and child.suffix.casefold() in (".txt", ".md")
            )
    return sorted(files)


def _strip_markdown_metadata(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                lines = lines[index + 1 :]
                break
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].lstrip().startswith("# "):
        lines = lines[1:]
    return "\n".join(lines)


def read_corpus(paths: Iterable[Path]) -> tuple[str, Counter[str]]:
    texts: list[str] = []
    words: Counter[str] = Counter()
    for path in corpus_files(paths):
        try:
            raw = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            print(f"[warn] Unable to read baseline {path}: {exc}", file=sys.stderr)
            continue
        normalized = normalize_ocr_text(_strip_markdown_metadata(raw), preserve_lines=True)
        if not normalized:
            continue
        texts.append(normalized)
        words.update(BENGALI_WORD_RE.findall(normalized))
    return "\n\n".join(texts), words


class BidirectionalCharLanguageModel:
    """Small add-one-smoothed character trigram scorer trained from clean text."""

    def __init__(self, text: str = "", *, alpha: float = 0.2):
        self.alpha = alpha
        self.vocabulary = set(text)
        self.forward: Counter[tuple[str, str, str]] = Counter()
        self.backward: Counter[tuple[str, str, str]] = Counter()
        self.forward_contexts: Counter[tuple[str, str]] = Counter()
        self.backward_contexts: Counter[tuple[str, str]] = Counter()
        if text:
            self._count(text, reverse=False)
            self._count(text[::-1], reverse=True)

    def _count(self, sequence: str, *, reverse: bool) -> None:
        padded = "  " + sequence + "  "
        trigrams = self.backward if reverse else self.forward
        contexts = self.backward_contexts if reverse else self.forward_contexts
        for index in range(2, len(padded)):
            context = (padded[index - 2], padded[index - 1])
            trigrams[(*context, padded[index])] += 1
            contexts[context] += 1

    def _direction_score(
        self,
        sequence: str,
        trigrams: Counter[tuple[str, str, str]],
        contexts: Counter[tuple[str, str]],
    ) -> float:
        padded = "  " + sequence + "  "
        vocab_size = max(1, len(self.vocabulary) + 1)
        score = 0.0
        observed = 0
        for index in range(2, len(padded)):
            context = (padded[index - 2], padded[index - 1])
            char = padded[index]
            count = trigrams[(context[0], context[1], char)]
            total = contexts[context]
            score += math.log((count + self.alpha) / (total + self.alpha * vocab_size))
            observed += 1
        return score / max(1, observed)

    def score(self, sequence: str) -> float:
        if not self.forward:
            return 0.0
        return 0.5 * (
            self._direction_score(sequence, self.forward, self.forward_contexts)
            + self._direction_score(sequence[::-1], self.backward, self.backward_contexts)
        )


class LexiconCorrector:
    """Use a clean-corpus vocabulary and domain lexicon as conservative anchors."""

    def __init__(
        self,
        *,
        terms: Iterable[str] = (),
        word_counts: Counter[str] | None = None,
        language_model: BidirectionalCharLanguageModel | None = None,
        context_chars: int = 12,
    ):
        self.terms = {normalize_ocr_text(term) for term in terms if normalize_ocr_text(term)}
        self.domain_words = {term for term in self.terms if " " not in term}
        self.word_counts = word_counts or Counter()
        self.known_words = set(self.word_counts) | self.domain_words
        self.language_model = language_model
        self.context_chars = max(2, context_chars)
        self.term_pattern = (
            re.compile("|".join(re.escape(term) for term in sorted(self.terms, key=len, reverse=True)))
            if self.terms
            else None
        )

    def _candidate_words(self, word: str) -> set[str]:
        result: set[str] = set()
        for index, char in enumerate(word):
            for replacement in CONFUSIONS.get(char, ()):
                candidate = word[:index] + replacement + word[index + 1 :]
                if candidate in self.known_words:
                    result.add(candidate)
        return result

    def _context_score(self, text: str, start: int, end: int, candidate: str) -> float:
        if self.language_model is None:
            return 0.0
        left = text[max(0, start - self.context_chars) : start]
        right = text[end : end + self.context_chars]
        return self.language_model.score(left + candidate + right)

    def _correct_word(self, text: str, match: re.Match[str]) -> str | None:
        word = match.group(0)
        if word in self.known_words:
            return None
        candidates = self._candidate_words(word)
        if not candidates:
            return None
        domain = candidates & self.domain_words
        if len(domain) == 1:
            return next(iter(domain))
        if self.language_model is None:
            return None

        ranked = sorted(
            (
                self._context_score(text, match.start(), match.end(), candidate)
                + math.log1p(self.word_counts.get(candidate, 0)) * 0.02,
                candidate,
            )
            for candidate in candidates
        )
        ranked.reverse()
        if len(ranked) == 1:
            return ranked[0][1]
        # Do not force a spelling when context is indecisive.
        return ranked[0][1] if ranked[0][0] - ranked[1][0] >= 0.035 else None

    def _merge_candidate(
        self,
        text: str,
        left_match: re.Match[str],
        right_match: re.Match[str],
        gap: str,
    ) -> str | None:
        left, right = left_match.group(0), right_match.group(0)
        possible = set()
        for left_fragment, right_fragment, conjunct in BROKEN_CONJUNCTS:
            if left.endswith(left_fragment) and right.startswith(right_fragment):
                candidate = (
                    left[: -len(left_fragment)]
                    + conjunct
                    + right[len(right_fragment) :]
                )
                if candidate in self.known_words:
                    possible.add(candidate)
        domain = possible & self.domain_words
        if len(domain) == 1:
            return next(iter(domain))
        if self.language_model is None or not possible:
            return None

        start, end = left_match.start(), right_match.end()
        before = text[max(0, start - self.context_chars) : start]
        after = text[end : end + self.context_chars]
        split_score = self.language_model.score(before + left + gap + right + after)
        ranked = sorted(
            (
                self.language_model.score(before + candidate + after)
                + math.log1p(self.word_counts.get(candidate, 0)) * 0.02,
                candidate,
            )
            for candidate in possible
            if self.word_counts.get(candidate, 0) >= 2
        )
        if not ranked:
            return None
        ranked.reverse()
        if ranked[0][0] - split_score < 0.025:
            return None
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.035:
            return None
        return ranked[0][1]

    def correct(self, text: str) -> str:
        matches = list(BENGALI_WORD_RE.finditer(text))
        if not matches:
            return text
        replacements: list[tuple[int, int, str]] = []
        index = 0
        while index < len(matches):
            current = matches[index]
            if index + 1 < len(matches):
                following = matches[index + 1]
                gap = text[current.end() : following.start()]
                if gap and gap.isspace() and "\n" not in gap:
                    joined = self._merge_candidate(text, current, following, gap)
                    if joined:
                        replacements.append((current.start(), following.end(), joined))
                        index += 2
                        continue
            correction = self._correct_word(text, current)
            if correction:
                replacements.append((current.start(), current.end(), correction))
            index += 1

        for start, end, value in reversed(replacements):
            text = text[:start] + value + text[end:]
        return text

    def correct_with_anchors(self, text: str, model: Any) -> str:
        """Apply a neural model only to spans outside exact protected terms."""
        if not self.term_pattern:
            return model.correct(text)
        pieces: list[str] = []
        cursor = 0
        for match in self.term_pattern.finditer(text):
            if match.start() > cursor:
                pieces.append(model.correct(text[cursor : match.start()]))
            pieces.append(match.group(0))
            cursor = match.end()
        if cursor < len(text):
            pieces.append(model.correct(text[cursor:]))
        return "".join(pieces)


def clean_ocr_text(
    text: str,
    *,
    corrector: LexiconCorrector | None = None,
    model: Any = None,
    preserve_lines: bool = False,
) -> str:
    normalized = normalize_ocr_text(text, preserve_lines=preserve_lines)
    if model is not None:
        if corrector is not None:
            normalized = corrector.correct_with_anchors(normalized, model)
        else:
            normalized = model.correct(normalized)
    if corrector is not None:
        normalized = corrector.correct(normalized)
    return normalize_ocr_text(normalized, preserve_lines=preserve_lines)


def load_neural_corrector(checkpoint: Path, *, max_edit_ratio: float = 0.34):
    """Load a locally trained Transformer; no network download is performed."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional training extra
        raise RuntimeError("Neural correction requires PyTorch; install `pip install -e '.[train]'`.") from exc
    from neural_corrector import SPECIAL_TOKENS, build_model

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch releases before the weights_only option.
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("format") != "tantra-char-denoiser-v1":
        raise ValueError(f"Unrecognized correction checkpoint: {checkpoint}")

    vocab = payload["vocab"]
    config = payload["config"]
    model = build_model(len(vocab), config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    char_to_id = {char: index for index, char in enumerate(vocab)}
    pad_id = char_to_id[SPECIAL_TOKENS[0]]
    bos_id = char_to_id[SPECIAL_TOKENS[1]]
    eos_id = char_to_id[SPECIAL_TOKENS[2]]
    unk_id = char_to_id[SPECIAL_TOKENS[3]]
    max_length = int(config["max_length"])

    class LocalCharCorrector:
        def _infer_chunk(self, source: str) -> str:
            import torch

            if not source:
                return source
            # Keep room for the end token and never exceed the trained position table.
            source = source[: max_length - 1]
            source_ids = [char_to_id.get(char, unk_id) for char in source]
            source_tensor = torch.tensor([source_ids], dtype=torch.long)
            with torch.inference_mode():
                memory, source_padding = model.encode(source_tensor)
                target_ids = [bos_id]
                max_output = min(max_length - 1, max(8, len(source) + 12))
                for _ in range(max_output):
                    target_tensor = torch.tensor([target_ids], dtype=torch.long)
                    logits = model.decode(target_tensor, memory, source_padding)
                    next_id = int(logits[0, -1].argmax().item())
                    if next_id == eos_id:
                        break
                    if next_id not in (pad_id, bos_id):
                        target_ids.append(next_id)
                output = "".join(
                    vocab[token_id]
                    for token_id in target_ids[1:]
                    if token_id not in (pad_id, bos_id, eos_id, unk_id)
                )
            if not output or not BENGALI_PRESENT_RE.search(output):
                return source
            ratio = _edit_ratio(source, output)
            if ratio > max_edit_ratio:
                return source
            return output

        def correct(self, text: str) -> str:
            # Paragraphs stay intact; very long ones split at the nearest space so
            # the model sees both left and right context without truncating silently.
            blocks = re.split(r"(\n\s*\n)", text)
            result: list[str] = []
            for block in blocks:
                if not block or re.fullmatch(r"\n\s*\n", block):
                    result.append(block)
                    continue
                cursor = 0
                while cursor < len(block):
                    limit = min(len(block), cursor + max_length - 1)
                    end = limit
                    if limit < len(block):
                        boundary = block.rfind(" ", cursor + max_length // 2, limit)
                        if boundary > cursor:
                            end = boundary
                        else:
                            # A pathological run with no safe whitespace boundary is
                            # preserved verbatim instead of split inside a Bengali word.
                            result.append(block[cursor:])
                            break
                    source = block[cursor:end]
                    result.append(self._infer_chunk(source))
                    cursor = end
                    if cursor < len(block) and block[cursor] == " ":
                        result.append(" ")
                        cursor += 1
            return "".join(result)

    return LocalCharCorrector()


def _edit_ratio(source: str, target: str) -> float:
    """Normalized Levenshtein distance; paragraphs are capped to model length."""
    if source == target:
        return 0.0
    if not source or not target:
        return 1.0
    previous = list(range(len(target) + 1))
    for row, left in enumerate(source, start=1):
        current = [row]
        for column, right in enumerate(target, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1] / max(len(source), len(target))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_for(path: Path) -> Mapping[str, Any]:
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        return data if isinstance(data, Mapping) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _metadata_text(metadata: Mapping[str, Any], field: str, fallback: str) -> str:
    value = metadata.get(field)
    if isinstance(value, (list, tuple)):
        value = " ".join(map(str, value))
    return str(value).strip() if value else fallback


def markdown_document(
    text: str,
    *,
    title: str,
    source: str,
    identifier: str = "",
    source_file: str = "",
    correction_profile: str = "layout+lexicon",
) -> str:
    safe_title = " ".join(title.replace("#", "").split()) or "Untitled Bengali source"
    front_matter = [
        "---",
        f"title: {json.dumps(safe_title, ensure_ascii=False)}",
        f"source: {json.dumps(source, ensure_ascii=False)}",
        f"source_file: {json.dumps(source_file, ensure_ascii=False)}",
        f"archive_identifier: {json.dumps(identifier, ensure_ascii=False)}",
        f"correction_profile: {json.dumps(correction_profile, ensure_ascii=False)}",
        "---",
    ]
    return "\n".join(front_matter) + f"\n\n# {safe_title}\n\n{text.strip()}\n"


def _output_for(source: Path, input_path: Path, output_dir: Path) -> Path:
    if input_path.is_file():
        return output_dir / f"{source.stem}.md"
    relative = source.relative_to(input_path)
    return output_dir / relative.parent / f"{source.stem}.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize and contextually correct Archive.org Bengali OCR into Markdown."
    )
    parser.add_argument("--input", type=Path, required=True, help="OCR text file or directory (recursive).")
    parser.add_argument("--output", type=Path, required=True, help="Directory for unified Markdown training files.")
    parser.add_argument("--baseline", type=Path, action="append", default=[], help="Optional clean Bengali corpus file/directory; repeat to add more. Enables vocabulary and bidirectional character-context scoring.")
    parser.add_argument("--terms", type=Path, default=default_terms_path(), help="Editable domain lexicon (one term per line).")
    parser.add_argument("--model", type=Path, help="Optional checkpoint from train_corrector.py; local inference only.")
    parser.add_argument("--max-edit-ratio", type=float, default=0.34, help="Reject neural rewrites above this normalized edit distance (default: 0.34).")
    parser.add_argument("--preserve-lines", action="store_true", help="Preserve single OCR line breaks (useful for verse); default reflows physical lines.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output files; without this, existing outputs are left untouched.")
    return parser


def _source_files(input_path: Path, output_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path.resolve()]
    resolved_output = output_path.resolve()
    return [
        path.resolve()
        for path in sorted(input_path.rglob("*"))
        if path.is_file()
        and path.suffix.casefold() in (".txt", ".md")
        and resolved_output not in path.resolve().parents
        and path.name.casefold() not in ("manifest.txt",)
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"Input path does not exist: {args.input}", file=sys.stderr)
        return 2
    if not args.input.is_file() and not args.input.is_dir():
        print(f"Input is not a file or directory: {args.input}", file=sys.stderr)
        return 2
    if not 0.0 <= args.max_edit_ratio <= 1.0:
        print("--max-edit-ratio must be between 0 and 1", file=sys.stderr)
        return 2

    terms = load_terms(args.terms)
    baseline_text, word_counts = read_corpus(args.baseline)
    language_model = BidirectionalCharLanguageModel(baseline_text) if baseline_text else None
    corrector = LexiconCorrector(terms=terms, word_counts=word_counts, language_model=language_model)

    model = None
    if args.model:
        try:
            model = load_neural_corrector(args.model, max_edit_ratio=args.max_edit_ratio)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Unable to load correction model: {exc}", file=sys.stderr)
            return 2

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    sources = _source_files(input_root, output_root)
    if not sources:
        print(f"No .txt or .md files found under {args.input}", file=sys.stderr)
        return 1

    manifest: list[dict[str, Any]] = []
    failures = 0
    for index, source in enumerate(sources, start=1):
        target = _output_for(source, input_root, output_root)
        if target.exists() and not args.overwrite:
            print(f"[{index}/{len(sources)}] skip existing {target}")
            continue
        try:
            raw = source.read_text(encoding="utf-8-sig", errors="replace")
            metadata = _metadata_for(source)
            metadata_id = metadata.get("identifier")
            if not metadata_id and (source.parent / "metadata.json").is_file():
                metadata_id = source.parent.name
            identifier = str(metadata_id or "")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", identifier):
                identifier = ""
            title = _metadata_text(metadata, "title", source.stem.replace("_", " "))
            cleaned = clean_ocr_text(
                raw,
                corrector=corrector,
                model=model,
                preserve_lines=args.preserve_lines,
            )
            profile = "layout+lexicon+bidirectional-char-lm" if language_model else "layout+lexicon"
            if model:
                profile += "+local-char-transformer"
            source_reference = (
                f"https://archive.org/details/{identifier}"
                if identifier
                else source.name
            )
            document = markdown_document(
                cleaned,
                title=title,
                source=source_reference,
                source_file=source.name,
                identifier=identifier,
                correction_profile=profile,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(document, encoding="utf-8")
            temporary.replace(target)
            manifest.append(
                {
                    "source": str(source),
                    "source_sha256": sha256_file(source),
                    "output": str(target),
                    "output_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
                    "archive_identifier": identifier,
                    "title": title,
                    "characters_in": len(raw),
                    "characters_out": len(cleaned),
                    "correction_profile": profile,
                }
            )
            print(f"[{index}/{len(sources)}] {source} → {target} ({len(cleaned):,} chars)")
        except (OSError, UnicodeError, ValueError) as exc:
            failures += 1
            print(f"[error] {source}: {exc}", file=sys.stderr)

    if manifest:
        args.output.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output / "clean_manifest.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    print(f"Completed: {len(manifest)} cleaned, {failures} failed; output: {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
