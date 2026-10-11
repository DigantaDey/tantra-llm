"""Inference: OCR text -> corrected Bengali, with a dictionary anchor.

For every word the corrector weighs three kinds of candidates:

* the model's greedy reading of the word (highest per-character probability);
* the word exactly as scanned;
* esoteric-dictionary terms within two edits of the scanned word.

Each candidate's score is the model's own log-probability of producing it
(summed over characters, with the same edit labels the model was trained on)
plus ``dict_bonus`` if the candidate is a dictionary term. A large bonus pins
Tantric vocabulary; a small one lets strong contextual evidence win. This is
the "balance linguistic probability with esoteric accuracy" mechanism.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from tantra.alignment import apply_labels, to_labels
from tantra.bengali import word_spans
from tantra.dataset import make_windows
from tantra.dictionary import EsotericDictionary
from tantra.model import CharVocab, ModelConfig, TagDenoiser, TagVocab

CHECKPOINT_FORMAT = "tantra-denoiser/v1"
UNKNOWN_LABEL_LOGP = math.log(1e-4)  # used when a candidate needs a label the model never learned
NEG_INF = float("-inf")


@dataclass
class CorrectionStats:
    lines: int = 0
    windows: int = 0
    changed_windows: int = 0
    dictionary_anchors: int = 0
    anchored_terms: dict[str, int] = field(default_factory=dict)

    def merge(self, other: CorrectionStats) -> None:
        self.lines += other.lines
        self.windows += other.windows
        self.changed_windows += other.changed_windows
        self.dictionary_anchors += other.dictionary_anchors
        for term, n in other.anchored_terms.items():
            self.anchored_terms[term] = self.anchored_terms.get(term, 0) + n


class Corrector:
    """Runs a trained :class:`TagDenoiser` over OCR lines."""

    def __init__(
        self,
        model: TagDenoiser,
        chars: CharVocab,
        tags: TagVocab,
        dictionary: EsotericDictionary | None = None,
        *,
        dict_bonus: float = 3.0,
        batch_size: int = 32,
    ) -> None:
        self.model = model.eval()
        self.chars = chars
        self.tags = tags
        self.dictionary = dictionary
        self.dict_bonus = dict_bonus
        self.batch_size = batch_size
        self.max_len = model.config.max_len

    # ----- persistence ------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str | Path,
        dictionary: EsotericDictionary | None = None,
        *,
        dict_bonus: float = 3.0,
        batch_size: int = 32,
    ) -> Corrector:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a tantra denoiser checkpoint")
        chars = CharVocab.from_list(checkpoint["char_vocab"])
        tags = TagVocab.from_list(checkpoint["tag_vocab"])
        config = ModelConfig(**checkpoint["config"])
        model = TagDenoiser(len(chars), len(tags), config)
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, chars, tags, dictionary, dict_bonus=dict_bonus, batch_size=batch_size)

    def checkpoint(self, extra: dict | None = None) -> dict:
        payload = {
            "format": CHECKPOINT_FORMAT,
            "config": asdict(self.model.config),
            "char_vocab": self.chars.to_list(),
            "tag_vocab": self.tags.to_list(),
            "state_dict": self.model.state_dict(),
        }
        if extra:
            payload.update(extra)
        return payload

    # ----- public API -------------------------------------------------------
    def correct_lines(
        self,
        lines: list[str],
        *,
        use_dictionary: bool = True,
    ) -> tuple[list[str], CorrectionStats]:
        """Correct a list of text lines. Output has the same number of lines and the same spacing."""
        stats = CorrectionStats(lines=len(lines))
        windows: list[str] = []
        owners: list[tuple[int, int]] = []  # (line index, window index within line)
        per_line: list[list[str]] = []
        for li, line in enumerate(lines):
            parts = make_windows(line, self.max_len) if line else []
            per_line.append(parts)
            for wi, part in enumerate(parts):
                windows.append(part)
                owners.append((li, wi))

        corrected = self._correct_windows(windows, stats, use_dictionary=use_dictionary)
        out_parts: list[list[str]] = [[] for _ in lines]
        for (li, _wi), text in zip(owners, corrected, strict=True):
            out_parts[li].append(text)
        results = ["".join(parts) for parts in out_parts]
        return results, stats

    # ----- internals --------------------------------------------------------
    def _correct_windows(self, windows: list[str], stats: CorrectionStats, *, use_dictionary: bool) -> list[str]:
        stats.windows += len(windows)
        if not windows:
            return []
        # Sort by length so each batch pads as little as possible; restore order afterwards.
        order = sorted(range(len(windows)), key=lambda i: len(windows[i]))
        result: list[str | None] = [None] * len(windows)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idxs = order[start : start + self.batch_size]
                batch = [windows[i] for i in idxs]
                log_probs = self._forward(batch)
                for slot, (i, window) in enumerate(zip(idxs, batch, strict=True)):
                    n = len(window)
                    text = self._decode(window, log_probs[slot, :n], stats, use_dictionary)
                    result[i] = text
                    if text != window:
                        stats.changed_windows += 1
        return [r if r is not None else "" for r in result]

    def _forward(self, batch: list[str]) -> torch.Tensor:
        encoded = [self.chars.encode(w) for w in batch]
        width = max(len(e) for e in encoded)
        ids = torch.zeros((len(batch), width), dtype=torch.long)  # 0 == PAD
        for row, seq in enumerate(encoded):
            ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return torch.log_softmax(self.model(ids), dim=-1)

    def _render(self, ch: str, label_id: int) -> str:
        label = self.tags.labels[label_id]
        if label == "<KEEP>":
            return ch
        if label == "<DEL>":
            return ""
        return label

    def _decode(
        self,
        window: str,
        log_probs: torch.Tensor,
        stats: CorrectionStats,
        use_dictionary: bool,
    ) -> str:
        best_ids = log_probs.argmax(dim=-1).tolist()
        pieces = [self._render(ch, lid) for ch, lid in zip(window, best_ids, strict=True)]
        if not use_dictionary or self.dictionary is None:
            return "".join(pieces)

        replacements: dict[int, tuple[int, str]] = {}
        for start, end in word_spans(window):
            word = window[start:end]
            if not self.dictionary.lookup(word):
                continue  # nothing esoteric nearby: trust the model
            greedy = "".join(pieces[start:end])
            best_text, best_score = greedy, self._greedy_score(log_probs, best_ids, start, end, greedy)
            for candidate in self._candidates(word, greedy):
                score = self._candidate_score(word, candidate, log_probs, start)
                if score > best_score:
                    best_text, best_score = candidate, score
            if best_text != greedy:
                replacements[start] = (end, best_text)
                stats.dictionary_anchors += 1
                stats.anchored_terms[best_text] = stats.anchored_terms.get(best_text, 0) + 1

        if not replacements:
            return "".join(pieces)
        out: list[str] = []
        i = 0
        while i < len(window):
            if i in replacements:
                end, text = replacements[i]
                out.append(text)
                i = end
            else:
                out.append(pieces[i])
                i += 1
        return "".join(out)

    def _candidates(self, word: str, greedy: str) -> list[str]:
        assert self.dictionary is not None
        options = {word} | {term for term, _ in self.dictionary.lookup(word)}
        options.discard(greedy)
        return sorted(options)

    def _greedy_score(self, log_probs: torch.Tensor, best_ids: list[int], start: int, end: int, greedy: str) -> float:
        total = sum(float(log_probs[i, best_ids[i]]) for i in range(start, end))
        if self.dictionary is not None and greedy in self.dictionary:
            total += self.dict_bonus
        return total

    def _candidate_score(self, word: str, candidate: str, log_probs: torch.Tensor, start: int) -> float:
        labels = to_labels(word, candidate)
        if labels is None or apply_labels(word, labels) != candidate:
            return NEG_INF
        total = 0.0
        for k, label in enumerate(labels):
            tag_id = self.tags.id_of(label)
            total += float(log_probs[start + k, tag_id]) if tag_id is not None else UNKNOWN_LABEL_LOGP
        if self.dictionary is not None and candidate in self.dictionary:
            total += self.dict_bonus
        return total
