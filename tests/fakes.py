"""In-memory stand-ins for archive.org, used by the scraper tests (no network)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch

from tantra.corrector import Corrector
from tantra.dictionary import EsotericDictionary
from tantra.model import CharVocab, ModelConfig, TagDenoiser, TagVocab
from tantra.scraper import RemoteFile


class FakeArchive:
    def __init__(self, records: list[dict[str, Any]], files: dict[str, list[tuple[str, str, bytes]]] | None = None):
        self.records = records
        # identifier -> [(filename, format, content)]
        self._files = files or {}
        self.search_queries: list[str] = []
        self.download_calls: list[tuple[str, str]] = []
        self.fail_times: dict[str, int] = {}  # filename -> remaining failures to inject
        self.corrupt_once: set[str] = set()

    def search(self, query: str, fields: Iterable[str], limit: int) -> Iterator[dict[str, Any]]:
        self.search_queries.append(query)
        for record in self.records[:limit]:
            yield dict(record)

    def metadata(self, identifier: str) -> dict[str, Any]:
        for record in self.records:
            if record.get("identifier") == identifier:
                return dict(record)
        raise KeyError(identifier)

    def files(self, identifier: str) -> list[RemoteFile]:
        return [
            RemoteFile(name=name, format=fmt, size=len(content), md5=hashlib.md5(content).hexdigest())
            for name, fmt, content in self._files.get(identifier, [])
        ]

    def download(self, identifier: str, filename: str, dest_dir: Path) -> None:
        self.download_calls.append((identifier, filename))
        if self.fail_times.get(filename, 0) > 0:
            self.fail_times[filename] -= 1
            raise ConnectionError("simulated network failure")
        content = next(c for n, _, c in self._files[identifier] if n == filename)
        if filename in self.corrupt_once:
            self.corrupt_once.discard(filename)
            content = content[:-1] + b"?"
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / filename).write_bytes(content)


# ----------------------------------------------------------------- shared test doubles
TINY_CONFIG = ModelConfig(d_model=32, n_heads=2, n_layers=1, max_len=64, dropout=0.0)


def controlled_corrector(bonus: float, dictionary: EsotericDictionary | None) -> Corrector:
    """A corrector whose per-character probabilities we fix by hand.

    For the glyph 'শ' the model prefers keeping it (0.6) over writing 'স' (0.3).
    With a dictionary bonus, the lexicon term ন্যাস should win over the greedy ন্যাশ.
    """
    chars = CharVocab.from_counts({c: 3 for c in "ন্যাশস "}, min_count=1)
    tags = TagVocab(["স"])  # id 2
    model = TagDenoiser(len(chars), len(tags), TINY_CONFIG)
    corrector = Corrector(model, chars, tags, dictionary, dict_bonus=bonus)
    n_tags = len(tags)

    def fake_forward(batch: list[str]) -> torch.Tensor:
        width = max(len(w) for w in batch)
        base = math.log(1e-6)
        out = torch.full((len(batch), width, n_tags), base)
        for b, window in enumerate(batch):
            for i, ch in enumerate(window):
                out[b, i, 0] = math.log(0.99)  # KEEP
                if ch == "শ":
                    out[b, i, 0] = math.log(0.6)
                    out[b, i, tags.id_of("স")] = math.log(0.3)
        return out

    corrector._forward = fake_forward  # type: ignore[method-assign]
    return corrector


class EchoBackend:
    """Stands in for ``LlamaCppBackend``: streams a fixed reply word by word and records every prompt."""

    context_size = 512

    def __init__(
        self, model_path, *, n_ctx: int = 2048, threads: int = 4, seed: int = 0, reply: str = "উত্তর একটি দুই"
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.threads = threads
        self.seed = seed
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def stream_chat(self, messages, params):
        self.calls.append([dict(m) for m in messages])
        for word in self.reply.split(" "):
            yield word + " "
