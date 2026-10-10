"""Bidirectional character-level tagger that restores clean Bengali from OCR output.

Architecture (deliberately small so it trains on a laptop CPU in under an hour):

    chars -> embedding + learned positions -> N x pre-norm Transformer encoder
          -> LayerNorm -> linear head over edit labels (KEEP / DELETE / replacement)

The encoder is *bidirectional*: every character sees the whole window on both
sides, which is what resolves ambiguous glyphs ("which ক is this?") from context.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch
from torch import nn

from tantra.alignment import DELETE, KEEP

PAD_ID = 0
UNK_ID = 1
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
IGNORE_INDEX = -100


class CharVocab:
    """Character <-> id mapping. Ids 0 and 1 are reserved for padding and unknown characters."""

    def __init__(self, chars: Iterable[str]) -> None:
        self.chars: list[str] = [PAD_TOKEN, UNK_TOKEN, *chars]
        self._index = {ch: i for i, ch in enumerate(self.chars) if i >= 2}

    @classmethod
    def from_counts(cls, counts: dict[str, int], min_count: int = 2) -> CharVocab:
        kept = sorted(ch for ch, n in counts.items() if n >= min_count and len(ch) == 1 and ch != "\n")
        kept.append("\n")  # line breaks are always representable
        return cls(kept)

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self._index.get(ch, UNK_ID) for ch in text]

    def to_list(self) -> list[str]:
        return list(self.chars[2:])

    @classmethod
    def from_list(cls, chars: list[str]) -> CharVocab:
        return cls(chars)


class TagVocab:
    """Edit-label <-> id mapping. ``KEEP`` is always id 0 and ``DELETE`` id 1."""

    def __init__(self, labels: Iterable[str]) -> None:
        ordered = [KEEP, DELETE]
        for label in labels:
            if label not in ordered:
                ordered.append(label)
        self.labels: list[str] = ordered
        self._index = {label: i for i, label in enumerate(ordered)}

    def __len__(self) -> int:
        return len(self.labels)

    def id_of(self, label: str) -> int | None:
        return self._index.get(label)

    def to_list(self) -> list[str]:
        return list(self.labels)

    @classmethod
    def from_list(cls, labels: list[str]) -> TagVocab:
        vocab = cls([])
        vocab.labels = list(labels)
        vocab._index = {label: i for i, label in enumerate(vocab.labels)}
        return vocab


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 4
    ff_mult: int = 4
    max_len: int = 256
    dropout: float = 0.1


class TagDenoiser(nn.Module):
    """Bidirectional encoder mapping a character window to per-character label logits."""

    def __init__(self, n_chars: int, n_tags: int, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        self.embed = nn.Embedding(n_chars, config.d_model, padding_idx=PAD_ID)
        self.pos = nn.Embedding(config.max_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.ff_mult,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, n_tags)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``ids``: LongTensor [batch, time] -> logits [batch, time, n_tags]."""
        _, t = ids.shape
        if t > self.config.max_len:
            raise ValueError(f"window of {t} characters exceeds max_len={self.config.max_len}")
        positions = torch.arange(t, device=ids.device).unsqueeze(0)
        hidden = self.embed(ids) + self.pos(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=ids.eq(PAD_ID))
        return self.head(self.norm(hidden))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def config_to_dict(config: ModelConfig) -> dict[str, float | int]:
    return asdict(config)
