"""Optional tiny character Transformer used by train_corrector.py and cleaner.py.

PyTorch is imported lazily so scraping and the conservative rule-based cleaner
remain usable on machines that do not have the training extra installed.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


def build_model(vocab_size: int, config: Mapping[str, Any]):
    """Construct the checkpoint's small encoder-decoder denoising Transformer."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional PyTorch
        raise RuntimeError(
            "PyTorch is required for neural OCR correction; install `pip install -e '.[train]'`."
        ) from exc

    pad_id = int(config["pad_id"])
    d_model = int(config.get("d_model", 128))
    nhead = int(config.get("nhead", 4))
    layers = int(config.get("layers", 2))
    feedforward = int(config.get("feedforward", d_model * 3))
    max_length = int(config.get("max_length", 256))
    dropout = float(config.get("dropout", 0.1))

    class DenoisingTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
            self.position = nn.Embedding(max_length, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
            self.output = nn.Linear(d_model, vocab_size)
            self.d_model = d_model
            self.max_length = max_length
            self.pad_id = pad_id

        def _embed(self, token_ids):
            positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
            return self.embedding(token_ids) * (self.d_model ** 0.5) + self.position(positions)

        def encode(self, source_ids):
            source_padding = source_ids.eq(self.pad_id)
            memory = self.encoder(
                self._embed(source_ids),
                src_key_padding_mask=source_padding,
            )
            return memory, source_padding

        def decode(self, target_ids, memory, source_padding):
            length = target_ids.shape[1]
            causal_mask = torch.triu(
                torch.ones((length, length), dtype=torch.bool, device=target_ids.device),
                diagonal=1,
            )
            target_padding = target_ids.eq(self.pad_id)
            decoded = self.decoder(
                self._embed(target_ids),
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=target_padding,
                memory_key_padding_mask=source_padding,
            )
            return self.output(decoded)

        def forward(self, source_ids, target_ids):
            memory, source_padding = self.encode(source_ids)
            return self.decode(target_ids, memory, source_padding)

    return DenoisingTransformer()
