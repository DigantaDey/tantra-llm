#!/usr/bin/env python3
"""Create a *tiny, randomly initialised* causal LM for exercising the pipeline offline.

The weights are random, so the model talks nonsense. It exists so that
``quantize.sh``, ``app.py`` and ``benchmark.py`` can be exercised end-to-end on a
machine without access to the Hugging Face Hub. Never ship its output.

Why LLaMA + SentencePiece? llama.cpp identifies BPE tokenizers by a hash of
their behaviour and rejects unknown ones, whereas SentencePiece vocabularies
take a generic path. The production model (Qwen2.5) is converted by the same
``quantize.sh`` script; that conversion must be checked against the real
checkpoint on a machine with Hugging Face access.

Usage:
    python scripts/make_tiny_model.py --corpus data/baseline --out /tmp/tiny-model
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import LlamaConfig, LlamaForCausalLM

CHATML = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="directory of .txt files for the tokenizer")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocab", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    files = sorted(args.corpus.glob("*.txt"))[:2]
    if not files:
        raise SystemExit(f"no .txt files in {args.corpus}")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / "tok")
        spm.SentencePieceTrainer.train(
            input=",".join(str(f) for f in files),
            model_prefix=prefix,
            vocab_size=args.vocab,
            model_type="bpe",
            character_coverage=1.0,
            byte_fallback=True,
            bos_id=1,
            eos_id=2,
            unk_id=0,
            pad_id=-1,
            input_sentence_size=20000,
            shuffle_input_sentence=True,
            minloglevel=2,
        )
        model_bytes = Path(prefix + ".model").read_bytes()
    (args.out / "tokenizer.model").write_bytes(model_bytes)
    # Read the vocabulary size from SentencePiece itself: it is the ground truth for the converter.
    vocab_size = spm.SentencePieceProcessor(model_file=str(args.out / "tokenizer.model")).get_piece_size()
    (args.out / "chat_template.jinja").write_text(CHATML, encoding="utf-8")

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(args.out)
    print(f"wrote tiny random LLaMA ({sum(p.numel() for p in model.parameters()):,} params) to {args.out}")


if __name__ == "__main__":
    main()
