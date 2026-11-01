"""Prompt formatting and tokenizers (a byte-level one for tiny/CPU tests so no downloads are needed)."""
from __future__ import annotations

import torch


def format_prompt(instruction: str) -> str:
    return f"In: {instruction}\nOut:"


class ByteTokenizer:
    """Minimal tokenizer for tests: one token per byte. vocab_size = 256 (+ pad id 0 reused)."""

    vocab_size = 256
    pad_token_id = 0

    def __call__(self, texts: list[str]) -> dict:
        return _left_pad([list(t.encode()) for t in texts], self.pad_token_id)


class HFTokenizer:
    def __init__(self, name: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.pad_token_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.vocab_size = len(self.tok)

    def __call__(self, texts: list[str]) -> dict:
        return _left_pad([self.tok(t, add_special_tokens=False)["input_ids"] for t in texts], self.pad_token_id)


def _left_pad(seqs: list[list[int]], pad: int) -> dict:
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad, dtype=torch.long)
    mask = torch.zeros(len(seqs), n, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = torch.tensor(s)
        mask[i, n - len(s):] = 1
    return {"input_ids": ids, "attention_mask": mask}


def encode_instructions(tokenizer, instructions: list[str]) -> dict:
    return tokenizer([format_prompt(i) for i in instructions])
