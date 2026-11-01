"""OpenVLA-style uniform 256-bin action tokenizer. Actions are clipped to [-1, 1]."""
from __future__ import annotations

import numpy as np
import torch


class ActionTokenizer:
    def __init__(self, n_bins: int = 256, base_id: int = 0):
        self.n_bins, self.base_id = n_bins, base_id  # token id = base_id + bin

    def encode_bins(self, a):
        """float [-1,1] -> int bin in [0, n_bins-1]. Works on np arrays and torch tensors."""
        if isinstance(a, torch.Tensor):
            return torch.round((a.clamp(-1, 1) + 1) / 2 * (self.n_bins - 1)).long()
        return np.round((np.clip(a, -1, 1) + 1) / 2 * (self.n_bins - 1)).astype(np.int64)

    def decode_bins(self, bins):
        return bins / (self.n_bins - 1) * 2 - 1  # max round-trip error 1/(n_bins-1)

    def encode(self, a):
        return self.encode_bins(a) + self.base_id

    def decode(self, ids):
        return self.decode_bins(ids - self.base_id)
