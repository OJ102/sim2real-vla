"""VLADataset over HDF5 shards: (image, proprio, instruction, action chunk [t:t+H]) with last-action padding + mask."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from vla.model.prompts import encode_instructions


class VLADataset(Dataset):
    """part: all | train | val  (val = every 20th episode globally, i.e. 5% held-out demos for tok_acc).
    subset: optional path to a subset_{pct}.json index file ([[shard, ep], ...])."""

    def __init__(self, root: str, chunk: int = 4, part: str = "all", subset: str | None = None):
        self.root, self.chunk = Path(root), chunk
        st = json.load(open(self.root / "stats.json"))
        self.p_mean = np.array(st["proprio_mean"], np.float32)
        self.p_std = np.array(st["proprio_std"], np.float32)
        self.shards = sorted(p.name for p in self.root.glob("*.h5"))
        allowed = None
        if subset:
            allowed = {(s, e) for s, e in json.load(open(subset))}
        self._files: dict = {}
        shard_idx, ep_idx, ep_start, ep_len, instr, frame_ep = [], [], [], [], [], []
        g = 0
        for si, s in enumerate(self.shards):
            with h5py.File(self.root / s, "r") as f:
                starts, lens, ins = f["ep_start"][:], f["ep_len"][:], [x.decode() for x in f["instruction"][:]]
            for e in range(len(lens)):
                keep = (part == "all" or (part == "val") == (g % 20 == 0)) and (allowed is None or (s, e) in allowed)
                g += 1
                if keep:
                    frame_ep.extend([len(ep_len)] * int(lens[e]))
                    shard_idx.append(si); ep_idx.append(e); ep_start.append(int(starts[e]))
                    ep_len.append(int(lens[e])); instr.append(ins[e])
        self.ep_shard, self.ep_start, self.ep_len, self.instr = (np.array(shard_idx), np.array(ep_start),
                                                                 np.array(ep_len), instr)
        self.frame_ep = np.array(frame_ep, dtype=np.int64)
        self.ep_first = np.concatenate([[0], np.cumsum(self.ep_len)[:-1]]) if len(self.ep_len) else np.array([])

    def __len__(self) -> int:
        return len(self.frame_ep)

    def _file(self, si: int):
        if si not in self._files:  # per-process lazy handle (DataLoader workers each open their own)
            self._files[si] = h5py.File(self.root / self.shards[si], "r")
        return self._files[si]

    def __getitem__(self, i: int) -> dict:
        e = int(self.frame_ep[i])
        t = i - int(self.ep_first[e])
        f = self._file(int(self.ep_shard[e]))
        row = self.ep_start[e] + t
        end = self.ep_start[e] + self.ep_len[e]
        acts = f["actions"][row:min(row + self.chunk, end)]
        mask = np.zeros(self.chunk, np.float32)
        mask[: len(acts)] = 1
        if len(acts) < self.chunk:  # pad with the last action, flagged by the mask
            acts = np.concatenate([acts, np.repeat(acts[-1:], self.chunk - len(acts), 0)])
        return {"image": torch.from_numpy(f["images"][row]),
                "proprio": torch.from_numpy((f["proprio"][row] - self.p_mean) / self.p_std),
                "actions": torch.from_numpy(acts),
                "action_mask": torch.from_numpy(mask), "instruction": self.instr[e]}


def make_collate(tokenizer):
    def collate(items: list[dict]) -> dict:
        out = {k: torch.stack([it[k] for it in items]) for k in ("image", "proprio", "actions", "action_mask")}
        out.update(encode_instructions(tokenizer, [it["instruction"] for it in items]))
        return out

    return collate
