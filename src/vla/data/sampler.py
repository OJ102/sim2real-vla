"""ResumableDistributedSampler: deterministic per (seed, epoch), sharded by rank, resumable mid-epoch.

Every rank draws the same global permutation; rank r takes indices[r::world_size] (after trimming to a multiple of
world_size). `consumed` = samples already used BY THIS RANK this epoch. DataLoader prefetching runs ahead of what
the trainer has actually consumed, so the trainer should checkpoint `state_dict(consumed=step_in_epoch*per_rank_batch)`.
"""
from __future__ import annotations

import torch
from torch.utils.data import Sampler


class ResumableDistributedSampler(Sampler):
    def __init__(self, n: int, rank: int = 0, world_size: int = 1, seed: int = 0, epoch: int = 0, consumed: int = 0,
                 shuffle: bool = True):
        self.n, self.rank, self.world_size, self.seed, self.epoch = n, rank, world_size, seed, epoch
        self.consumed, self.shuffle = consumed, shuffle
        self.num_per_rank = n // world_size  # drop the tail so all ranks do equal work

    def _indices(self) -> list[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed * 1_000_003 + self.epoch)
            idx = torch.randperm(self.n, generator=g).tolist()
        else:
            idx = list(range(self.n))
        return idx[: self.num_per_rank * self.world_size][self.rank::self.world_size]

    def __iter__(self):
        idx = self._indices()
        while self.consumed < len(idx):
            i = idx[self.consumed]
            self.consumed += 1
            yield i

    def __len__(self) -> int:
        return self.num_per_rank - self.consumed

    def set_epoch(self, epoch: int) -> None:
        self.epoch, self.consumed = epoch, 0

    def state_dict(self, consumed: int | None = None) -> dict:
        return {"seed": self.seed, "epoch": self.epoch, "consumed": self.consumed if consumed is None else consumed,
                "n": self.n, "world_size": self.world_size}

    def load_state_dict(self, sd: dict) -> None:
        assert sd["n"] == self.n and sd["world_size"] == self.world_size, "dataset size / world size changed"
        self.seed, self.epoch, self.consumed = sd["seed"], sd["epoch"], sd["consumed"]
