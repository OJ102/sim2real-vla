"""Rank roles for the decoupled actor-learner world (one torchrun world, learner ranks + rollout ranks)."""
from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist


@dataclass(frozen=True)
class Roles:
    learner_ranks: tuple[int, ...]
    rollout_ranks: tuple[int, ...]

    @property
    def world_size(self) -> int:
        return len(self.learner_ranks) + len(self.rollout_ranks)

    @property
    def sync_src(self) -> int:
        """Global rank that broadcasts weights (learner rank 0)."""
        return self.learner_ranks[0]

    @property
    def sync_ranks(self) -> tuple[int, ...]:
        return (self.sync_src, *self.rollout_ranks)

    def role_of(self, rank: int) -> str:
        if rank in self.learner_ranks:
            return "learner"
        if rank in self.rollout_ranks:
            return "rollout"
        raise ValueError(f"rank {rank} has no role in {self}")


def make_roles(n_learner: int, n_rollout: int) -> Roles:
    """Learners take ranks [0, n_learner), rollout workers the rest."""
    if n_learner < 1 or n_rollout < 1:
        raise ValueError("need >=1 learner and >=1 rollout rank")
    return Roles(tuple(range(n_learner)), tuple(range(n_learner, n_learner + n_rollout)))


def init_sync_group(roles: Roles, backend: str | None = None):
    """Process group {learner rank 0} + all rollout ranks. Collective: every rank in the world must call it."""
    assert dist.get_world_size() == roles.world_size, (dist.get_world_size(), roles)
    return dist.new_group(ranks=list(roles.sync_ranks), backend=backend)
