"""Decoupled actor-learner GRPO entry point (one torchrun world = learner ranks + rollout ranks).

    torchrun --nproc_per_node=2 -m vla.rl.grpo --config configs/rl.yaml [key=value ...]       # 1 learner + 1 rollout
    S (staleness) = 0: serial on-policy; S = 1: rollout k+1 overlaps learner update k (one-step off-policy + TIS).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from vla.model.vla import VLA, VLAConfig
from vla.rl.learner import learner_loop
from vla.rl.rollout_worker import rollout_loop
from vla.rl.roles import init_sync_group, make_roles
from vla.sim.rewards import DEFAULT_WEIGHTS
from vla.train.checkpoint import export_final
from vla.train.dist import init_dist, wrap_fsdp2
from vla.utils.config import dump_config, load_config
from vla.utils.logging import MetricsLogger


@dataclass
class RLConfig:
    name: str = "rl"
    model_config: str = "configs/model_0p5b.yaml"
    head: str = "discrete"
    init_ckpt: str = ""                 # BC final.pt / DCP dir (identical start on all roles; else initial sync only)
    data_dir: str = ""                  # for proprio stats.json
    runs_dir: str = ""
    shm_dir: str = "/dev/shm/vla_rl"
    n_learner: int = 1
    n_rollout: int = 1
    staleness: int = 0                  # S
    iterations: int = 100
    tasks: list = field(default_factory=lambda: ["lift", "push"])
    seeds_per_iter: int = 8             # per rollout rank
    group_size: int = 8                 # G
    temperature: float = 1.0
    engine_mode: str = "eager"          # rollout engine: eager | static | graph (CUDA graphs; set graph on GPU)
    max_prompt_tokens: int = 32         # static-cache sizing: fixed prefix budget so one CUDA graph serves all batches
    lr: float = 2.0e-6
    epochs: int = 2
    minibatch: int = 64
    micro_batch: int = 16
    clip_eps: float = 0.2
    beta: float = 0.02
    tis_cap: float = 2.0
    grad_clip: float = 1.0
    seed: int = 0
    seed0: int = 500000                 # RL train seeds (demo seeds are 100000-199999, eval seeds < 100000)
    reward_weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    sync_dtype: str = "bfloat16"
    mem_cap_gb: float = 20.0            # per process (shared GPUs)
    ckpt_every: int = 10
    wait_timeout_s: float = 1800.0

    @property
    def n_envs(self) -> int:
        return self.seeds_per_iter * self.group_size

    @property
    def sync_dtype_t(self):
        return getattr(torch, self.sync_dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(RLConfig, a.config, a.overrides)
    assert cfg.temperature > 0, "temperature must be > 0 (one cfg.temperature drives both rollout sampling and learner log-probs)"
    rank, world, local, device = init_dist()
    roles = make_roles(cfg.n_learner, cfg.n_rollout)
    assert world == roles.world_size, f"world {world} != {cfg.n_learner}+{cfg.n_rollout}"
    role = roles.role_of(rank)
    cuda = device.type == "cuda"
    if cuda and cfg.mem_cap_gb:
        total = torch.cuda.get_device_properties(device).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, cfg.mem_cap_gb / total), device)
    run = Path(cfg.runs_dir or os.environ.get("VLA_RUNS", "runs")) / cfg.name
    shm = Path(cfg.shm_dir) / cfg.name
    if rank == 0:
        run.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(shm, ignore_errors=True)
        shm.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, str(run / "config.yaml"))
    dist.barrier()
    sync_pg = init_sync_group(roles)
    learner_pg = dist.new_group(ranks=list(roles.learner_ranks))
    log = lambda s: print(s, flush=True)  # noqa: E731

    mcfg = load_config(VLAConfig, cfg.model_config, [f"head={cfg.head}"])
    torch.manual_seed(cfg.seed)
    model = VLA(mcfg)
    if cfg.init_ckpt:
        from vla.infer.policy import load_weights

        load_weights(model, cfg.init_ckpt)
    stats = None
    if cfg.data_dir:
        stats = json.load(open(Path(cfg.data_dir) / "stats.json"))
    p_mean = np.array(stats["proprio_mean"], np.float32) if stats else None
    p_std = np.array(stats["proprio_std"], np.float32) if stats else None

    if role == "rollout":
        model.to(device, torch.bfloat16 if cuda else torch.float32).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        rollout_loop(cfg, roles, rank, device, model, sync_pg, p_mean, p_std, shm, log)
    else:
        model.to(device)
        ref = copy.deepcopy(model).to(torch.bfloat16 if cuda else torch.float32).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        for p in model.vision.parameters():
            p.requires_grad_(False)  # vision encoder frozen
        if len(roles.learner_ranks) > 1:
            from torch.distributed.device_mesh import DeviceMesh

            mesh = DeviceMesh.from_group(learner_pg, "cuda" if cuda else "cpu")
            model = wrap_fsdp2(model, bf16=cuda, mesh=mesh)
        logger = MetricsLogger(run, enabled=rank == roles.learner_ranks[0])
        model.train()
        res = learner_loop(cfg, roles, rank, device, model, ref, learner_pg, sync_pg, shm, logger, log)
        export_final(model, run / "final.pt", group=learner_pg)
        logger.close()
        if rank == roles.learner_ranks[0]:
            log(f"[learner] done: {cfg.iterations} iterations, {res['tokens']/res['secs']:.0f} tokens/s overall")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
