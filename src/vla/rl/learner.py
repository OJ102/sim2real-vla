"""Learner loop: consume rollout batches, GRPO update (PPO-clip x TIS + k3 KL to the frozen BC reference), sync weights.

Per iteration j (version j): wait for one batch file per rollout rank -> advantages (group-relative) -> pi_old and
pi_ref log-probs -> `epochs` passes of minibatch PPO -> if j < K-S broadcast version j+1 to the rollout ranks."""
from __future__ import annotations

import contextlib
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from vla.model.prompts import encode_instructions
from vla.rl.algo import group_advantages, grpo_loss, staleness_hist
from vla.rl.rollout_worker import batch_path
from vla.rl.weight_sync import sync_weights
from vla.train.checkpoint import export_final


def wait_for(paths: list[Path], timeout: float, log) -> float:
    t0, last = time.perf_counter(), time.perf_counter()
    while not all(p.exists() for p in paths):
        time.sleep(0.2)
        if time.perf_counter() - last > 60:
            log(f"[learner] heartbeat: waiting for {[p.name for p in paths if not p.exists()]} "
                f"({time.perf_counter()-t0:.0f}s)")
            last = time.perf_counter()
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError(f"rollout batch not produced within {timeout}s: {paths}")
    return time.perf_counter() - t0


def load_batches(paths: list[Path], G: int):
    """-> flat sample arrays + per-sample advantage/version, and per-episode stats for logging."""
    cols = {k: [] for k in ("image", "proprio", "bins", "logp", "adv", "version", "instr")}
    ep_stats = {"ret": [], "succ": [], "comp": {}}
    metas = []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        R = torch.from_numpy(d["ep_return"])
        adv_ep = group_advantages(R.view(-1, G)).view(-1).numpy()
        ep = d["ep"]
        cols["image"].append(d["image"]); cols["proprio"].append(d["proprio"]); cols["bins"].append(d["bins"])
        cols["logp"].append(d["logp"]); cols["adv"].append(adv_ep[ep])
        cols["version"].append(np.full(len(ep), int(d["version"]))); cols["instr"].extend(d["instr"][ep].tolist())
        ep_stats["ret"].append(d["ep_return"]); ep_stats["succ"].append(d["ep_success"])
        for k in d.files:
            if k.startswith("comp_"):
                ep_stats["comp"].setdefault(k[5:], []).append(d[k])
        metas.append(json.loads(str(d["meta"])))
    out = {k: (np.concatenate(v) if k != "instr" else v) for k, v in cols.items()}
    return out, ep_stats, metas


def learner_loop(cfg, roles, rank, device, model, ref, learner_pg, sync_pg, shm: Path, logger, log=print):
    cuda = device.type == "cuda"
    lr_i = roles.learner_ranks.index(rank)
    nL = len(roles.learner_ranks)
    is_src = rank == roles.sync_src
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0, betas=(0.9, 0.95))
    S, K = cfg.staleness, cfg.iterations
    tok = model.tokenizer if hasattr(model, "tokenizer") else None
    base = model.module if hasattr(model, "module") else model
    tok = base.tokenizer
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else contextlib.nullcontext()

    def mk(batch_np, idx):
        b = {"image": torch.from_numpy(batch_np["image"][idx]).to(device),
             "proprio": torch.from_numpy(batch_np["proprio"][idx]).to(device),
             "bins": torch.from_numpy(batch_np["bins"][idx]).to(device),
             **{k: v.to(device) for k, v in encode_instructions(tok, [batch_np["instr"][i] for i in idx]).items()}}
        return b

    info = sync_weights(model, sync_pg if is_src else None, roles.sync_src, 0, sync_dtype=cfg.sync_dtype_t)
    log(f"[learner] initial sync done {info['latency_ms']:.0f} ms")
    tot_tokens = 0
    t_run = time.perf_counter()
    for j in range(K):
        t_iter = time.perf_counter()
        paths = [batch_path(shm, r, j) for r in roles.rollout_ranks]
        waited = wait_for(paths, cfg.wait_timeout_s, log)
        data, es, metas = load_batches(paths, cfg.group_size)
        N = len(data["bins"])
        usable = (N // (nL * cfg.micro_batch)) * (nL * cfg.micro_batch)
        order = np.random.default_rng(cfg.seed * 1000 + j).permutation(N)[:usable]
        mine = order[lr_i::nL]  # this rank's shard (equal size on all learner ranks)
        t_upd = time.perf_counter()
        # pi_old (learner weights at the start of the update) and pi_ref, in micro-batches
        lp_old, lp_ref = [], []
        with torch.no_grad(), amp:
            model.eval()
            for s in range(0, len(mine), cfg.micro_batch):
                b = mk(data, mine[s:s + cfg.micro_batch])
                lp_old.append(model(b)["logp"].float())
                lp_ref.append(ref.token_logprobs(b, b["bins"]).float())
            model.train()
        lp_old, lp_ref = torch.cat(lp_old), torch.cat(lp_ref)
        beh = torch.from_numpy(data["logp"][mine]).to(device)
        adv = torch.from_numpy(data["adv"][mine]).float().to(device)
        ver = data["version"][mine]
        eq = torch.from_numpy(ver == j).to(device)
        mism_eq = ((lp_old - beh).abs().mean(-1)[eq].mean().item() if bool(eq.any()) else float("nan"))
        mets = []
        n_mine = len(mine)
        mb = min(cfg.minibatch, n_mine)
        for ep in range(cfg.epochs):
            perm = np.random.default_rng(cfg.seed * 7 + j * 31 + ep).permutation(n_mine)
            for s in range(0, n_mine - mb + 1, mb):
                sel = perm[s:s + mb]
                opt.zero_grad(set_to_none=True)
                acc = {}
                for m0 in range(0, mb, cfg.micro_batch):
                    ms = sel[m0:m0 + cfg.micro_batch]
                    b = mk(data, mine[ms])
                    with amp:
                        lp_new = model(b)["logp"].float()
                    t_ = torch.from_numpy(ms).to(device)
                    loss, m = grpo_loss(lp_new, lp_old[t_], beh[t_], lp_ref[t_], adv[t_], None, cfg.clip_eps,
                                        cfg.beta, cfg.tis_cap)
                    (loss * len(ms) / mb).backward()
                    for k, v in m.items():
                        acc.setdefault(k, []).append(v)
                gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                m = {k: float(np.mean(v)) for k, v in acc.items()}
                m["grad_norm"] = float(gn)
                mets.append(m)
        upd_time = time.perf_counter() - t_upd
        tot_tokens += n_mine * cfg.epochs * base.n_act_tokens * nL
        # weight sync: after update j for j < K-S (rollouts take exactly K-S syncs)
        sync_ms = 0.0
        if j < K - S:
            t_s = time.perf_counter()
            info = sync_weights(model, sync_pg if is_src else None, roles.sync_src, j + 1, sync_dtype=cfg.sync_dtype_t)
            sync_ms = (time.perf_counter() - t_s) * 1e3
        it_time = time.perf_counter() - t_iter
        if rank == roles.learner_ranks[0]:
            agg = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}
            ret, succ = np.concatenate(es["ret"]), np.concatenate(es["succ"])
            rec = {"iter": j, "return_mean": float(ret.mean()), "return_std": float(ret.std()),
                   "success": float(succ.mean()), "n_samples": int(N),
                   **{f"R_{k}": float(np.concatenate(v).mean()) for k, v in es["comp"].items()},
                   **agg, "engine_mismatch_equal_version": mism_eq,
                   "staleness_hist": staleness_hist(torch.from_numpy(data["version"]), j),
                   "sync_latency_ms": sync_ms, "learner_tokens_per_s": n_mine * nL * cfg.epochs * base.n_act_tokens / upd_time,
                   "env_steps_per_s": float(sum(m["env_steps"] for m in metas) / max(m["gen_time"] for m in metas)),
                   "learner_idle_frac": waited / it_time, "rollout_idle_s": float(np.mean([m["idle_s"] for m in metas])),
                   "iter_time_s": it_time}
            logger.log(j, rec)
            log(f"[learner] it {j} ret {rec['return_mean']:.2f} succ {rec['success']:.2f} loss {agg['loss']:.4f} "
                f"kl {agg['kl']:.5f} tisw {agg['tis_w_mean']:.3f} idle {rec['learner_idle_frac']:.2f} {it_time:.1f}s")
        for p in paths:
            if lr_i == 0:
                p.unlink(missing_ok=True)
        if cfg.ckpt_every and (j + 1) % cfg.ckpt_every == 0 and j + 1 < K:
            export_final(model, shm.parent / f"{shm.name}_iter{j+1}.pt", group=learner_pg)
    return {"tokens": tot_tokens, "secs": time.perf_counter() - t_run}
