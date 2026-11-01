"""RL math (pure torch, CPU-testable): group advantages, truncated importance sampling, PPO-clip, KL k3.

Notation: logp_new = current learner log-prob (with grad); logp_old = learner log-prob recomputed at the start of the
update (pi_old, detached); logp_beh = behaviour log-prob mu recorded by the rollout engine. mu may differ from pi_old
by (a) policy staleness and (b) inference-vs-training numerics; TIS corrects both with w = min(pi_old / mu, cap).
"""
from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """rewards [n_groups, G] -> (R - group mean) / (group std + eps), same shape."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, unbiased=False, keepdim=True)
    return (rewards - mean) / (std + eps)


def tis_weight(logp_old: torch.Tensor, logp_beh: torch.Tensor, cap: float = 2.0) -> torch.Tensor:
    """Detached truncated importance weight min(pi_old/mu, cap). Exactly 1 when logp_old == logp_beh."""
    return torch.exp(logp_old.detach() - logp_beh.detach()).clamp(max=cap)


def kl_k3(logp_new: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Unbiased, non-negative per-token KL(pi_new || pi_ref) estimator (Schulman k3), samples from pi_new."""
    d = logp_ref - logp_new
    return torch.exp(d) - d - 1


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return x.mean() if mask is None else (x * mask).sum() / mask.sum().clamp(min=1)


def grpo_loss(logp_new, logp_old, logp_beh, logp_ref, adv, mask=None, clip_eps: float = 0.2, beta: float = 0.02,
              tis_cap: float = 2.0, use_tis: bool = True):
    """All tensors [B, T] except adv [B] or [B, T] (episode advantage broadcast over tokens). Returns (loss, metrics)."""
    if adv.dim() == 1:
        adv = adv[:, None].expand_as(logp_new)
    ratio = torch.exp(logp_new - logp_old.detach())
    pg = torch.minimum(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
    w = tis_weight(logp_old, logp_beh, tis_cap) if use_tis else torch.ones_like(ratio)
    kl = kl_k3(logp_new, logp_ref.detach())
    loss = _masked_mean(-pg * w + beta * kl, mask)
    with torch.no_grad():
        raw = torch.exp(logp_old - logp_beh)
        metrics = {
            "loss": loss.item(),
            "kl": _masked_mean(kl, mask).item(),
            "ratio_clipfrac": _masked_mean(((ratio - 1).abs() > clip_eps).float(), mask).item(),
            "tis_w_mean": _masked_mean(w, mask).item(),
            "tis_w_max": (w if mask is None else w[mask.bool()]).max().item(),
            "tis_clipfrac": _masked_mean((raw > tis_cap).float(), mask).item(),
            # inference-vs-learner numerics gap; meaningful when behaviour and learner policy versions are equal
            "engine_mismatch": _masked_mean((logp_old - logp_beh).abs(), mask).item(),
        }
    return loss, metrics


def staleness_hist(batch_versions: torch.Tensor, learner_version: int) -> dict[int, int]:
    """Histogram of (learner_version - policy version used to generate each episode)."""
    gaps = (learner_version - batch_versions).tolist()
    out: dict[int, int] = {}
    for g in gaps:
        out[int(g)] = out.get(int(g), 0) + 1
    return out
