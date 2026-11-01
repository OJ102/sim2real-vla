"""Flow-matching action expert: DiT with adaLN timestep conditioning and cross-attention to backbone hidden states.

Rectified flow: x_t = t*eps + (1-t)*a, target velocity u = eps - a; t=1 is pure noise. Euler sampling t:1 -> 0.
Tokens: the noisy action chunk (H tokens of act_dim) + 1 proprio token.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None].float() * 1000 * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.ada = nn.Linear(d, 6 * d)  # shift/scale/gate for self-attn and mlp (adaLN-Zero)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c, ctx, ctx_pad):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c)[:, None].chunk(6, dim=-1)
        h = self.n1(x) * (1 + sc1) + s1
        x = x + g1 * self.self_attn(h, h, h, need_weights=False)[0]
        x = x + self.cross_attn(self.n2(x), ctx, ctx, key_padding_mask=ctx_pad, need_weights=False)[0]
        h = self.n3(x) * (1 + sc2) + s2
        return x + g2 * self.mlp(h)


class FlowDiT(nn.Module):
    def __init__(self, ctx_dim: int, act_dim: int = 4, chunk: int = 4, proprio_dim: int = 11, d: int = 512,
                 layers: int = 6, heads: int = 8):
        super().__init__()
        self.chunk, self.act_dim, self.d = chunk, act_dim, d
        self.act_in = nn.Linear(act_dim, d)
        self.proprio_in = nn.Linear(proprio_dim, d)
        self.pos = nn.Parameter(torch.randn(chunk + 1, d) * 0.02)
        self.ctx_proj = nn.Linear(ctx_dim, d)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(DiTBlock(d, heads) for _ in range(layers))
        self.out_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.out_ada = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, act_dim)
        nn.init.zeros_(self.out_ada.weight); nn.init.zeros_(self.out_ada.bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x_t, t, proprio, ctx, ctx_mask):
        """x_t [B,H,act_dim], t [B], proprio [B,P], ctx [B,S,ctx_dim], ctx_mask [B,S] (1=valid) -> velocity [B,H,act_dim]."""
        dt = self.act_in.weight.dtype  # compute dtype (bf16 under FSDP2 mixed precision): t / noise are created in fp32
        x_t, proprio, ctx = x_t.to(dt), proprio.to(dt), ctx.to(dt)
        x = torch.cat([self.act_in(x_t), self.proprio_in(proprio)[:, None]], dim=1) + self.pos
        c = self.t_mlp(timestep_embedding(t, self.d).to(x.dtype))
        ctx = self.ctx_proj(ctx)
        pad = ctx_mask == 0
        for blk in self.blocks:
            x = blk(x, c, ctx, pad)
        sh, sc = self.out_ada(c)[:, None].chunk(2, dim=-1)
        return self.out(self.out_norm(x[:, : self.chunk]) * (1 + sc) + sh)


def sample_t(batch: int, device, s: float = 0.999) -> torch.Tensor:
    """pi0 timestep density: Beta(1.5, 1) scaled by s; mass concentrated near t=1 (high noise)."""
    return s * torch.distributions.Beta(1.5, 1.0).sample((batch,)).to(device)


def flow_loss(dit: FlowDiT, actions, proprio, ctx, ctx_mask, t=None, noise=None):
    b = actions.shape[0]
    t = sample_t(b, actions.device) if t is None else t
    eps = torch.randn_like(actions) if noise is None else noise
    x_t = t[:, None, None] * eps + (1 - t[:, None, None]) * actions
    v = dit(x_t, t, proprio, ctx, ctx_mask)
    return F.mse_loss(v.float(), (eps - actions).float())  # loss in fp32


@torch.no_grad()
def euler_sample(dit: FlowDiT, proprio, ctx, ctx_mask, steps: int = 10, noise=None, ts=None):
    """ODE Euler from t=1 (noise) to t=0. `ts` (len steps+1, decreasing) overrides the uniform grid."""
    b = proprio.shape[0]
    x = (torch.randn(b, dit.chunk, dit.act_dim, device=proprio.device, dtype=proprio.dtype) if noise is None
         else noise.clone().to(proprio.dtype))
    if ts is None:
        ts = [1.0 - i / steps for i in range(steps + 1)]
    for i in range(len(ts) - 1):
        t = torch.full((b,), float(ts[i]), device=proprio.device, dtype=proprio.dtype)
        x = x - (ts[i] - ts[i + 1]) * dit(x, t, proprio, ctx, ctx_mask)
    return x.clamp(-1, 1)


# ------------------------------------------------------------------------------------------------ Flow-GRPO (SDE)
# Convention (same as above): x_t = (1-t) x0 + t eps, v = eps - x0, t=1 is noise. Flow-GRPO (Liu et al. 2025) turns the
# probability-flow ODE into the marginal-preserving reverse SDE
#   dx = [v + sigma_t^2/(2t) (x + (1-t) v)] dt + sigma_t dw   (dt < 0),  sigma_t = a * sqrt(t / (1 - t)),
# discretised with Euler-Maruyama over a grid of t values (decreasing, step h = t_i - t_{i+1} > 0):
#   x_{i+1} = x_i - h [v + sigma^2/(2t) (x_i + (1-t) v)] + sigma sqrt(h) * z,   z ~ N(0, I)
# so every step is Gaussian with a closed-form log-prob and KL. t=1 would give sigma=inf, so the grid starts at
# 1 - delta (delta = 1/(2K)) with x_init = eps (approximation: the true marginal at 1-delta is
# (delta x0 + (1-delta) eps)).

def sde_grid(K: int, delta: float | None = None) -> list[float]:
    delta = 1.0 / (2 * K) if delta is None else delta
    return [(1 - delta) * (1 - i / K) for i in range(K + 1)]  # t_0 = 1-delta ... t_K = 0 (never evaluated at 0)


def sde_sigma(t: float, a: float) -> float:
    return a * (t / (1.0 - t)) ** 0.5


def sde_step_stats(x, v, t: float, h: float, a: float):
    """Gaussian transition N(mean, std^2 I) of one Euler-Maruyama step: returns (mean, std)."""
    sig = sde_sigma(t, a)
    mean = x - h * (v + (sig * sig / (2 * t)) * (x + (1 - t) * v))
    return mean, sig * h ** 0.5


def gaussian_logprob(x, mean, std: float):
    """log N(x; mean, std^2 I) summed over the last two dims (chunk, act_dim) -> shape [B]."""
    d = x[0].numel()
    return (-0.5 * ((x - mean) / std) ** 2).sum((-2, -1)) - d * (torch.log(torch.tensor(std)) + 0.5 * math.log(2 * math.pi))


def gaussian_kl(mean_p, mean_q, std: float):
    """KL(N(mean_p, std^2) || N(mean_q, std^2)) summed over the last two dims: ||dmean||^2 / (2 std^2)."""
    return ((mean_p - mean_q) ** 2).sum((-2, -1)) / (2 * std * std)


@torch.no_grad()
def flow_sde_sample(dit: FlowDiT, proprio, ctx, ctx_mask, K: int = 5, a: float = 0.7, generator=None, noise=None,
                    ts=None):
    """SDE rollout. Returns dict: actions (clamped final latent), x [B,K,H,A] (state BEFORE each step),
    x_next [B,K,H,A], logp [B,K] (behaviour log-prob of each step), t list (len K), std list (len K)."""
    b = proprio.shape[0]
    dev, dt_ = proprio.device, proprio.dtype
    x = (torch.randn(b, dit.chunk, dit.act_dim, device=dev, dtype=dt_, generator=generator) if noise is None
         else noise.clone().to(dt_))
    ts = sde_grid(K) if ts is None else ts
    xs, xn, lps, stds = [], [], [], []
    for i in range(K):
        t, h = ts[i], ts[i] - ts[i + 1]
        v = dit(x, torch.full((b,), t, device=dev, dtype=dt_), proprio, ctx, ctx_mask)
        mean, std = sde_step_stats(x, v, t, h, a)
        z = torch.randn(x.shape, device=dev, dtype=dt_, generator=generator)
        x_next = mean + std * z if std > 0 else mean
        lps.append(gaussian_logprob(x_next, mean, std) if std > 0 else torch.zeros(b, device=dev, dtype=dt_))
        xs.append(x); xn.append(x_next); stds.append(std)
        x = x_next
    return {"actions": x.clamp(-1, 1), "x": torch.stack(xs, 1), "x_next": torch.stack(xn, 1),
            "logp": torch.stack(lps, 1), "t": ts[:K], "std": stds}
