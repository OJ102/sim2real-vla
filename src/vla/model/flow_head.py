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
    return F.mse_loss(v, eps - actions)


@torch.no_grad()
def euler_sample(dit: FlowDiT, proprio, ctx, ctx_mask, steps: int = 10, noise=None):
    b = proprio.shape[0]
    x = (torch.randn(b, dit.chunk, dit.act_dim, device=proprio.device, dtype=proprio.dtype) if noise is None
         else noise.clone().to(proprio.dtype))
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((b,), 1.0 - i * dt, device=proprio.device, dtype=proprio.dtype)
        x = x - dt * dit(x, t, proprio, ctx, ctx_mask)
    return x.clamp(-1, 1)
