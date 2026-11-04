"""Batched VLA policy for closed-loop eval: loads a final.pt / DCP checkpoint, queries action chunks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from vla.model.prompts import encode_instructions
from vla.model.vla import VLA, VLAConfig
from vla.utils.config import load_config


def load_weights(model: VLA, path: str) -> None:
    """path: final.pt, a DCP step dir, or a ckpt dir containing `latest`."""
    p = Path(path)
    if p.is_file():
        sd = torch.load(p, map_location="cpu")
        if getattr(model, "ground_head", None) is None:
            # the ground head is a BC-only aux loss; RL / inference build the model with ground_weight=0
            dropped = [k for k in sd if k.startswith("ground_head.")]
            if dropped:
                sd = {k: v for k, v in sd.items() if not k.startswith("ground_head.")}
                print(f"load_weights: dropped {len(dropped)} ground_head.* keys (model has no ground head); "
                      f"all other keys strict", flush=True)
        missing, unexpected = model.load_state_dict({k: v.float() if v.is_floating_point() else v
                                                     for k, v in sd.items()}, strict=False)
        bad = [k for k in missing if not k.startswith("act_head")]
        if bad or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={bad[:5]} unexpected={unexpected[:5]}")
        return
    import torch.distributed.checkpoint as dcp

    if (p / "latest").exists():
        p = p / (p / "latest").read_text().strip()
    state = {"model": model.state_dict()}
    dcp.load(state, checkpoint_id=str(p))
    model.load_state_dict(state["model"])


class VLAPolicy:
    """act(images [B,H,W,3] u8, proprio [B,11] raw, instructions) -> actions [B,chunk,4] in [-1,1]."""

    def __init__(self, model_config: str, ckpt: str | None, stats_path: str | None, head: str = "",
                 device: str = "cpu", dtype=torch.float32, overrides: list[str] | None = None,
                 temperature: float | None = None):
        ov = ([f"head={head}"] if head else []) + (overrides or [])
        self.cfg = load_config(VLAConfig, model_config, ov)
        self.model = VLA(self.cfg)
        if ckpt:
            load_weights(self.model, ckpt)
        self.device, self.dtype = torch.device(device), dtype
        self.model.to(self.device, dtype).eval()
        self.p_mean = self.p_std = None
        if stats_path:
            st = json.load(open(stats_path))
            self.p_mean = np.array(st["proprio_mean"], np.float32)
            self.p_std = np.array(st["proprio_std"], np.float32)
        # temperature=None: greedy via model.predict (unchanged path). Discrete + T: sample softmax(logits/T)
        self.engine = None
        if temperature is not None and self.cfg.head == "discrete":
            if not temperature > 0:
                raise ValueError(f"temperature must be > 0, got {temperature}")
            from vla.infer.engine import InferenceEngine
            # same static-cache budget as RL rollouts (rl/rollout_worker.py, GRPOConfig.max_prompt_tokens=32)
            self.engine = InferenceEngine(self.model, temperature=temperature, mode="static",
                                          max_prefix=(self.cfg.image_size // 16) ** 2 + 1 + 32)

    def make_generator(self, seed: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(seed)

    @torch.no_grad()
    def act(self, images: np.ndarray, proprio: np.ndarray, instructions: list[str], greedy: bool = True,
            generator: torch.Generator | None = None) -> np.ndarray:
        if self.p_mean is not None:
            proprio = (proprio - self.p_mean) / self.p_std
        batch = {"image": torch.from_numpy(images).to(self.device),
                 "proprio": torch.from_numpy(proprio.astype(np.float32)).to(self.device),
                 **{k: v.to(self.device) for k, v in encode_instructions(self.model.tokenizer, instructions).items()}}
        if self.engine is not None:
            return self.engine.generate(batch, greedy=False, generator=generator)["actions"].float().cpu().numpy()
        return self.model.predict(batch).float().cpu().numpy()
