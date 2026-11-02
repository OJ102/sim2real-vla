"""VLA: SigLIP vision + 2-layer MLP projector + Qwen2 LM, with a discrete-token or flow-matching action head.

Sequence: [N image tokens][1 proprio token][prompt "In: {instr}\\nOut:" (left-padded)] then, for the discrete head,
H*act_dim action tokens. Discrete loss = CE over the 256 action rows only (equivalent to masking logits to the
action vocabulary, without materialising a [B, 16, 152k] logits tensor). Flow head: the LM sees only the prefix and
a DiT cross-attends to its last hidden states.

    python -m vla.model.vla --config configs/model_0p5b.yaml --count-params
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from vla.utils.config import load_config

from .action_tokenizer import ActionTokenizer
from .flow_head import FlowDiT, euler_sample, flow_loss, gaussian_logprob, sde_step_stats
from .prompts import ByteTokenizer, HFTokenizer


@dataclass
class FlowConfig:
    dim: int = 512
    layers: int = 6
    heads: int = 8
    steps: int = 10


@dataclass
class VLAConfig:
    tiny: bool = False                      # random-init 2-layer towers, byte tokenizer, no downloads
    pretrained: bool = True                 # load HF weights (ignored when tiny)
    vision_name: str = "google/siglip-base-patch16-224"
    llm_name: str = "Qwen/Qwen2.5-0.5B"
    head: str = "discrete"                  # discrete | flow
    attn_impl: str = "sdpa"                 # eager | sdpa | flash_attention_2
    chunk: int = 4
    act_dim: int = 4
    proprio_dim: int = 11
    n_bins: int = 256
    prefix_order: str = "img_first"         # img_first: [img, proprio, txt] (default) | txt_first: [txt, img, proprio]
    ground_weight: float = 0.0              # >0: aux target-xy grounding head at the last prefix token (BC training only)
    image_size: int = 224                   # images are resized to this before the vision tower
    flow: FlowConfig = field(default_factory=FlowConfig)


class ActionLMHead(nn.Module):
    """Logits over only the action-token rows of the (tied) output embedding.

    copy=False: slice view of the live LM weight (used for training: gradients reach the embedding).
    copy=True : inference-only copy of the 256 rows (0.23M params instead of ~136M for the full head).
    """

    def __init__(self, llm: nn.Module, base: int, n: int = 256, copy: bool = False):
        super().__init__()
        self.base, self.n = base, n
        if copy:
            self.register_buffer("w", llm.get_output_embeddings().weight[base:base + n].detach().clone())
            self._llm = None
        else:
            self._llm = [llm]  # list: do not register as a submodule

    def weight(self) -> torch.Tensor:
        if self._llm is None:
            return self.w
        return self._llm[0].get_output_embeddings().weight[self.base:self.base + self.n]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.weight().to(h.dtype))


class VLA(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        from transformers import (AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig,
                                  SiglipVisionModel)

        self.cfg = cfg
        if cfg.prefix_order not in ("img_first", "txt_first"):
            raise ValueError(f"prefix_order must be img_first|txt_first, got {cfg.prefix_order}")
        impl = cfg.attn_impl
        if cfg.tiny:
            vcfg = SiglipVisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                      image_size=cfg.image_size, patch_size=16, vision_use_head=False)
            self.vision = SiglipVisionModel._from_config(vcfg, attn_implementation=impl)
            lcfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
                               max_position_embeddings=1024)
            self.llm = Qwen2ForCausalLM._from_config(lcfg, attn_implementation=impl)
            self.tokenizer = ByteTokenizer()
        else:
            if cfg.pretrained:
                vc = SiglipVisionConfig.from_pretrained(cfg.vision_name)
                vc.vision_use_head = False  # attention-pooling head unused (we use all patch tokens)
                self.vision = SiglipVisionModel.from_pretrained(cfg.vision_name, config=vc, attn_implementation=impl,
                                                             dtype=torch.float32)
                self.llm = AutoModelForCausalLM.from_pretrained(cfg.llm_name, attn_implementation=impl,
                                                              dtype=torch.float32)
            else:  # random init with real shapes (param counting, no weight download)
                vc = AutoConfig.from_pretrained(cfg.vision_name).vision_config
                vc.vision_use_head = False
                self.vision = SiglipVisionModel._from_config(vc, attn_implementation=impl)
                self.llm = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(cfg.llm_name),
                                                            attn_implementation=impl)
            self.tokenizer = HFTokenizer(cfg.llm_name)
        d_llm, d_v = self.llm.config.hidden_size, self.vision.config.hidden_size
        # new action tokens appended after the full embedding matrix (rows beyond the tokenizer vocab are unused)
        self.act_base = self.llm.get_input_embeddings().weight.shape[0]
        self.llm.resize_token_embeddings(self.act_base + cfg.n_bins)
        self.projector = nn.Sequential(nn.Linear(d_v, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))
        self.proprio_mlp = nn.Sequential(nn.Linear(cfg.proprio_dim, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))
        self.act_tok = ActionTokenizer(cfg.n_bins, self.act_base)
        self.act_head = ActionLMHead(self.llm, self.act_base, cfg.n_bins)
        self.n_act_tokens = cfg.chunk * cfg.act_dim
        # created ONLY when used so checkpoints trained without it still load with strict=True
        self.ground_head = (nn.Sequential(nn.Linear(d_llm, 256), nn.GELU(), nn.Linear(256, 2))
                            if cfg.ground_weight > 0 else None)
        self.dit = None
        if cfg.head == "flow":
            self.dit = FlowDiT(d_llm, cfg.act_dim, cfg.chunk, cfg.proprio_dim, cfg.flow.dim, cfg.flow.layers,
                               cfg.flow.heads)
        elif cfg.head != "discrete":
            raise ValueError(cfg.head)

    # ------------------------------------------------------------------ inputs
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """uint8 [B,H,W,3] (or float [B,3,H,W] in 0..1) -> normalised float [B,3,S,S]."""
        if image.dtype == torch.uint8:
            image = image.permute(0, 3, 1, 2).float() / 255.0
        s = self.cfg.image_size
        if image.shape[-1] != s:
            image = F.interpolate(image, size=(s, s), mode="bilinear", align_corners=False)
        return (image - 0.5) / 0.5

    def embed_prefix(self, batch: dict):
        emb = self.llm.get_input_embeddings()
        dtype = emb.weight.dtype
        img = self.vision(pixel_values=self.preprocess_image(batch["image"]).to(dtype)).last_hidden_state
        img = self.projector(img)
        pro = self.proprio_mlp(batch["proprio"].to(dtype))[:, None]
        txt = emb(batch["input_ids"])
        b = img.shape[0]
        ones = torch.ones(b, img.shape[1] + 1, dtype=torch.long, device=img.device)  # image tokens + proprio token
        if self.cfg.prefix_order == "txt_first":
            # [txt (left-padded), img, proprio]: with causal attention every image token can condition on the
            # instruction (the colour word). Position P-1 is the proprio token, still the action/ground read-out.
            return torch.cat([txt, img, pro], dim=1), torch.cat([batch["attention_mask"], ones], dim=1)
        return torch.cat([img, pro, txt], dim=1), torch.cat([ones, batch["attention_mask"]], dim=1)

    @staticmethod
    def _pos(mask: torch.Tensor) -> torch.Tensor:
        return (mask.cumsum(-1) - 1).clamp(min=0)

    # ------------------------------------------------------------------ training
    def forward(self, batch: dict) -> dict:
        """batch: image, proprio [B,P] (normalised), input_ids, attention_mask, actions [B,H,act_dim] in [-1,1]."""
        if "x_next" in batch:  # Flow-GRPO path: per-denoising-step Gaussian log-probs (+ means for the closed-form KL)
            return self.flow_rl_logp(batch)
        if "bins" in batch:  # RL path: per-token log-probs; goes through forward() so FSDP2 root hooks fire
            return {"logp": self.token_logprobs(batch, batch["bins"], batch.get("temperature", 1.0))}
        prefix, mask = self.embed_prefix(batch)
        P = prefix.shape[1]
        if self.cfg.head == "flow":
            h = self.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=self._pos(mask),
                               use_cache=False).last_hidden_state
            if "swap" in batch:  # counterfactual-colour rows have no valid action target: exclude from the action loss
                rows = flow_loss(self.dit, batch["actions"].to(h.dtype), batch["proprio"].to(h.dtype), h, mask,
                                 per_row=True)
                keep = 1.0 - batch["swap"].float()
                main = (rows * keep).sum() / keep.sum().clamp(min=1.0)
            else:
                main = flow_loss(self.dit, batch["actions"].to(h.dtype), batch["proprio"].to(h.dtype), h, mask)
            return self._with_ground({"main_loss": main}, h[:, P - 1], batch)
        bins = self.act_tok.encode_bins(batch["actions"].reshape(batch["actions"].shape[0], -1))  # [B,16]
        logits, h_last = self._discrete_logits(prefix, mask, bins)
        if "swap" in batch:
            keep = 1.0 - batch["swap"].float()
            den = keep.sum().clamp(min=1.0)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), bins.reshape(-1), reduction="none").view(bins.shape)
            main = (ce.mean(1) * keep).sum() / den
            acc = (((logits.argmax(-1) == bins).float().mean(1)) * keep).sum() / den
        else:
            main = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), bins.reshape(-1))
            acc = (logits.argmax(-1) == bins).float().mean()
        return self._with_ground({"main_loss": main, "tok_acc": acc}, h_last, batch)

    def _with_ground(self, out: dict, h_last: torch.Tensor, batch: dict) -> dict:
        """Add the (optional) target-xy grounding loss. h_last = hidden state at prefix position P-1, the only position
        that has seen both the colour word (text) and the image. Rows with ground_mask 0 do not contribute."""
        out["loss"] = out["main_loss"]
        if self.ground_head is None or "ground_xy" not in batch:
            return out
        pred = self.ground_head(h_last).float()
        gxy, gm = batch["ground_xy"].float(), batch["ground_mask"].float()
        den = gm.sum().clamp(min=1.0)
        out["ground_loss"] = (((pred - gxy) ** 2).mean(-1) * gm).sum() / den
        with torch.no_grad():
            gstd = batch.get("ground_std")
            d = (pred - gxy) * (gstd.float() if gstd is not None else 1.0)
            out["ground_err_cm"] = 100.0 * (d.norm(dim=-1) * gm).sum() / den  # metres -> cm when ground_std is given
        out["loss"] = out["main_loss"] + self.cfg.ground_weight * out["ground_loss"]
        return out

    def flow_rl_logp(self, batch: dict) -> dict:
        """batch: image/proprio/prompt + x [B,K,H,A] (state before each step), x_next, t (list of K), sde_a.
        Returns {"logp": [B,K], "mean": [B,K,H,A]} of the Gaussian transition under the CURRENT DiT. The backbone runs
        under no_grad unless batch["train_backbone"] (default: only the DiT is trained in flow RL)."""
        with torch.set_grad_enabled(bool(batch.get("train_backbone", False)) and torch.is_grad_enabled()):
            prefix, mask = self.embed_prefix(batch)
            h = self.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=self._pos(mask),
                               use_cache=False).last_hidden_state
        pro = batch["proprio"].to(h.dtype)
        ts, a = [float(t) for t in batch["t"]], float(batch["sde_a"])
        x, xn = batch["x"].float(), batch["x_next"].float()  # chain/log-prob math in fp32; only the DiT runs in bf16
        K = x.shape[1]
        logps, means = [], []
        for i in range(K):
            h_ = ts[i] - (ts[i + 1] if i + 1 < K else 0.0)
            v = self.dit(x[:, i], torch.full((x.shape[0],), ts[i], device=x.device, dtype=h.dtype), pro, h, mask).float()
            mean, std = sde_step_stats(x[:, i], v, ts[i], h_, a)
            logps.append(gaussian_logprob(xn[:, i], mean, std).float())
            means.append(mean.float())
        return {"logp": torch.stack(logps, 1), "mean": torch.stack(means, 1)}

    def _discrete_logits(self, prefix, mask, bins):
        """Teacher-forced (logits over the action rows [B, n_act_tokens, n_bins] float32, hidden state at P-1 [B, D])."""
        P = prefix.shape[1]
        a_emb = self.llm.get_input_embeddings()(bins + self.act_base)
        x = torch.cat([prefix, a_emb[:, :-1]], dim=1)
        m = torch.cat([mask, torch.ones(x.shape[0], self.n_act_tokens - 1, dtype=mask.dtype, device=mask.device)], 1)
        h = self.llm.model(inputs_embeds=x, attention_mask=m, position_ids=self._pos(m),
                           use_cache=False).last_hidden_state
        return self.act_head(h[:, P - 1:P - 1 + self.n_act_tokens]).float(), h[:, P - 1]

    def token_logprobs(self, batch: dict, bins: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Per-token log pi(bin | prefix, previous bins) over the action vocabulary: [B, n_act_tokens].
        Uses softmax(logits / temperature), the same distribution InferenceEngine samples from, so mu == pi_old."""
        prefix, mask = self.embed_prefix(batch)
        logp = F.log_softmax(self._discrete_logits(prefix, mask, bins)[0] / temperature, dim=-1)
        return logp.gather(-1, bins[..., None]).squeeze(-1)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, batch: dict, noise: torch.Tensor | None = None) -> torch.Tensor:
        """-> actions [B, chunk, act_dim] in [-1, 1]. Discrete: greedy over action rows with a KV cache."""
        prefix, mask = self.embed_prefix(batch)
        b = prefix.shape[0]
        if self.cfg.head == "flow":
            h = self.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=self._pos(mask)).last_hidden_state
            return euler_sample(self.dit, batch["proprio"].to(h.dtype), h, mask, self.cfg.flow.steps, noise)
        out = self.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=self._pos(mask), use_cache=True)
        cache, h = out.past_key_values, out.last_hidden_state[:, -1]
        emb = self.llm.get_input_embeddings()
        toks = []
        for i in range(self.n_act_tokens):
            tok = self.act_head(h).argmax(-1)
            toks.append(tok)
            if i == self.n_act_tokens - 1:
                break
            mask = torch.cat([mask, torch.ones(b, 1, dtype=mask.dtype, device=mask.device)], 1)
            out = self.llm.model(inputs_embeds=emb(tok + self.act_base)[:, None], attention_mask=mask,
                                 position_ids=self._pos(mask)[:, -1:], past_key_values=cache, use_cache=True)
            cache, h = out.past_key_values, out.last_hidden_state[:, -1]
        bins = torch.stack(toks, 1)
        return self.act_tok.decode_bins(bins.float()).reshape(b, self.cfg.chunk, self.cfg.act_dim)


def count_params(model: nn.Module) -> dict:
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    out = {"total": n(model), "vision": n(model.vision), "llm": n(model.llm),
           "projector": n(model.projector), "proprio_mlp": n(model.proprio_mlp)}
    if model.dit is not None:
        out["flow_dit"] = n(model.dit)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--count-params", action="store_true")
    ap.add_argument("overrides", nargs="*", help="key=value overrides")
    a = ap.parse_args()
    cfg = load_config(VLAConfig, a.config, a.overrides)
    if a.count_params:
        cfg.pretrained = False  # shapes only, no weight download
    model = VLA(cfg)
    c = count_params(model)
    print({k: f"{v/1e6:.2f}M" for k, v in c.items()})
    print(f"total params: {c['total']/1e9:.3f}B")
