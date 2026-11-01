"""Colour-grounding probe (PLAN_v2_grounding step 7): does the policy read the colour word at the first frame?

    uv run python scripts/probe_grounding.py --ckpt $VLA_RUNS/bc_gdisc/best.pt --data $VLA_DATA/demos \
        --model-config configs/model_0p5b.yaml --head discrete --ground-weight 0.2 --out results/probe_bc_gdisc.json

Reads t=0 frames from the shards (300 train episodes + every 20th = val episodes), no simulator needed.
Per task x part: cos_expert (pred xy-delta vs mean of the first 8 expert xy actions), cos_geo (vs unit(target_xy - ee_xy)),
colour swap to each other colour with the image unchanged: swap_cos_new (vs heading to the swapped cube), swap_cos_orig
(vs original heading), swap_delta (mean |a_swap - a|); ground_err_cm_t0 when a ground head exists.
Models without a ground head (old bc_disc / bc_flow): ground_err_cm_t0 is skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COLORS = ("red", "green", "blue")


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def cos(a, b):
    return (unit(a) * unit(b)).sum(-1)


def load_t0(data: Path, n_train: int):
    """-> list of dict (one per episode, t=0 frame) with part train/val."""
    import h5py

    rows, g = [], 0
    for s in sorted(data.glob("*.h5")):
        with h5py.File(s, "r") as f:
            if "cube_pos" not in f:
                raise SystemExit(f"{s.name} has no cube_pos: regenerate the demos with the grounding-aware generate.py")
            starts, lens = f["ep_start"][:], f["ep_len"][:]
            ins = [x.decode() for x in f["instruction"][:]]
            tasks = [x.decode() for x in f["task"][:]]
            colors = [x.decode() for x in f["color"][:]]
            tidx = f["target_idx"][:]
            for e in range(len(lens)):
                part = "val" if g % 20 == 0 else "train"
                g += 1
                r = int(starts[e])
                a8 = f["actions"][r:r + min(8, int(lens[e])), :2]
                rows.append({"part": part, "task": tasks[e], "color": colors[e], "instr": ins[e], "tidx": int(tidx[e]),
                             "image": f["images"][r], "proprio": f["proprio"][r], "cube_pos": f["cube_pos"][r],
                             "a8": a8})
    train = [r for r in rows if r["part"] == "train"]
    rng = np.random.default_rng(0)
    keep = set(rng.permutation(len(train))[:n_train].tolist())
    return [r for i, r in enumerate(train) if i in keep] + [r for r in rows if r["part"] == "val"]


def swap_instr(instr: str, c: str, c2: str) -> str:
    return instr.replace(f"{c} cube", f"{c2} cube")


def main():
    import torch

    from vla.infer.policy import load_weights
    from vla.model.prompts import encode_instructions
    from vla.model.vla import VLA, VLAConfig
    from vla.utils.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="final.pt / best.pt / DCP dir; random init if omitted (smoke tests)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-config", default="configs/model_0p5b.yaml")
    ap.add_argument("--head", default="")
    ap.add_argument("--ground-weight", type=float, default=0.0, help=">0 builds the ground head (needed to load such ckpts)")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    dev = torch.device(a.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    gw = a.ground_weight
    if gw <= 0 and a.ckpt and Path(a.ckpt).is_file():  # auto-detect a ground head from the weights file
        if any(k.startswith("ground_head.") for k in torch.load(a.ckpt, map_location="cpu", mmap=True)):
            gw = 1.0
            print("ground head detected in checkpoint: building it (ground_weight only matters for training)")
    ov = ([f"head={a.head}"] if a.head else []) + ([f"ground_weight={gw}"] if gw > 0 else []) + a.override
    model = VLA(load_config(VLAConfig, a.model_config, ov))
    if a.ckpt:
        load_weights(model, a.ckpt)
    model.to(dev, torch.bfloat16 if dev.type == "cuda" else torch.float32).eval()
    data = Path(a.data)
    st = json.load(open(data / "stats.json"))
    p_mean, p_std = np.array(st["proprio_mean"], np.float32), np.array(st["proprio_std"], np.float32)
    g_mean, g_std = np.array(st.get("target_xy_mean", [0, 0]), np.float32), np.array(st.get("target_xy_std", [1, 1]), np.float32)
    eps = load_t0(data, a.n_train)
    flow = model.cfg.head == "flow"

    @torch.no_grad()
    def predict(items, instrs):
        out_a, out_g = [], []
        for i in range(0, len(items), a.bs):
            sl = slice(i, i + a.bs)
            img = torch.from_numpy(np.stack([r["image"] for r in items[sl]])).to(dev)
            pro = torch.from_numpy(((np.stack([r["proprio"] for r in items[sl]]) - p_mean) / p_std).astype(np.float32)).to(dev)
            tok = {k: v.to(dev) for k, v in encode_instructions(model.tokenizer, instrs[sl]).items()}
            batch = {"image": img, "proprio": pro, **tok}
            noise = None
            if flow:  # fixed seeded noise so swaps differ only through the colour word
                g = torch.Generator().manual_seed(0)
                noise = torch.randn(len(img), model.cfg.chunk, model.cfg.act_dim, generator=g).to(dev)
            out_a.append(model.predict(batch, noise=noise).float().cpu().numpy())
            if model.ground_head is not None:
                prefix, mask = model.embed_prefix(batch)
                h = model.llm.model(inputs_embeds=prefix, attention_mask=mask, position_ids=model._pos(mask),
                                    use_cache=False).last_hidden_state[:, -1]
                out_g.append(model.ground_head(h).float().cpu().numpy())
        return np.concatenate(out_a), (np.concatenate(out_g) if out_g else None)

    base_a, base_g = predict(eps, [r["instr"] for r in eps])
    res = {"ckpt": a.ckpt, "head": model.cfg.head, "n_episodes": len(eps), "results": {}}
    swaps = {}
    for k, r in enumerate(eps):
        swaps[k] = [c2 for c2 in COLORS if c2 != r["color"]]
    swap_pred = {}
    for j in range(2):  # j-th other colour
        items = list(range(len(eps)))
        instrs = [swap_instr(eps[k]["instr"], eps[k]["color"], swaps[k][j]) for k in items]
        swap_pred[j], _ = predict(eps, instrs)
    for part in ("train", "val"):
        for task in sorted({r["task"] for r in eps}):
            idx = [k for k, r in enumerate(eps) if r["part"] == part and r["task"] == task]
            if not idx:
                continue
            ee = np.stack([eps[k]["proprio"][:2] for k in idx])
            tgt = np.stack([eps[k]["cube_pos"][eps[k]["tidx"], :2] for k in idx])
            exp_dir = np.stack([eps[k]["a8"][:, :2].mean(0) for k in idx])
            pred = base_a[idx][:, :, :2].mean(1)
            m = {"n": len(idx), "cos_expert": float(cos(pred, exp_dir).mean()),
                 "cos_geo": float(cos(pred, tgt - ee).mean())}
            sn, so, sd = [], [], []
            for j in range(2):
                sp = swap_pred[j][idx][:, :, :2].mean(1)
                new_t = np.stack([eps[k]["cube_pos"][COLORS.index(swaps[k][j]), :2] for k in idx])
                sn.append(cos(sp, new_t - ee).mean()); so.append(cos(sp, tgt - ee).mean())
                sd.append(np.abs(swap_pred[j][idx] - base_a[idx]).mean())
            m.update({"swap_cos_new": float(np.mean(sn)), "swap_cos_orig": float(np.mean(so)), "swap_delta": float(np.mean(sd))})
            if base_g is not None:
                pxy = base_g[idx] * g_std + g_mean
                m["ground_err_cm_t0"] = float(100 * np.linalg.norm(pxy - tgt, axis=-1).mean())
            res["results"].setdefault(part, {})[task] = m
    # gate on the val part (reach, lift): cos_expert>=0.9, swap_cos_new>=0.8, swap_cos_orig <= cos_expert-0.4
    gate = {}
    for task in ("reach", "lift"):
        m = res["results"].get("val", {}).get(task)
        if m:
            gate[task] = bool(m["cos_expert"] >= 0.9 and m["swap_cos_new"] >= 0.8 and m["swap_cos_orig"] <= m["cos_expert"] - 0.4)
    res["gate"] = {"per_task": gate, "pass": bool(gate) and all(gate.values())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["results"], indent=1))
    print("GATE", res["gate"])


if __name__ == "__main__":
    main()
