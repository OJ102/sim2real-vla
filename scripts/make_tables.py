"""Render results/*.json into README markdown tables (+ plots). Missing inputs are skipped with a note.

    uv run python scripts/make_tables.py [--results results] [--runs runs_pulled] [--out results/tables.md] [--plots results/plots]

Inputs (all optional):
  eval_<policy>.json          rollout_eval output  -> policy table ("expert", "bc_disc", "bc_flow", "rl_disc_seed0", ...)
  expert_ceiling.json         expert success per split/task
  scaling.json, bench_bc_step.json, ckpt_stall.json   -> distributed-training tables
  latency.json                P7 benchmark         -> latency table
  ablations.json              {"runs": [{"name","split","task","success","ci95":[lo,hi]}]} -> ablation table + plot
  <runs>/*/metrics.jsonl      RL runs (iter/return_mean/success/kl...) -> learning curves
Every timing number carries its `contended` flag (shared GPUs).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from pathlib import Path

SPLITS = ["train", "heldout", "heldout_photo"]


def load(p):
    try:
        return json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None


def pct(r):
    return f"{100 * r['success']:.1f} [{100 * r['ci95'][0]:.0f},{100 * r['ci95'][1]:.0f}]"


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def flag(c):
    return " (contended)" if c else ""


# ------------------------------------------------------------------------------------------------ policy results
def policy_table(res: Path):
    files = sorted(glob.glob(str(res / "eval_*.json")))
    if not files:
        return "_No eval_*.json yet._", {}
    rows, data = [], {}
    seeds: dict[str, list] = {}
    for f in files:
        name = Path(f).stem[len("eval_"):]
        d = load(f)
        if not d:
            continue
        m = re.match(r"(.+)_seed\d+$", name)
        if m:
            seeds.setdefault(m.group(1), []).append(d)
        data[name] = d
        for task in sorted({t for s in d["results"].values() for t in s}):
            cells = []
            for sp in SPLITS:
                r = d["results"].get(sp, {}).get(task)
                cells.append(pct(r) + (f" ({100*r['success_vs_ceiling']:.0f}% of exp.)" if r and "success_vs_ceiling" in r else "")
                             if r else "-")
            tr = d["results"].get("train", {}).get(task) or next(iter(d["results"].values())).get(task, {})
            rows.append([name + flag(d["meta"].get("contended")), task, *cells,
                         f"{tr.get('ee_jerk_rms', float('nan')):.1f}",
                         f"{tr.get('joint_limit_viol_per_ep', 0) + tr.get('vel_viol_per_ep', 0) + tr.get('collision_viol_per_ep', 0):.2f}"])
    # aggregate multi-seed RL runs: mean +- std of success across seeds
    for base, ds in seeds.items():
        if len(ds) < 2:
            continue
        for task in sorted({t for d in ds for s in d["results"].values() for t in s}):
            cells = []
            for sp in SPLITS:
                v = [d["results"][sp][task]["success"] for d in ds if task in d["results"].get(sp, {})]
                cells.append(f"{100*statistics.mean(v):.1f} ± {100*statistics.pstdev(v):.1f} (n={len(v)} seeds)" if v else "-")
            rows.append([f"**{base} (mean±std)**", task, *cells, "", ""])
    hdr = ["Policy", "Task", "Train succ % [95% CI]", "Held-out succ %", "Held-out photo succ %", "EE jerk RMS", "Violations/ep"]
    return md(hdr, rows), data


# ------------------------------------------------------------------------------------------------ distributed training
def scaling_table(res: Path):
    d = load(res / "scaling.json")
    if not d:
        return "_No scaling.json yet._"
    rows = [[c["backend"], c["n_gpu"], "on" if c["act_ckpt"] else "off", f"{c['samples_per_s_median']:.0f} "
             f"({c['samples_per_s_min']:.0f}-{c['samples_per_s_max']:.0f})", f"{100*c['mfu_median']:.1f}%",
             f"{c['peak_mem_gb_max_rank']:.1f}", f"{c['scaling_efficiency']:.2f}" if c.get("scaling_efficiency") else "-",
             "yes" if c["any_contended"] else "no"] for c in d["configs"]]
    return md(["Backend", "GPUs", "Act. ckpt", "Samples/s median (min-max)", "MFU", "Peak GB/rank", "Scaling eff.", "Contended"], rows)


def bench_table(res: Path):
    d = load(res / "bench_bc_step.json")
    if not d:
        return "_No bench_bc_step.json yet._"
    rows = [[r["config"], f"{r['samples_per_s']:.0f}", f"{100*r['mfu']:.1f}%", f"{r['compute_s']:.2f}",
             f"{r['peak_mem_gb']:.1f}", "yes" if r.get("contended") else "no"] for r in d if "samples_per_s" in r]
    return md(["Configuration (cumulative)", "Samples/s", "MFU", "Step s", "Peak GB", "Contended"], rows)


def stall_table(res: Path):
    d = load(res / "ckpt_stall.json")
    if not d:
        return "_No ckpt_stall.json yet._"
    rows = [[k, v["saves"], f"{v['stall_s_median']:.2f} ({v['stall_s_min']:.2f}-{v['stall_s_max']:.2f})",
             f"{v['publish_s_median']:.1f}", f"{v['step_s_during_median']:.2f}"] for k, v in d["legs"].items()]
    t = md(["Mode_location", "Saves", "Loop stall s median (min-max)", "Time to published s", "Step s while writing"], rows)
    return t + f"\n\nAsync/sync stall ratio: shm {d['async_stall_over_sync_shm']:.2f}, NFS {d['async_stall_over_sync_nfs']:.2f}" \
        + flag(d.get("contended"))


# ------------------------------------------------------------------------------------------------ latency
def latency_table(res: Path):
    d = load(res / "latency.json")
    if not d:
        return "_No latency.json yet._"
    rows = []
    for r in d["rows"]:
        rows.append([r["row"], r["name"], r["batch"], f"{r['p50_ms_per_query_batch']:.1f}", f"{r['p95_ms_per_query_batch']:.1f}",
                     f"{r['p50_ms_per_query']:.2f}", f"{r['p50_ms_per_action']:.2f}",
                     f"{r['token_agreement_vs_row2']:.3f}" if r.get("token_agreement_vs_row2") is not None else "-",
                     f"{r['measured_over_floor']:.1f}x" if r.get("measured_over_floor") else "-",
                     f"{r['peak_mem_gb']:.1f}" if r.get("peak_mem_gb") else "-", "yes" if r.get("contended") else "no"])
    t = md(["Row", "Config", "Batch", "p50 ms/batch", "p95 ms/batch", "p50 ms/query", "ms/action", "Token agree vs row 2",
            "Measured / HBM floor", "Peak GB", "Contended"], rows)
    rf = next((r["roofline"] for r in d["rows"] if r["row"] == 3 and r["batch"] == 1 and "roofline" in r), None)
    rf2 = next((r["roofline"] for r in d["rows"] if r["row"] == 4 and r["batch"] == 1 and "roofline" in r), None)
    if rf and rf2:
        t += (f"\n\nRoofline (batch 1): arithmetic intensity {rf['arith_intensity_flop_per_byte']:.2f} FLOP/B vs A100 ridge "
              f"{rf['ridge_flop_per_byte']:.0f}: memory-bound. lm_head is {100*rf['head_share_of_weight_bytes']:.0f}% of weight "
              f"bytes per token before slicing, {100*rf2['head_share_of_weight_bytes']:.2f}% after.")
    return t


# ------------------------------------------------------------------------------------------------ ablations
def ablation_table(res: Path):
    d = load(res / "ablations.json")
    if not d:
        return "_No ablations.json yet._", None
    runs = d["runs"]
    names = list(dict.fromkeys(r["name"] for r in runs))
    rows = []
    for n in names:
        cells = []
        for sp in SPLITS:
            v = [r for r in runs if r["name"] == n and r["split"] == sp]
            cells.append(", ".join(f"{r['task']} {100*r['success']:.0f}" for r in v) or "-")
        rows.append([n, *cells])
    return md(["Run", "Train (task succ %)", "Held-out", "Held-out photo"], rows), runs


# ------------------------------------------------------------------------------------------------ plots
def plots(res: Path, runs_dir: Path, out: Path, data: dict, ablation_runs):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["(matplotlib not installed: plots skipped)"]
    out.mkdir(parents=True, exist_ok=True)
    made = []
    # 1) success per split, policies side by side
    if data:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        tasks = ["lift", "push"]
        names = [n for n in data if not re.search(r"_seed\d+$", n)] or list(data)
        for ax, task in zip(axes, tasks):
            w = 0.8 / max(len(names), 1)
            for i, n in enumerate(names):
                ys, lo, hi = [], [], []
                for sp in SPLITS:
                    r = data[n]["results"].get(sp, {}).get(task)
                    ys.append(100 * r["success"] if r else 0)
                    lo.append(100 * (r["success"] - r["ci95"][0]) if r else 0)
                    hi.append(100 * (r["ci95"][1] - r["success"]) if r else 0)
                ax.bar([j + i * w for j in range(3)], ys, w, yerr=[lo, hi], label=n, capsize=2)
            ax.set_xticks([j + 0.4 - w / 2 for j in range(3)]); ax.set_xticklabels(SPLITS); ax.set_title(task)
        axes[0].set_ylabel("success % (95% Wilson CI)"); axes[1].legend(fontsize=7)
        fig.tight_layout(); p = out / "success_by_split.png"; fig.savefig(p, dpi=130); made.append(str(p)); plt.close(fig)
    # 2) scaling
    sc = load(res / "scaling.json")
    if sc:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for key in sorted({(c["backend"], c["act_ckpt"]) for c in sc["configs"]}):
            pts = sorted((c["n_gpu"], c["samples_per_s_median"]) for c in sc["configs"] if (c["backend"], c["act_ckpt"]) == key)
            ax.plot(*zip(*pts), marker="o", label=f"{key[0]} ckpt={'on' if key[1] else 'off'}")
        ax.set_xlabel("GPUs"); ax.set_ylabel("samples/s"); ax.legend(fontsize=7); fig.tight_layout()
        p = out / "scaling.png"; fig.savefig(p, dpi=130); made.append(str(p)); plt.close(fig)
    # 3) latency
    lat = load(res / "latency.json")
    if lat:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for B in sorted({r["batch"] for r in lat["rows"]}):
            rows = [r for r in lat["rows"] if r["batch"] == B]
            ax.bar([f"{r['row']}" + ("\n(B=%d)" % B) for r in rows], [r["p50_ms_per_query"] for r in rows], label=f"batch {B}")
        ax.set_yscale("log"); ax.set_ylabel("p50 ms / query (amortised)"); ax.legend(fontsize=7); fig.tight_layout()
        p = out / "latency.png"; fig.savefig(p, dpi=130); made.append(str(p)); plt.close(fig)
    # 4) RL learning curves
    curves = {}
    for f in glob.glob(str(runs_dir / "*" / "metrics.jsonl")) + glob.glob(str(res / "*" / "metrics.jsonl")):
        rows = [json.loads(x) for x in open(f)]
        rows = [r for r in rows if "return_mean" in r]
        if rows:
            curves[Path(f).parent.name] = rows
    if curves:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for n, rows in curves.items():
            it = [r["iter"] for r in rows]
            axes[0].plot(it, [r["return_mean"] for r in rows], label=n)
            axes[1].plot(it, [r["success"] for r in rows])
            axes[2].plot(it, [r["kl"] for r in rows])
        for ax, t in zip(axes, ["mean return", "train success", "KL to BC ref"]):
            ax.set_title(t); ax.set_xlabel("iteration")
        axes[0].legend(fontsize=6); fig.tight_layout()
        p = out / "rl_curves.png"; fig.savefig(p, dpi=130); made.append(str(p)); plt.close(fig)
    if ablation_runs:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        names = list(dict.fromkeys(r["name"] for r in ablation_runs))
        for i, sp in enumerate(SPLITS):
            ys = []
            for n in names:
                v = [r["success"] for r in ablation_runs if r["name"] == n and r["split"] == sp]
                ys.append(100 * statistics.mean(v) if v else 0)
            ax.bar([j + i * 0.27 for j in range(len(names))], ys, 0.27, label=sp)
        ax.set_xticks([j + 0.27 for j in range(len(names))]); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("success % (mean over tasks)"); ax.legend(fontsize=7); fig.tight_layout()
        p = out / "ablations.png"; fig.savefig(p, dpi=130); made.append(str(p)); plt.close(fig)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs_pulled")
    ap.add_argument("--out", default="results/tables.md")
    ap.add_argument("--plots", default="results/plots")
    a = ap.parse_args()
    res = Path(a.results)
    ceil = load(res / "expert_ceiling.json")
    pol, data = policy_table(res)
    abl, abl_runs = ablation_table(res)
    sections = [("Policy results (closed-loop, fixed eval seeds; % of expert = success / expert ceiling)", pol)]
    if ceil:
        rows = [[sp, t, f"{100*r['success']:.1f} [{100*r['ci95'][0]:.0f},{100*r['ci95'][1]:.0f}]"]
                for sp, d in ceil["ceiling"].items() for t, r in d.items()]
        sections.append((f"Scripted-expert ceiling (n={ceil.get('episodes')}/task)", md(["Split", "Task", "Success %"], rows)))
    sections += [("Ablations", abl), ("Distributed training: scaling sweep", scaling_table(res)),
                 ("Training step micro-benchmark", bench_table(res)), ("Async checkpoint stall", stall_table(res)),
                 ("Inference latency", latency_table(res))]
    made = plots(res, Path(a.runs), Path(a.plots), data, abl_runs)
    text = "\n\n".join(f"### {t}\n\n{b}" for t, b in sections)
    text += "\n\n### Plots\n\n" + "\n".join(f"![{Path(p).stem}]({p})" if p.endswith(".png") else p for p in made) + "\n"
    text += ("\n_All timing numbers are measured on shared GPUs; rows marked contended ran while other users' jobs were active. "
             "See the contention protocol in the README._\n")
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
