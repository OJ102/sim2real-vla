"""scripts/make_tables.py: synthetic results dir -> markdown + PNGs; missing/malformed inputs never crash."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def cell(s, lo=None, hi=None, vc=None):
    d = {"success": s, "ci95": [lo if lo is not None else max(0, s - .1), hi if hi is not None else min(1, s + .1)],
         "ee_jerk_rms": 12.3, "joint_limit_viol_per_ep": 0.1, "vel_viol_per_ep": 0.2, "collision_viol_per_ep": 0.3}
    if vc is not None:
        d["success_vs_ceiling"] = vc
    return d


def ev(s_tr, contended=False):
    return {"meta": {"contended": contended},
            "results": {sp: {t: cell(s_tr - i * .1, vc=(s_tr - i * .1) / 0.95) for t in ("lift", "push")}
                        for i, sp in enumerate(["train", "heldout", "heldout_photo"])}}


def build(res: Path, skip=()):
    res.mkdir(parents=True, exist_ok=True)
    files = {
        "eval_expert.json": ev(0.95),
        "eval_bc_disc.json": ev(0.6, contended=True),
        "eval_rl_seed0.json": ev(0.5), "eval_rl_seed1.json": ev(0.7),
        "expert_ceiling.json": {"episodes": 100, "ceiling": {"train": {"lift": {"success": 1.0, "ci95": [.96, 1.0]}}}},
        "scaling.json": {"configs": [{"backend": "fsdp", "n_gpu": n, "act_ckpt": False, "samples_per_s_median": 100 * n,
                                      "samples_per_s_min": 90 * n, "samples_per_s_max": 110 * n, "mfu_median": .3,
                                      "peak_mem_gb_max_rank": 10.0, "scaling_efficiency": 1.0, "any_contended": False}
                                     for n in (1, 2, 4)]},
        "latency.json": {"meta": {}, "rows": [
            {"row": r, "name": f"row{r}", "batch": 1, "p50_ms_per_query_batch": 10.0 / r, "p95_ms_per_query_batch": 12.0 / r,
             "p50_ms_per_query": 10.0 / r, "p50_ms_per_action": 2.5 / r, "token_agreement_vs_row2": 1.0,
             "measured_over_floor": 5.0, "peak_mem_gb": 1.0, "contended": False,
             "roofline": {"arith_intensity_flop_per_byte": 0.5, "ridge_flop_per_byte": 156.0,
                          "head_share_of_weight_bytes": 0.25 if r == 3 else 0.0003}} for r in (2, 3, 4)]},
        "ckpt_stall.json": {"legs": {f"{m}_{l}": {"saves": 5, "stall_s_median": 1.0, "stall_s_min": .5, "stall_s_max": 2.0,
                                                 "publish_s_median": 3.0, "step_s_during_median": .4}
                                     for m in ("sync", "async") for l in ("shm", "nfs")},
                            "async_stall_over_sync_shm": .1, "async_stall_over_sync_nfs": .2, "contended": False},
        "ablations.json": {"runs": [{"name": n, "split": sp, "task": "lift", "success": .5, "ci95": [.4, .6]}
                                    for n in ("full", "nocam") for sp in ("train", "heldout")]},
    }
    for k, v in files.items():
        if k not in skip:
            (res / k).write_text(json.dumps(v))
    runs = res / "runs"
    for n in ("rl_a", "rl_b"):
        (runs / n).mkdir(parents=True, exist_ok=True)
        (runs / n / "metrics.jsonl").write_text("\n".join(json.dumps({"iter": i, "return_mean": i * 1.0, "success": .1 * i,
                                                                      "kl": .01 * i}) for i in range(4)))


def run(res, runs=None, ok=True):
    out, plots = res / "t.md", res / "plots"
    r = subprocess.run([sys.executable, "scripts/make_tables.py", "--results", str(res), "--runs", str(runs or res / "runs"),
                        "--out", str(out), "--plots", str(plots)], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert (r.returncode == 0) == ok, r.stdout[-800:] + r.stderr[-1500:]
    return out.read_text() if ok else None, plots


def test_empty_results_dir(tmp_path):
    (tmp_path / "res").mkdir()
    text, plots = run(tmp_path / "res", tmp_path / "noruns")
    assert text.count("_No ") >= 6 and "eval_*.json yet" in text and "latency.json yet" in text
    assert not list(plots.glob("*.png")) if plots.exists() else True


def test_full_tables_and_plots(tmp_path):
    build(tmp_path)
    text, plots = run(tmp_path)
    assert text.count("_No ") == 1 and "_No bench_bc_step.json yet._" in text   # only the micro-benchmark is not synthesised
    # one row per (policy, task); contended flag only on the contended run
    for pol in ("expert", "bc_disc", "rl_seed0", "rl_seed1"):
        assert sum(1 for l in text.splitlines() if l.startswith(f"| {pol}") and ("| lift |" in l or "| push |" in l)) == 2, pol
    assert "bc_disc (contended)" in text and "expert (contended)" not in text
    assert "% of exp.)" in text and "100% of exp." in text                  # 0.95/0.95 -> 100
    # multi-seed aggregate: success 0.5 and 0.7 -> 60.0 ± 10.0, n=2 seeds
    agg = [l for l in text.splitlines() if l.startswith("| **rl (mean±std)**") and "| lift |" in l]
    assert len(agg) == 1 and "60.0 ± 10.0 (n=2 seeds)" in agg[0] and "50.0 ± 10.0" in agg[0]   # heldout 0.4/0.6 -> 50 ± 10
    assert "Scripted-expert ceiling (n=100/task)" in text and "Async/sync stall ratio: shm 0.10, NFS 0.20" in text
    assert "Roofline (batch 1)" in text and "25% of weight" in text and "0.03% after" in text
    for p in ("success_by_split", "scaling", "latency", "rl_curves", "ablations"):
        f = plots / f"{p}.png"
        assert f.is_file() and f.stat().st_size > 1000, p


@pytest.mark.parametrize("missing", ["eval_expert.json", "eval_bc_disc.json", "expert_ceiling.json", "scaling.json",
                                     "latency.json", "ckpt_stall.json", "ablations.json"])
def test_each_file_missing_individually(tmp_path, missing):
    build(tmp_path, skip=(missing,))
    text, _ = run(tmp_path)
    if missing in ("scaling.json", "latency.json", "ckpt_stall.json", "ablations.json"):
        assert f"_No {missing} yet._" in text
    else:
        assert "Policy results" in text


def test_roofline_sentence_needs_rows_3_and_4(tmp_path):
    build(tmp_path)
    d = json.loads((tmp_path / "latency.json").read_text())
    d["rows"] = [r for r in d["rows"] if r["row"] != 4]
    (tmp_path / "latency.json").write_text(json.dumps(d))
    text, _ = run(tmp_path)
    assert "Roofline (batch 1)" not in text and "| row3 |" in text


def test_malformed_json_skipped_not_crash(tmp_path):
    build(tmp_path)
    for f in ("eval_bc_disc.json", "scaling.json", "latency.json", "expert_ceiling.json", "ablations.json", "ckpt_stall.json"):
        (tmp_path / f).write_text("{ not json")
    (tmp_path / "runs" / "rl_a" / "metrics.jsonl").write_text("")
    text, plots = run(tmp_path)
    assert "bc_disc" not in text and "_No scaling.json yet._" in text and (plots / "success_by_split.png").is_file()


def test_launch_sh_single_string_and_argv_forms(tmp_path):
    import os, shutil, time, uuid
    if shutil.which("tmux") is None:
        pytest.skip("no tmux")
    env = {**os.environ, "VLA_RUNS": str(tmp_path / "runs"), "VLA_LOGS": str(tmp_path / "logs")}
    for args, code, needle in ((["export FOO=bar; echo foo=$FOO; exit 5"], 5, "foo=bar"),
                               (["bash", "-c", "echo hi; exit 6"], 6, "hi")):
        name = "t" + uuid.uuid4().hex[:8]
        r = subprocess.run(["bash", "scripts/remote/launch.sh", name, *args], cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        a, b = tmp_path / "runs" / name / "stdout.log", tmp_path / "logs" / f"{name}.log"
        t0 = time.time()
        while time.time() - t0 < 30 and not (a.exists() and b.exists() and "EXIT=" in a.read_text() and "EXIT=" in b.read_text()):
            time.sleep(0.3)
        try:
            assert a.read_text() == b.read_text() and needle in a.read_text()
            assert a.read_text().strip().splitlines()[-1].startswith(f"EXIT={code} ")
        finally:
            subprocess.run(["tmux", "kill-session", "-t", name], capture_output=True)
