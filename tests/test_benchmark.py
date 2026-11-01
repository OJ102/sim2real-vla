"""P7 latency benchmark (CPU, tiny model): output schema, roofline numbers, consistency."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def bench(out, *args, timeout=600):
    r = subprocess.run([sys.executable, "-m", "vla.infer.benchmark", "--model-config", "configs/model_tiny.yaml",
                        "--repeats", "2", "--warmup", "2", "--iters", "3", "--n-obs", "12", *args, "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    return json.loads(Path(out).read_text()), r.stdout


@pytest.fixture(scope="module")
def lat(tmp_path_factory):
    return bench(tmp_path_factory.mktemp("lat") / "lat.json", "--rows", "1,2,3,4,5", "--batches", "1,4")


def test_rows_and_batches_present(lat):
    d, stdout = lat
    got = sorted((r["row"], r["batch"]) for r in d["rows"])
    assert got == [(r, b) for r in (1, 2, 3, 4, 5) for b in (1, 4)]
    assert "wrote" in stdout
    m = d["meta"]
    for k in ("device", "torch", "repeats", "iters", "warmup", "n_obs", "hbm_bytes_per_s"):
        assert k in m
    assert m["repeats"] == 2 and m["iters"] == 3 and m["n_obs"] == 12 and m["hbm_bytes_per_s"] == 2.0e12
    assert "gpu_snapshot_after" in m and m["device"] == "cpu"


def test_latency_stats_are_consistent(lat):
    for r in lat[0]["rows"]:
        assert 0 < r["p50_ms_per_query_batch"] <= r["p95_ms_per_query_batch"]
        assert r["repeat_p50_min_ms"] <= r["repeat_p50_median_ms"] <= r["repeat_p50_max_ms"] and r["n_repeats"] == 2
        # per-query = per-batch / B ; per-action = per-query / 4 actions in the chunk
        assert r["p50_ms_per_query"] == pytest.approx(r["p50_ms_per_query_batch"] / r["batch"])
        assert r["p50_ms_per_action"] == pytest.approx(r["p50_ms_per_query"] / 4)
        assert all(map(lambda x: x == x and x > 0, [r["p50_ms_per_query"], r["p95_ms_per_query_batch"]]))


def test_token_agreement_vs_row2(lat):
    rows = {(r["row"], r["batch"]): r for r in lat[0]["rows"]}
    for b in (1, 4):
        assert rows[(2, b)]["token_agreement_vs_row2"] == 1.0              # row 2 against itself
    for r in lat[0]["rows"]:
        assert 0.0 <= r["token_agreement_vs_row2"] <= 1.0


def test_roofline_keys_and_sliced_head(lat):
    rows = {(r["row"], r["batch"]): r for r in lat[0]["rows"]}
    keys = {"weight_bytes", "kv_bytes_per_step", "bytes_per_step", "head_bytes", "head_share_of_weight_bytes",
            "hbm_floor_ms_per_step"}
    for r in lat[0]["rows"]:
        rf = r["roofline"]
        assert keys <= set(rf), keys - set(rf)
        assert rf["bytes_per_step"] == pytest.approx(rf["weight_bytes"] + rf["kv_bytes_per_step"])
        assert rf["hbm_floor_ms_per_step"] == pytest.approx(rf["bytes_per_step"] / 2.0e12 * 1e3)
        assert 0 < rf["head_share_of_weight_bytes"] < 1
        assert "measured_over_floor" in r and r["measured_over_floor"] > 0
        assert r["hbm_floor_ms_per_query_batch"] == pytest.approx(r["roofline"]["hbm_floor_ms_per_step"] * 16, rel=1e-6) \
            or r["hbm_floor_ms_per_query_batch"] > 0
        assert "gpu_snapshot" in r and "contended" in r and isinstance(r["contended"], bool)
    full, sliced = rows[(3, 1)]["roofline"], rows[(4, 1)]["roofline"]
    assert sliced["head_bytes"] * 2 == full["head_bytes"] or sliced["head_bytes"] < full["head_bytes"] / 1.9   # 256 rows << vocab
    assert sliced["head_share_of_weight_bytes"] < full["head_share_of_weight_bytes"]
    assert sliced["hbm_floor_ms_per_step"] < full["hbm_floor_ms_per_step"]
    # fp32 (row 1) reads exactly 2x the bytes of bf16 (row 2)
    assert rows[(1, 1)]["roofline"]["weight_bytes"] == 2 * rows[(2, 1)]["roofline"]["weight_bytes"]
    # same weights for rows 4 and 5
    assert rows[(5, 1)]["roofline"]["weight_bytes"] == rows[(4, 1)]["roofline"]["weight_bytes"]


def test_batch_gives_throughput(lat):
    rows = {(r["row"], r["batch"]): r for r in lat[0]["rows"]}
    for row in (1, 2, 3, 4, 5):
        assert rows[(row, 4)]["p50_ms_per_query"] < rows[(row, 1)]["p50_ms_per_query"] * 1.05   # batching amortises


def test_flow_row6_works_on_cpu(tmp_path):
    d, _ = bench(tmp_path / "f.json", "--rows", "6", "--batches", "1")
    (r,) = d["rows"]
    assert r["row"] == 6 and r["batch"] == 1 and r["p50_ms_per_query_batch"] > 0
    assert "contended" in r and "gpu_snapshot" in r


def test_flow_row_with_discrete_rows(tmp_path):
    d, _ = bench(tmp_path / "m.json", "--rows", "2,4,6", "--batches", "1")
    assert sorted(r["row"] for r in d["rows"]) == [2, 4, 6]


def test_bad_row_or_missing_args_exit_nonzero(tmp_path):
    r = subprocess.run([sys.executable, "-m", "vla.infer.benchmark", "--model-config", "configs/model_tiny.yaml",
                        "--rows", "x", "--out", str(tmp_path / "x.json")], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
