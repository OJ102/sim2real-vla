"""P8 scripts/render_demo.py (commit 9cd35e8): demo mp4s, tasks x splits grid, index.json. CPU, tiny random-init model.

Videos are inspected with imageio_ffmpeg (bundled ffmpeg; there is no ffprobe): stream metadata from the first yield of
read_frames() and frame counts from count_frames_and_secs(). Frames are never all decoded into memory.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# RENDER_DEMO_SCRIPT: run a mutated copy instead (teeth checks); default = the real script
SCRIPT = Path(os.environ.get("RENDER_DEMO_SCRIPT", ROOT / "scripts/render_demo.py"))
HOLD = 15                                           # final frame held 1.5 s at 10 fps


def _run(*args, timeout=600):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def _ok(*args, timeout=600):
    r = _run(*args, timeout=timeout)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2500:]
    return r


def _meta(path):
    import imageio_ffmpeg
    gen = imageio_ffmpeg.read_frames(str(path))
    try:
        return next(gen)
    finally:
        gen.close()


def _nframes(path):
    import imageio_ffmpeg
    return imageio_ffmpeg.count_frames_and_secs(str(path))[0]


def _index(out):
    return json.loads((out / "index.json").read_text())


VLA = ["--model-config", "configs/model_tiny.yaml", "--head", "discrete", "--tasks", "reach,push",
       "--splits", "train,heldout_photo", "--episodes", "2", "--res", "192", "--grid-cell", "160", "--no-require-nvidia"]
SMALL = ["--model-config", "configs/model_tiny.yaml", "--tasks", "reach", "--splits", "train", "--episodes", "2",
         "--res", "192", "--grid-cell", "160", "--no-require-nvidia"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    _ok(*VLA, "--grid-episode", "1", "--out-root", str(root), "--name", "greedy")
    # grid-episode 1 on the expert too: its episodes differ in length (push/heldout 100 vs 31 steps), the random-init
    # policy's usually don't, so only here does the grid length actually pin down which episode the grid shows
    _ok("--policy", "expert", "--episodes", "2", "--res", "256", "--grid-cell", "192", "--grid-episode", "1",
        "--no-require-nvidia", "--out-root", str(root), "--name", "expert")
    for n in ("t1a", "t1b"):
        _ok(*SMALL, "--head", "discrete", "--temperature", "1", "--out-root", str(root), "--name", n)
    return root


# ---------------------------------------------------------------- 1) files + encoding
@pytest.mark.parametrize("name,tasks,splits,res,cell", [
    ("greedy", ["reach", "push"], ["train", "heldout_photo"], 192, 160),
    ("expert", ["reach", "lift", "push"], ["train", "heldout", "heldout_photo"], 256, 192)])
def test_files_and_encoding(runs, name, tasks, splits, res, cell):
    out = runs / name
    want = {f"{t}_{s}_s{seed}.mp4" for t in tasks for s in splits for seed in (1000, 1001)} | {"grid.mp4", "index.json"}
    assert {p.name for p in out.iterdir()} == want
    for t in tasks:
        for s in splits:
            for seed in (1000, 1001):
                m = _meta(out / f"{t}_{s}_s{seed}.mp4")
                assert (m["codec"], m["pix_fmt"].split("(")[0], m["fps"], tuple(m["size"])) == ("h264", "yuv420p", 10.0, (res, res)), m
    g = _meta(out / "grid.mp4")
    assert tuple(g["size"]) == (len(splits) * cell, len(tasks) * cell)          # (width, height) = (cols*cell, rows*cell)
    assert g["codec"] == "h264" and g["fps"] == 10.0


# ---------------------------------------------------------------- 2) index.json
@pytest.mark.parametrize("name", ["greedy", "expert", "t1a"])
def test_index_cells_consistent(runs, name):
    idx = _index(runs / name)
    for key, c in idx["cells"].items():
        eps = c["episodes"]
        assert c["n"] == len(eps) == idx["meta"]["episodes"] == 2
        assert c["success"] == pytest.approx(sum(e["success"] for e in eps) / len(eps))
        assert [e["seed"] for e in eps] == [idx["meta"]["seed0"], idx["meta"]["seed0"] + 1] == [1000, 1001]
        task, split = key.split("/")
        for e in eps:
            assert set(e) >= {"seed", "success", "steps", "instruction", "video"}
            assert e["video"] == f"{task}_{split}_s{e['seed']}.mp4" and (runs / name / e["video"]).exists()
            assert isinstance(e["success"], bool) and 1 <= e["steps"] <= 100
            assert isinstance(e["instruction"], str) and e["instruction"].strip()
    assert idx["meta"]["temperature"] == (1.0 if name == "t1a" else None)


# ---------------------------------------------------------------- 7) frame counts, grid length/episode
@pytest.mark.parametrize("name", ["greedy", "expert"])
def test_frames_per_episode_and_grid_length(runs, name):
    out, idx = runs / name, _index(runs / name)
    for c in idx["cells"].values():
        for e in c["episodes"]:
            assert _nframes(out / e["video"]) == e["steps"] + 1 + HOLD, e
    gi = idx["meta"]["grid_episode"]
    assert idx["grid"]["episode_index"] == gi == 1
    # grid = longest SHOWN episode (episode gi of each cell); the others hold their last frame
    assert _nframes(out / "grid.mp4") == max(c["episodes"][gi]["steps"] for c in idx["cells"].values()) + 1 + HOLD


# ---------------------------------------------------------------- 3) expert == rollout_eval.run_expert
def test_expert_matches_run_expert(runs):
    from vla.eval.rollout_eval import run_expert
    from vla.sim.vec_env import SubprocVecEnv
    idx = _index(runs / "expert")
    vec = SubprocVecEnv(2, require_nvidia=False)
    try:
        for key, c in idx["cells"].items():
            task, split = key.split("/")
            ref = run_expert(vec, task, split, [1000, 1001])
            assert [(e["success"], e["steps"]) for e in c["episodes"]] == [(r["success"], r["steps"]) for r in ref], key
    finally:
        vec.close()
    assert [(e["success"], e["steps"]) for e in idx["cells"]["push/heldout"]["episodes"]] == [(False, 100), (True, 31)]


# ---------------------------------------------------------------- 4) temperature reproducible, differs from greedy
def test_temperature_byte_identical_and_differs_from_greedy(runs):
    for seed in (1000, 1001):
        fn = f"reach_train_s{seed}.mp4"
        a, b = (runs / "t1a" / fn).read_bytes(), (runs / "t1b" / fn).read_bytes()
        assert a == b, fn
        assert a != (runs / "greedy" / fn).read_bytes(), fn          # same res / cell / seeds as the greedy run
    assert _index(runs / "t1a")["cells"] == _index(runs / "t1b")["cells"]


# ---------------------------------------------------------------- 5) flow + T: warning, greedy
def test_flow_with_temperature_warns(tmp_path):
    r = _ok("--model-config", "configs/model_tiny.yaml", "--head", "flow", "--tasks", "reach", "--splits", "train",
            "--episodes", "1", "--res", "64", "--grid-cell", "64", "--no-require-nvidia", "--temperature", "1",
            "--out-root", str(tmp_path), "--name", "flow")
    assert "--temperature ignored" in r.stderr
    assert _index(tmp_path / "flow")["meta"]["temperature"] is None


def test_expert_with_temperature_warns(tmp_path):
    r = _ok("--policy", "expert", "--tasks", "reach", "--splits", "train", "--episodes", "1", "--res", "64",
            "--grid-cell", "64", "--no-require-nvidia", "--temperature", "1", "--out-root", str(tmp_path), "--name", "ex")
    assert "--temperature ignored" in r.stderr
    assert _index(tmp_path / "ex")["meta"]["temperature"] is None


# ---------------------------------------------------------------- 6) argparse rejections
@pytest.mark.parametrize("args,msg", [
    (["--temperature", "0"], "--temperature must be finite and > 0"),
    (["--temperature=-1"], "--temperature must be finite and > 0"),
    (["--temperature", "nan"], "--temperature must be finite and > 0"),
    (["--res", "1000"], "--res must be in [16, 640]"),
    (["--res", "8"], "--res must be in [16, 640]"),
    (["--episodes", "2", "--grid-episode", "2"], "--grid-episode must be < --episodes"),
    (["--episodes", "2", "--grid-episode", "-1"], "--grid-episode must be < --episodes"),
    (["--tasks", "reach,stack"], "unknown task 'stack'"),
    (["--splits", "train,test"], "unknown split 'test'"),
])
def test_rejected_args_exit_2(tmp_path, args, msg):
    r = _run("--policy", "expert", "--no-require-nvidia", "--out-root", str(tmp_path), *args, timeout=120)
    assert r.returncode == 2, (r.returncode, r.stderr[-800:])
    assert msg in r.stderr, r.stderr[-800:]
    assert not any(tmp_path.iterdir())                                # nothing written
