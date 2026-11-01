"""scripts/probe_grounding.py (plan step 7): runs on CPU with a tiny ckpt and 2 shards; JSON schema."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK_KEYS = {"cos_expert", "cos_geo", "swap_cos_new", "swap_cos_orig", "swap_delta"}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("probe")
    demos = d / "demos"
    r = subprocess.run([sys.executable, "-m", "vla.data.generate", "--config", "configs/data.yaml", "--num-workers", "2",
                        "--no-require-nvidia", f"out_dir={demos}", "episodes_per_combo=2", "chunk_size=2", "tasks=[lift]",
                        "colors=[red,blue]"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    from vla.model.vla import VLA, VLAConfig
    from vla.utils.config import load_config
    torch.manual_seed(0)
    ck = {}
    for name, gw in (("plain", 0.0), ("ground", 0.2)):
        m = VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), [f"ground_weight={gw}"]))
        torch.save(m.state_dict(), d / f"{name}.pt"); ck[name] = d / f"{name}.pt"
    return d, demos, ck


def probe(d, demos, ck, tag, *extra):
    out = d / f"{tag}.json"
    r = subprocess.run([sys.executable, "scripts/probe_grounding.py", "--ckpt", str(ck), "--data", str(demos), "--out", str(out),
                        "--n-train", "2", "--device", "cpu", "--model-config", "configs/model_tiny.yaml", *extra],
                       cwd=ROOT, capture_output=True, text=True, timeout=420)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    return json.loads(out.read_text())


def _find(d, key):
    """Return every dict in the (arbitrarily nested) JSON that holds `key`."""
    out = []
    if isinstance(d, dict):
        if key in d:
            out.append(d)
        for v in d.values():
            out += _find(v, key)
    return out


def test_probe_json_schema_plain_and_ground(setup):
    d, demos, ck = setup
    for tag, gw in (("plain", False), ("ground", True)):
        res = probe(d, demos, ck[tag], tag)
        cells = _find(res, "cos_expert")
        assert cells, res
        for c in cells:
            assert TASK_KEYS <= set(c), TASK_KEYS - set(c)
            for k in TASK_KEYS:
                assert isinstance(c[k], (int, float)) and c[k] == c[k]
            assert -1.0001 <= c["cos_expert"] <= 1.0001 and c["swap_delta"] >= 0
        g = _find(res, "ground_err_cm_t0")
        assert bool(g) == gw, (tag, g)                                   # only models with a ground head report it
        txt = json.dumps(res)
        assert "train" in txt and "val" in txt and "lift" in txt
