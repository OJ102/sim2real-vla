"""Static-KV-cache / CUDA-graph decode engine tests. CPU (tiny model) + GPU-only tests skipped without CUDA."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vla.infer.engine import InferenceEngine
from vla.model.prompts import encode_instructions
from vla.model.vla import VLA, VLAConfig
from vla.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
INSTR = ["lift the red cube", "push the blue cube to the yellow zone", "reach the green cube"]


def tiny(seed=0):
    torch.manual_seed(seed)
    m = VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), [])).eval()
    with torch.no_grad():   # new action rows start near-identical (mean init) -> randomise to get informative logits
        w = m.llm.get_input_embeddings().weight
        w[m.act_base:] = torch.randn(256, w.shape[1], generator=torch.Generator().manual_seed(7)) * 0.5
    return m


def batch(m, instr, seed=1):
    g = torch.Generator().manual_seed(seed)
    b = len(instr)
    return {"image": torch.randint(0, 256, (b, 64, 64, 3), dtype=torch.uint8, generator=g),
            "proprio": torch.randn(b, 11, generator=g), **encode_instructions(m.tokenizer, list(instr))}


def n_prefix(m):
    return (m.cfg.image_size // 16) ** 2 + 1


def same(a, b, atol=1e-5):
    assert torch.equal(a["bins"], b["bins"]), (a["bins"] != b["bins"]).sum()
    assert torch.allclose(a["logp"], b["logp"], atol=atol), (a["logp"] - b["logp"]).abs().max()
    assert torch.allclose(a["actions"], b["actions"])


# ------------------------------------------------------------------ 1) static == eager
def test_static_greedy_equals_eager_and_repeatable():
    m = tiny(); b = batch(m, INSTR)
    eager = InferenceEngine(m, 1.0, "eager").generate(b, greedy=True)
    st = InferenceEngine(m, 1.0, "static")
    r1, r2 = st.generate(b, greedy=True), st.generate(b, greedy=True)
    same(eager, r1); same(r1, r2)
    assert torch.equal(r1["bins"], r2["bins"]) and torch.equal(r1["logp"], r2["logp"])   # bit-repeatable


@pytest.mark.parametrize("max_prefix_extra", [None, 120])
def test_static_varying_prompt_lengths(max_prefix_extra):
    m = tiny()
    mp = None if max_prefix_extra is None else n_prefix(m) + max_prefix_extra
    st = InferenceEngine(m, 1.0, "static", max_prefix=mp); ea = InferenceEngine(m, 1.0, "eager")
    seqs = [["lift the red cube"] * 2, ["push the blue cube to the yellow zone, slowly and carefully please"] * 2,
            ["reach it"] * 2, ["push the blue cube to the yellow zone"] * 2]   # shorter, longer, shortest, mid
    for i, ins in enumerate(seqs):
        b = batch(m, ins, seed=10 + i)
        same(ea.generate(b, greedy=True), st.generate(b, greedy=True))
    # mixed lengths *within* a batch (left padding)
    b = batch(m, ["reach it", "push the blue cube to the yellow zone"], seed=99)
    same(ea.generate(b, greedy=True), st.generate(b, greedy=True))


def test_static_prompt_longer_than_max_prefix_fails_loudly_not_silently():
    """max_prefix is a hard limit once a decoder exists: a longer prompt must not silently give wrong actions."""
    m = tiny(); st = InferenceEngine(m, 1.0, "static", max_prefix=n_prefix(m) + 30)
    st.generate(batch(m, ["reach it"] * 2), greedy=True)
    long_b = batch(m, ["push the blue cube to the yellow zone, slowly and carefully please"] * 2)
    try:
        out = st.generate(long_b, greedy=True)
    except Exception:  # noqa: BLE001  (currently an obscure RuntimeError on the mask copy; reported to builder)
        return
    same(InferenceEngine(m, 1.0, "eager").generate(long_b, greedy=True), out)      # or it must be correct


def test_static_batch_size_change_allocates_new_decoder():
    m = tiny(); st = InferenceEngine(m, 1.0, "static"); ea = InferenceEngine(m, 1.0, "eager")
    for ins in (INSTR, INSTR[:1], INSTR[:2], INSTR):
        b = batch(m, ins, seed=3)
        same(ea.generate(b, greedy=True), st.generate(b, greedy=True))


def test_static_sampling_same_seed_identical_to_eager():
    m = tiny(); b = batch(m, INSTR)
    for T in (1.0, 0.7):
        a = InferenceEngine(m, T, "eager").generate(b, generator=torch.Generator().manual_seed(11))
        s = InferenceEngine(m, T, "static").generate(b, generator=torch.Generator().manual_seed(11))
        same(a, s)
    s1 = InferenceEngine(m, 1.0, "static").generate(b, generator=torch.Generator().manual_seed(12))
    assert not torch.equal(s1["bins"], a["bins"]) and len(set(s1["bins"].flatten().tolist())) > 1


def test_static_logp_matches_teacher_forced():
    m = tiny(); b = batch(m, INSTR)
    out = InferenceEngine(m, 1.0, "static").generate(b, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        lp = m.token_logprobs(b, out["bins"])
    assert torch.allclose(out["logp"], lp, atol=1e-4)


def test_static_sees_in_place_weight_updates():
    m = tiny(); b = batch(m, INSTR); st = InferenceEngine(m, 1.0, "static"); ea = InferenceEngine(m, 1.0, "eager")
    before = st.generate(b, greedy=True)["logp"].clone()
    ptrs = [p.data_ptr() for p in m.parameters()]
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    assert [p.data_ptr() for p in m.parameters()] == ptrs
    after = st.generate(b, greedy=True)
    same(ea.generate(b, greedy=True), after)
    assert not torch.allclose(before, after["logp"])
    v = st.verify_matches_eager(b)
    assert v["token_agreement"] == 1.0 and v["max_logp_diff"] < 1e-5


def test_engine_mode_validation_and_flow_rejected():
    m = tiny()
    with pytest.raises(AssertionError):
        InferenceEngine(m, 1.0, "bogus")
    torch.manual_seed(0)
    mf = VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), ["head=flow"]))
    with pytest.raises(AssertionError):
        InferenceEngine(mf, 1.0, "static")


# ------------------------------------------------------------------ 2) graph mode on CPU falls back
def test_graph_mode_cpu_falls_back_and_equals_eager():
    if torch.cuda.is_available():
        pytest.skip("CPU fallback test")
    m = tiny(); b = batch(m, INSTR)
    g = InferenceEngine(m, 1.0, "graph")
    out = g.generate(b, greedy=True)
    assert g.graph_status == {3: "cpu: no CUDA graphs"}
    same(InferenceEngine(m, 1.0, "eager").generate(b, greedy=True), out)
    same(out, g.generate(b, greedy=True))                              # still fine on the second call
    v = g.verify_matches_eager(b)
    assert v["token_agreement"] == 1.0 and v["graph_status"] == {3: "cpu: no CUDA graphs"}


# ------------------------------------------------------------------ 3) static mode inside the RL loop
def _grpo(tmp, name, *extra):
    env = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo", "TOKENIZERS_PARALLELISM": "false"}
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
                        "--master_addr=127.0.0.1", f"--master_port={port}", "-m", "vla.rl.grpo", "--config",
                        "configs/rl_tiny.yaml", f"runs_dir={tmp / 'runs'}", f"shm_dir={tmp / 'shm'}", f"name={name}",
                        "iterations=2", *extra], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2500:]
    return [json.loads(l) for l in (tmp / "runs" / name / "metrics.jsonl").read_text().splitlines()]


def test_rl_loop_static_equals_eager(tmp_path):
    e = _grpo(tmp_path, "rl_eager", "engine_mode=eager")
    s = _grpo(tmp_path, "rl_static", "engine_mode=static")
    assert s[0]["return_mean"] == pytest.approx(e[0]["return_mean"], abs=1e-4)
    assert s[0]["success"] == e[0]["success"]
    for x in s:
        assert x["engine_mismatch_equal_version"] < 1e-4 and x["tis_w_mean"] == pytest.approx(1.0, abs=1e-3)
    assert s[0]["R_success"] == pytest.approx(e[0]["R_success"], abs=1e-6)


# ------------------------------------------------------------------ GPU only
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@needs_cuda
def test_gpu_graph_capture_and_weight_copy():
    m = tiny().cuda().to(torch.bfloat16); b = {k: v.cuda() for k, v in batch(m, INSTR).items()}
    g = InferenceEngine(m, 1.0, "graph")
    out = g.generate(b, greedy=True)
    status = list(g.graph_status.values())[0]
    assert status == "captured" or status.startswith("capture failed:"), status
    ptrs = [p.data_ptr() for p in m.parameters()]
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.01 * torch.randn_like(p))                         # in-place, like weight_sync's copy_
    assert [p.data_ptr() for p in m.parameters()] == ptrs
    v = g.verify_matches_eager(b)
    assert v["token_agreement"] == 1.0 and v["max_logp_diff"] < 1e-2, v
    assert out["bins"].shape == (3, 16)


@needs_cuda
def test_gpu_graph_replays_equal_static_on_new_inputs():
    m = tiny().cuda().to(torch.bfloat16)
    g, s = InferenceEngine(m, 1.0, "graph"), InferenceEngine(m, 1.0, "static")
    for seed in (1, 2, 3):
        b = {k: v.cuda() for k, v in batch(m, INSTR, seed=seed).items()}
        a, c = g.generate(b, greedy=True), s.generate(b, greedy=True)
        assert (a["bins"] == c["bins"]).float().mean() > 0.99 and (a["logp"] - c["logp"]).abs().max() < 1e-2
