"""Amendment A (model side): rows with swap=1 are excluded from the main (action) loss but keep the aux ground loss."""
from pathlib import Path

import pytest
import torch

from vla.model.prompts import encode_instructions
from vla.model.vla import VLA, VLAConfig
from vla.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def make(head, seed=0):
    torch.manual_seed(seed)
    gw = 0.2 if head == "discrete" else 0.01
    m = VLA(load_config(VLAConfig, str(ROOT / "configs/model_tiny.yaml"), [f"head={head}", f"ground_weight={gw}"])).eval()
    with torch.no_grad():     # untrained action rows are near-identical -> CE is constant; randomise so the loss depends on targets
        w = m.llm.get_input_embeddings().weight
        w[m.act_base:] = torch.randn(256, w.shape[1], generator=torch.Generator().manual_seed(9)) * 0.5
        if head == "flow":
            m.dit.out.weight.normal_(0, 0.2)
    return m


def batch(m, swap, B=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {"image": torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8, generator=g),
            "proprio": torch.randn(B, 11, generator=g), "actions": torch.rand(B, 4, 4, generator=g) * 2 - 1,
            "ground_xy": torch.randn(B, 2, generator=g), "ground_mask": torch.ones(B),
            "swap": torch.tensor(swap, dtype=torch.float),
            **encode_instructions(m.tokenizer, ["lift the red cube", "lift the blue cube", "reach the green cube", "lift the red cube"][:B])}


def fwd(m, b, seed=3):
    torch.manual_seed(seed)                       # fixed flow t/noise
    with torch.no_grad():
        return m(b)


@pytest.mark.parametrize("head", ["discrete", "flow"])
def test_all_swapped_batch_main_zero_no_nan_ground_finite(head):
    m = make(head)
    o = fwd(m, batch(m, [1, 1, 1, 1]))
    assert o["main_loss"].item() == 0 and torch.isfinite(o["main_loss"]), o["main_loss"]
    assert torch.isfinite(o["ground_loss"]) and o["ground_loss"] > 0 and torch.isfinite(o["loss"])
    assert o["loss"].item() == pytest.approx(m.cfg.ground_weight * o["ground_loss"].item(), rel=1e-5)


@pytest.mark.parametrize("head", ["discrete", "flow"])
def test_main_loss_ignores_swapped_rows(head):
    m = make(head)
    b = batch(m, [0, 1, 0, 1])
    base = fwd(m, b)
    pert = dict(b, actions=b["actions"].clone()); pert["actions"][[1, 3]] = 1.0 - pert["actions"][[1, 3]]    # change swapped rows' actions
    o = fwd(m, pert)
    assert o["main_loss"].item() == pytest.approx(base["main_loss"].item(), rel=1e-6, abs=1e-7)
    assert o["ground_loss"].item() == pytest.approx(base["ground_loss"].item(), rel=1e-6)
    # but perturbing a non-swapped row's actions does change it
    pert2 = dict(b, actions=b["actions"].clone()); pert2["actions"][0] = 1.0 - pert2["actions"][0]
    assert fwd(m, pert2)["main_loss"].item() != pytest.approx(base["main_loss"].item(), rel=1e-4)
    # main loss over {0,2} equals the main loss of the same two rows run alone (averaged over non-swapped rows only)
    if head == "discrete":
        two = {k: (v[[0, 2]] if torch.is_tensor(v) else v) for k, v in b.items()}
        two["swap"] = torch.zeros(2)
        assert fwd(m, two)["main_loss"].item() == pytest.approx(base["main_loss"].item(), rel=1e-4)


@pytest.mark.parametrize("head", ["discrete", "flow"])
def test_swap_zero_equals_no_swap_key(head):
    m = make(head)
    b = batch(m, [0, 0, 0, 0]); b2 = {k: v for k, v in b.items() if k != "swap"}
    a, c = fwd(m, b), fwd(m, b2)
    assert a["loss"].item() == pytest.approx(c["loss"].item(), rel=1e-5) and a["main_loss"].item() == pytest.approx(c["main_loss"].item(), rel=1e-5)


@pytest.mark.parametrize("head", ["discrete", "flow"])
def test_swapped_rows_still_train_the_ground_head(head):
    m = make(head).train()
    b = batch(m, [1, 1, 1, 1])
    m(b)["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.ground_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith("llm.model"))
