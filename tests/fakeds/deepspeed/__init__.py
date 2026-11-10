"""CPU stand-in for DeepSpeed ZeRO-3, only for the bc.py backend=deepspeed code path: the returned optimizer is NOT a
torch.optim.Optimizer (like DeepSpeedZeroOptimizer_Stage3), and the engine steps the inner optimizer + lr_scheduler at
the gradient-accumulation boundary, like the real engine.
FAKEDS_PARTITION=1: params have numel()==0 outside forward/backward (gathered at forward, re-partitioned after each
optimizer step), like ZeRO-3, so code that counts params after initialize sees ~0."""
import os

import torch

LOG = []


class FakeZeroOptimizer:  # deliberately not a torch.optim.Optimizer subclass
    def __init__(self, opt):
        self.optimizer = opt

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class FakeEngine(torch.nn.Module):
    def __init__(self, model, opt, cfg, sched):
        super().__init__()
        self.module, self.optimizer, self.lr_scheduler = model, FakeZeroOptimizer(opt), sched
        self.accum, self.micro, self.clip = cfg["gradient_accumulation_steps"], 0, cfg["gradient_clipping"]
        self._gn = None
        self._partition = os.environ.get("FAKEDS_PARTITION") == "1"
        self._full = {}
        self._scatter()

    def _scatter(self):
        if self._partition:
            for p in self.module.parameters():
                if p.numel():
                    self._full[p] = p.data
                    p.data = torch.empty(0, dtype=p.dtype, device=p.device)

    def _gather(self):
        for p, d in self._full.items():
            p.data = d
        self._full = {}

    def forward(self, *a, **k):
        self._gather()
        return self.module(*a, **k)

    def backward(self, loss):
        (loss / self.accum).backward()

    def step(self):
        self.micro += 1
        if self.micro % self.accum:
            return
        self._gn = float(torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.clip))
        LOG.append([g["lr"] for g in self.optimizer.optimizer.param_groups])
        print("FAKEDS lr " + " ".join(f"{x:.6g}" for x in LOG[-1]), flush=True)
        self.optimizer.optimizer.step()
        self.optimizer.optimizer.zero_grad(set_to_none=True)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self._scatter()

    def get_global_grad_norm(self):
        return self._gn


def initialize(model=None, optimizer=None, config=None, lr_scheduler=None, **kw):
    eng = FakeEngine(model, optimizer, config, lr_scheduler)
    return eng, eng.optimizer, None, lr_scheduler
