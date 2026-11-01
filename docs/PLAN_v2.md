# PLAN v2: delta on top of PLAN.md (v1)

Author: Opus planner, 2026-11-01. Status: APPROVED by user 2026-11-01 (lead change: quiet-wait timeout 20 min, not 2 h).

Phase 0 and Phase 1 are done. The sim is not redesigned. Everything not mentioned here stays as in v1.

## Ranked accept/defer
Ranked by value per GPU-hour.

| # | Upgrade | Hedra relevance | Cost (GPU-h) | Risk | Verdict |
|---|---|---|---|---|---|
| 1 | **A. Decoupled actor-learner RL** (one-step off-policy, NCCL weight sync, truncated importance sampling) | Very high: verl-style RL infrastructure, inference and training together | 3 for the throughput study (RL runs count in item 4) | High: NCCL deadlocks, swapping weights under CUDA graphs | **Accept, core.** Replaces RL-on-DDP. |
| 2 | **G1. Slice lm_head to the 256 action rows + roofline decode analysis** | High: inference optimization from first principles | ~0 | Low | **Accept** |
| 3 | **C. Perf rigor**: MFU, 1/2/4-GPU scaling, activation checkpointing, prefetch overlap, profiler traces, CUDA graphs | High | 3 | Medium: shared GPUs add noise | **Accept.** Triton kernel deferred. |
| 4 | **D. Rigor**: 3 RL seeds, DR-group ablation, RL penalty ablation, data-scaling curve, success relative to expert ceiling | High: research credibility | 21 | Medium: budget | **Accept, trimmed** |
| 5 | **B1. Flow-matching action expert, BC head-to-head with discrete tokens** | Very high: flow models | 5 | Medium | **Accept, core** |
| 6 | **E. Async DCP checkpointing, stall measured sync vs async × NFS vs /dev/shm** | Medium-high: training infrastructure | 0.5 | Low | **Accept** |
| 7 | **F1/F3. Held-out real-photo textures (DTD) + true held-out light direction and kp** | Medium: a stronger, honest OOD split | 0 (CPU) | Low | **Accept.** F3 is mandatory for honesty. |
| 8 | **B2. Flow-GRPO RL on the flow head** | Very high | 6 | High | **Accept as a gated stretch** |
| 9 | F2. 2-frame history ("video tokens") | Low-medium: 2 frames is a thin video claim, and frame history risks copycat failure in BC | 2+ | Medium | **Defer** |
| 10 | Custom Triton fused constrained-sampling kernel | Low: once lm_head is sliced, sampling over 256 logits is under 1% of decode time | — | — | **Reject.** README says why. |

Cuts from v1:
- No `flash-attn` pip build. There is no cu130/torch-2.14 wheel, and without sudo a source build means hours of nvcc. Use the PyTorch SDPA flash backend and label it honestly.
- No DeepSpeed training run. ZeRO-3 appears only in the scaling sweep, with `DS_BUILD_OPS=0` and torch AdamW.
- `reach` is saturated, so it is evaluated once and excluded from RL and ablations.

## Shared-GPU defensibility (applies to every benchmark)
- **Device pinning:** export `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0,1,2,4`. The learner and rollout processes each cap at 20 GB, so they fit beside other users. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Memory:** report only process-local `torch.cuda.max_memory_allocated` and `max_memory_reserved`, never nvidia-smi totals.
- **GPU snapshot:** every result JSON embeds `gpu_snapshot_{before,after}` from `scripts/gpu_snapshot.py`. It records per-GPU util, memory used, foreign compute apps and their memory, plus host load. Any foreign util above 20% sets `contended: true`.
- **Quiet window:** `scripts/quiet_wait.py --gpus 0,1,2,4 --util-max 15 --hold 60 --timeout 1200` blocks until a quiet window. On timeout the run still happens, flagged as contended.
- **Repeats:** timing results use 5 repeats, reported as median plus min/max. A-vs-B comparisons run interleaved (ABABAB) so drift cancels out.
- **Wording:** the README states the contention protocol, and every timing number carries its contended flag.

## Revised phase list
Phases marked "as v1" are unchanged.

- P0, P1: done.
- **P1b: DR fixes + DTD textures (new)**
- **P2: data (delta)**
- **P3: model (delta)**
- **P4: BC + async checkpointing + MFU (delta)**
- **P4b: perf study (new)**
- **P5: eval + ablations (delta)**
- **P6: async actor-learner RL (replaces v1 P6)**
- **P6b: Flow-GRPO (gated)**
- **P7: inference (delta)**
- P8: README, as v1 plus new tables

### P1b: Held-out axis fixes + real-photo textures (LOCAL + remote CPU; 0 GPU-h)
Files: `sim/randomization.py`, `sim/assets.py`, `scripts/fetch_dtd.sh`.

Light direction:

| Split | Azimuth | Elevation |
|---|---|---|
| train | within ±60° of front | 45–75° |
| heldout | 90–180° (side or back light) | or 20–40° |

kp jitter:

| Split | Range |
|---|---|
| train | ±10% |
| heldout | 10–25% magnitude |

DTD textures:
- `fetch_dtd.sh` downloads DTD to `$VLA_DATA/dtd`.
- `assets.py` loads 48 textures resized to 256² as MjSpec textures and materials, uploaded once at context creation.
- Categories 1–30 go to train. Categories 31–47 are used only by a new split, `heldout_photo`: heldout ranges plus the unseen photo textures on the table and floor.

Re-measure expert ceilings for every task and split and write them to `results/expert_ceiling.json`.

Accept:
- `tests/test_env.py` is extended. Heldout light direction and kp fall outside the train ranges.
- `heldout_photo` only ever uses material IDs from categories 31–47.
- The ceiling JSON exists.

### P2 delta
- Train data includes DTD categories 1–30.
- Demo subsets are written as index files: `subset_{10,30,100}.json`.
- Ablation datasets each fix one DR group at nominal, giving `demos_nocam/`, `demos_novis/` (lighting + texture) and `demos_nophys/` (friction, mass, kp). These run on CPU and take about 1 h each with 64 workers.
- Before training, copy shards to `/dev/shm/vla_data` so NFS does not throttle data loading. NFS stays the master copy.

### P3 delta
Files: `model/flow_head.py`, `model/vla.py`.

Flow action expert (GR00T/π0-style):
- A 6-layer DiT with d=512, about 40M params.
- Input tokens: the noisy action chunk (4 tokens of 4 dims each) plus the proprio token.
- Conditioning: adaLN on timestep t, and cross-attention to the backbone's last hidden states over the prefix (image + proprio + prompt).
- Rectified flow:
  - x_t = t·ε + (1−t)·a, with target velocity u = ε − a.
  - t is sampled from π0's Beta distribution (more weight at high noise).
  - Inference uses 10 Euler steps from t=1 to t=0.
- `head: discrete|flow` in config. In flow mode, the backbone sees only the prefix.

Discrete decode: build an inference-only `ActionLMHead` that holds the 256 action rows of the tied embedding. That is 0.23M params instead of 136M.

Accept (tiny, CPU):
- Flow loss on one overfit batch drops by more than 90%.
- With the noise fixed, the sampler is deterministic.
- Sliced-head logits equal the full-head logits gathered at the action IDs (tolerance 1e-5).

### P4 delta (11.5 GPU-h)
- **Main runs:** two BC runs, `bc_disc` and `bc_flow`. Same backbone init, data, steps and seed. About 5 GPU-h each on 4 GPUs.
- **MFU:** FLOPs/token = 6·N + 12·L·d·S (forward plus backward, including attention). Peak is 312 TFLOPS bf16. Log `mfu` per step.
- **New flags:** `act_ckpt: bool`, plus FSDP2 `reshard_after_forward` and explicit forward/backward prefetch.
- **Async checkpointing (E):**
  - `checkpoint.py` adds `mode: sync|async` using `dcp.async_save` on a gloo process group, staged to pinned CPU memory.
  - Before the next save, wait on the previous future.
  - Update `latest` only after the future completes.
- **Stall measurement:** log `ckpt_stall_s` as the wall time the train loop is blocked.
  - Test matrix: {sync, async} × {NFS, /dev/shm}, 5 saves each, interleaved.
  - /dev/shm checkpoints are mirrored to NFS by a background `rsync`.
  - Output goes to `results/ckpt_stall.json`.
- **Resume test:** the v1 resume test reruns using async checkpoints.

Accept:
- Async stall is under 20% of sync stall on NFS.
- The resume test passes for both modes.
- `memory.json` is as in v1.
- Both final checkpoints exist.

### P4b: Scaling and profiling study (REMOTE; 3 GPU-h)
Files: `scripts/scaling_sweep.sh`, `scripts/profile_step.py`.

Sweep:
- Configs: {FSDP2, ZeRO-3} × {1, 2, 4} GPUs × {act_ckpt off, on}.
- Each config runs 120 steps, the first 20 dropped, with 2 interleaved repeats, under quiet-wait.
- Report samples/s, MFU, scaling efficiency (throughput_N / (N·throughput_1)), and peak memory.

Traces:
- A torch.profiler Chrome trace of 3 FSDP2 steps on 4 GPUs.
- Use nsys if it is available without sudo. If not, torch.profiler alone.
- Annotate where all-gather and reduce-scatter overlap compute, with prefetch on vs off.

Outputs: `results/scaling.json` and gzipped traces under 20 MB in `results/traces/`.

Accept: the JSON has all 12 configs with snapshots, and the README section has an overlap screenshot.

### P5 delta (13 GPU-h, eval plus ablation training)
Eval protocol:
- Splits: train, heldout, heldout_photo.
- 200 episodes per task (lift, push), with fixed seed sets.
- Report Wilson CIs, plus success normalized by the expert ceiling for that task and split.

Short-schedule ablations (one-third of the main schedule, 1.5 GPU-h each, 1 seed, labeled as such):
- **Baseline:** short schedule on full data.
- **DR ablation:** no-camera, no-visual, no-physics.
- **Data scaling:** 10% and 30% of demos, at constant steps.

Output: `results/ablations.json`.

Accept: all 6 short runs plus the 2 main runs are evaluated on 3 splits, and `make_tables.py` renders the ablation and scaling plots.

### P6: Decoupled actor-learner RL (replaces v1 P6; 3 GPU-h study + 16 GPU-h runs)
Files: `rl/{roles,weight_sync,rollout_worker,learner,algo}.py`, `infer/engine.py`, `configs/rl*.yaml`.

**Topology:**
- One torchrun world. `learner_ranks` and `rollout_ranks` come from config: 1+1 for the seed runs, and 2+2 or 1+3 for the throughput study.
- The learner uses FSDP2 over its own sub-mesh.
- `sync_pg` connects learner rank 0 to all rollout ranks.

**Rollout worker:**
- Runs `infer/engine.py`, the shared inference engine also used by P7: batched (B=64) static KV cache, CUDA graphs, sliced head.
- Drives a 64-env `AsyncVectorEnv` with EGL on its own GPU.
- Groups are 8 seeds × G=8.
- Each batch is written atomically to `/dev/shm/vla_rl/<run>/` (tmp file, then rename). A batch holds uint8 obs, tokens (or flow latents), behavior log-prob μ, policy version and reward components.

**Weight sync:**
1. Learner rank 0 calls `full_tensor()` on each param and packs the params into 256 MB buckets.
2. It runs an NCCL broadcast over `sync_pg`.
3. Rollout ranks `copy_` into their existing param storage, so the CUDA graphs stay valid.

**Staleness:**
- The rollout joins a sync after every S batches.
- S=0 is serial on-policy, the baseline. S=1 is one-step off-policy, overlapping rollout k+1 with update k.
- By construction, the version gap is at most S.

**Loss:**
- Per-token PPO-clip against π_old, the learner's log-prob recomputed at the start of the update.
- Multiplied by a detached truncated-IS weight w = min(π_old/μ, 2).
- Plus a KL penalty to the frozen BC reference with β=0.02.
- Rewards are as in v1.

**Logging:**
- env-steps/s, learner tokens/s
- sync latency (gather plus broadcast, in ms)
- per-role idle fraction
- staleness histogram
- w mean, max and clip fraction
- **engine mismatch** = mean |log π_learner − log μ| at equal version (the inference vs FSDP numerics gap that justifies TIS)

**Runs:**
- Throughput study: S∈{0,1} × topology {1+1, 2+2, 1+3}, 20 min each.
- `rl_disc` main: 3 seeds, 1+1 topology, fixed 100 iterations, about 4 GPU-h each.
- Penalty ablation: λ_j = λ_jk = 0, 1 seed, 4 GPU-h.

Accept:
- **Tests (gloo, CPU, 2 procs, tiny model):**
  - After a sync, rollout params bitwise equal the learner's full params.
  - After `copy_`, CUDA-graph or compiled output matches eager output with the new weights (GPU test).
  - w = 1 when μ = π.
- **Remote:**
  - S=1 env-steps/s is at least 1.3× S=0 on the same topology.
  - Mean return rises across all 3 seeds.
  - Final eval goes to `results/eval_rl.json` as mean ± std across seeds plus per-seed Wilson CIs.
  - The no-smoothness ablation shows the jerk difference.

### P6b: Flow-GRPO (gated stretch; 6 GPU-h)
**Gate (all three must hold):**
1. P6 is accepted.
2. `bc_flow` train success is at least 0.5 and within 10 points of `bc_disc`.
3. At least 8 GPU-h remain.

If the gate fails, flow is reported as a BC head-to-head only.

Method:
- Use the Flow-GRPO SDE sampler (Euler–Maruyama with σ_t = a·√(t/(1−t)), a=0.7, following the paper's equation exactly).
- Each step is Gaussian, so log-prob is the sum over steps. The ratio is computed per denoising step.
- K=5 steps during RL, and 10-step ODE at eval.
- KL is closed-form Gaussian.
- The same P6 infrastructure runs it, with latents stored in place of tokens.

Accept:
- **Toy test:** with σ→0 the sampler equals the ODE. With σ>0, the 1-D Gaussian marginals match the ODE within 2%.
- **Remote:** flow-RL results on 3 splits, compared against `rl_disc` seed 0.

### P7 delta (2 GPU-h)
- Rows:
  1. fp32 eager, no cache
  2. bf16 eager, no cache
  3. bf16 SDPA + KV cache
  4. row 3 + sliced lm_head
  5. row 4 + static cache + CUDA graphs (`torch.compile` reduce-overhead or manual capture)
  6. the flow head: 1 prefill + 10 DiT steps, with CUDA graphs
- Each row runs at batch 1 and batch 64.
- **Roofline:**
  - At batch 1, arithmetic intensity is about 1 FLOP/byte, against an A100 ridge of about 150, so decode is memory-bound.
  - HBM floor = bytes read per token / 2.0 TB/s. Report the measured/floor ratio.
  - Use profiler kernel-gap traces to show that eager mode is bound by launch overhead.
  - Show the lm_head share of bytes before and after slicing.
- Output: `results/latency.json` plus a trace.

Accept:
- Best discrete row p50 is under 100 ms per query.
- Token agreement with row 2 is at least 99%.
- Every row has snapshots, 5 repeats and a contended flag.

## GPU-h total
| Item | GPU-h |
|---|---|
| P3 | 0.3 |
| P4 | 11.5 |
| P4b | 3 |
| P5 | 13 |
| P6 | 19 |
| P6b | 6 |
| P7 | 2 |
| **Planned** | **≈55** |
| Contention reserve | 5 |
| **Total** | **≤60** |

If the budget overruns, cut in this order:
1. The 30% data point
2. The no-camera ablation
3. RL seed 3 (report n=2 honestly)
4. P6b

## Resume bullets (template; measured numbers only)
- Built a domain-randomized MuJoCo manipulation suite with a held-out real-photo-texture (DTD) split. VLA reaches [x]% of expert-ceiling success in-distribution and [y]% on unseen photo textures. DR ablations attribute [z] points of OOD success to [axis].
- Trained a 0.58B VLA (SigLIP + Qwen2.5-0.5B) with discrete-token and flow-matching (DiT, rectified flow) action heads. Flow vs discrete: [a]% vs [b]% success and [c] vs [d] ms latency.
- Built PyTorch FSDP2 training: [m]% MFU, [e]% 4-GPU scaling efficiency vs ZeRO-3's [f]%, activation checkpointing and comm/compute overlap verified in profiler traces. Async DCP checkpointing cut the per-checkpoint stall from [s1]s to [s2]s, and a kill-resume test matched within [r]%.
- Built decoupled actor-learner RL: CUDA-graph rollout engine, NCCL weight sync in [ms] ms, one-step off-policy with truncated importance sampling. [k]× env-steps/s over synchronous. GRPO-style RL with smoothness and safety penalties: success [p0]→[p1]% (3 seeds), jerk −[j]%.
- (if P6b passes) RL-fine-tuned the flow-matching action head with Flow-GRPO (SDE sampling for tractable log-probs): [q0]→[q1]%.
- Cut action latency [t0]→[t1] ms p50 on A100 (roofline analysis, action-vocab head slicing, static KV cache, CUDA graphs).

## Top 3 risks
1. **Shared GPUs (others at 60–99% util) distort timing and stretch runs.**
   - Use the quiet-wait protocol, interleaved repeats and contended flags.
   - Correctness and memory claims do not depend on timing.
   - Stay within the 5 GPU-h reserve and follow the cut order.
   - Keep each process at 20 GB or less so other users' memory does not OOM us.
2. **The async RL infrastructure deadlocks, or weight swaps break CUDA graphs.**
   - Build S=0 first in the same code, and pass the gloo CPU tests before running on GPU.
   - Set `NCCL_TIMEOUT` and add a heartbeat in the logs.
   - Fallback: run S=0 colocated (v1-equivalent) so RL results still land, and claim only what ran.
3. **The flow head underperforms, or Flow-GRPO is unstable.**
   - The gate isolates it. The BC head-to-head stays reportable either way.
   - Sweep a∈{0.5, 0.7} only if budget allows. Otherwise report the negative result as-is.
