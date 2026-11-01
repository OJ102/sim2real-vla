# Closed-Loop VLA Post-Training Pipeline: Build Plan v1

Author: Opus planner. Status: approved 2026-10-31. Budget: about 30 A100 GPU-hours.

## Design decisions
1. **Sim: custom MuJoCo tabletop with a Franka Panda** (from `robot_descriptions` / menagerie). LIBERO pins old versions of robosuite and mujoco, and its domain-randomization (DR) hooks are limited. A custom env gives full DR control, gymnasium vector envs and EGL rendering in about 500 lines of code.
2. **Tasks: 3 colored cubes and 3 tasks** (`reach`, `lift`, `push-to-zone`). Each instruction names a color and distractor cubes are present, so the model must actually use the language input.
3. **Action space: 4-D end-effector (EE) delta (dx, dy, dz, gripper) at 10 Hz.** Damped-least-squares IK maps it to joint position targets. The gripper stays pointed down, so the scripted expert stays trivial.
4. **Observations:** one front camera at 128x128 (resized to 224 in the model) plus proprioception (EE position, gripper width, 7 joint angles). This keeps the dataset around 8 GB.
5. **Model: SigLIP-base/16-224 (86M) + 2-layer MLP projector + Qwen2.5-0.5B, about 0.58B parameters in total.** It trains in hours. Full fp32 Adam state is about 9.3 GB unsharded, so FSDP shows a real drop in per-GPU memory. One 200-step run with Qwen2.5-1.5B shows FSDP fitting where DDP is tight.
6. **Actions: OpenVLA-style 256-bin tokens.** There are 256 new `<act_i>` tokens and the chunk size is H=4, so each query produces 16 tokens. Discrete tokens give exact log-probs for the RL ratio, and autoregressive decoding gives the KV-cache real work to do.
7. **FSDP2 (`fully_shard`)**, wrapping each Qwen2DecoderLayer and SiglipEncoderLayer. Params are bf16 and gradient reduction is fp32. Checkpoints are saved with `torch.distributed.checkpoint` (DCP) and `get_state_dict`. DeepSpeed ZeRO-3 is a second backend in the same script.
8. **RL: GRPO-style.** This is the PPO clipped objective with group-relative advantages and no value network. Each group is G rollouts from the same seed, so they share the same DR draw, and the group baseline isolates the policy's own behavior. Episodes are short (at most 100 steps) and the terminal reward dominates. A critic on a 0.58B VLM would add memory and instability. A KL penalty to the frozen BC reference applies.
9. **RL runs on DDP, not FSDP.** The model fits when replicated on each GPU. Token-by-token rollout under FSDP would all-gather the params on every decode step. The resume test and the sharding claims come from the BC phase.
10. **Logging:** a JSONL metrics file per run is the source of truth, and the tester parses it. TensorBoard (`torch.utils.tensorboard`) writes to `$VLA_RUNS/<name>/tb/` from the same logger. No wandb (user decision).
13. **No sudo on the server (user decision).** No apt installs. Headless rendering relies on the NVIDIA driver's own `libEGL_nvidia` (check `ldconfig -p | grep -i egl`). If that is missing, the fallback is OSMesa from conda-forge (`mesalib`) in user space. `setup.sh` must never call sudo.
11. **Config:** YAML files loaded into dataclasses, with CLI overrides as `key=value`. No Hydra.
12. **Multi-node readiness:** every launch uses `torchrun --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 --nnodes=1`. The README says "single-node 4xA100, multi-node-ready launch". It never claims a multi-node run.

## Repo layout
```
vla/
  pyproject.toml  uv.lock  README.md  .env.example  .gitignore
  configs/  data.yaml  model_tiny.yaml  model_0p5b.yaml  model_1p5b.yaml
            bc.yaml  bc_tiny.yaml  ds_zero3.json  rl.yaml  rl_tiny.yaml
  src/vla/
    sim/    assets.py  randomization.py  env.py  ik.py  expert.py  rewards.py
    data/   generate.py  dataset.py  sampler.py
    model/  vla.py  action_tokenizer.py  prompts.py
    train/  bc.py  dist.py  checkpoint.py  ds_backend.py
    rl/     rollout.py  grpo.py
    eval/   rollout_eval.py  metrics.py
    infer/  policy.py  benchmark.py
    utils/  config.py  logging.py
  scripts/  remote/{setup.sh,sync.sh,launch.sh,pull_results.sh}
            resume_test.sh  make_tables.py
  tests/    (owned by the tester)
  results/  (small committed JSON/CSV/PNG/MP4 files)
```
Remote paths come from `.env`: `VLA_DATA=/path/data/vla` and `VLA_RUNS=/path/runs/vla`. Each run writes `$VLA_RUNS/<name>/{config.yaml,metrics.jsonl,stdout.log,ckpt/step_XXXXXX/,ckpt/latest}`.

## Phases

### Phase 0: Scaffold and remote workflow (LOCAL; 0 GPU-h)
Files: `pyproject.toml`, `.gitignore`, `.env.example`, `src/vla/utils/{config,logging}.py`, `scripts/remote/*`.
- Use uv with Python 3.11.
  - Core deps: torch>=2.6, transformers, mujoco>=3.2, gymnasium>=1.0, robot_descriptions, h5py, imageio[ffmpeg], pyyaml, numpy.
  - Extra `[gpu]`: deepspeed and flash-attn, installed on the remote server only.
  - Commit `uv.lock`.
- Run `git init` and add a private GitHub remote. To sync code, `git push` locally, then run `ssh srv 'cd ~/vla && git pull'`. `sync.sh` is an rsync fallback that excludes data and runs.
- `setup.sh` installs uv and runs `uv sync --extra gpu`. It exports `MUJOCO_GL=egl`, then runs a check that prints the CUDA device count and does one EGL render.
- `launch.sh <name> <cmd...>` runs `tmux new -d -s <name> "<cmd> 2>&1 | tee $VLA_RUNS/<name>/stdout.log"`.
- `pull_results.sh` rsyncs `results/` and `$VLA_RUNS/*/metrics.jsonl` back to the local machine. It never pulls checkpoints.

Acceptance:
- `uv sync && uv run python -c "import vla, mujoco, torch, transformers"` exits 0.
- On the remote server, `bash scripts/remote/setup.sh` prints `cuda devices: 4` and `egl render ok (128, 128, 3)`.

### Phase 1: Env, domain randomization and expert (LOCAL smoke, REMOTE EGL check; 0 GPU-h)
Files: `sim/{assets,randomization,env,ik,expert,rewards}.py`.
- `assets.py` builds the MJCF scene: the Panda, a table, 3 cubes, a goal zone, 1 camera and 3 lights. It predefines 12 materials/textures (checker, gradient and flat builtins). IDs 0-7 are used for training and IDs 8-11 are held out.
- `randomization.py` defines `DRConfig` with `train` and `heldout` ranges. Held-out ranges are disjoint from the training ranges or wider:
  - Camera: position ±3 cm, look-at point ±2 cm, fov ±4° (held out: ±6 cm and ±8°).
  - Lighting: random light direction, diffuse 0.4-1.0 (held out: 0.15-0.4 or 1.0-1.4), 1-3 lights.
  - Materials: table and floor materials swapped via `geom_matid`. Cube shades are jittered, but each cube stays recognizably red, green or blue.
  - Friction: 0.6-1.2 (held out: 0.3-0.6 or 1.2-1.8).
  - Cube mass: 0.05-0.2 kg (held out: 0.2-0.4 kg).
  - Actuator kp: ±10%.
  - All of these are applied to `mjModel` at `reset(seed)`.
- `env.py` defines `TabletopEnv(gymnasium.Env)` with `task`, `split` and `image_size`. Each step holds the action for 0.1 s, and an episode runs at most 100 steps. `info` holds `success`, `qacc`, `qvel`, `contacts` and `ee_pos`.
- `expert.py` is a waypoint controller per task with small added noise. `rewards.py` holds pure reward functions (spec in Phase 6).

Acceptance:
- `uv run python -m vla.sim.env --demo --task lift --episodes 3 --out results/demo` writes 3 MP4s.
- `uv run python -m vla.sim.expert --episodes 100 --split train` prints a success rate of at least 0.95 for each task.
- The tester writes `tests/test_env.py`, which checks three things:
  - The same seed gives identical observations.
  - Different seeds give different camera, material and friction values.
  - Held-out samples fall outside the training ranges.
- On the remote server, the same expert command succeeds under `MUJOCO_GL=egl`.

### Phase 2: Data generation and dataset (REMOTE CPU; 0 GPU-h, about 1 h wall time)
Files: `data/{generate,dataset,sampler}.py`, `configs/data.yaml`.
- Workers run in a multiprocessing Pool, each with its own env and EGL context, and only successful episodes are kept.
  - Target: 3 tasks × 1,000 episodes × 3 colors, about 9k episodes and 450k frames.
  - Output: HDF5 shards (`lzf`) with `images uint8[T,128,128,3]`, `proprio`, `actions`, `instruction`, `task` and `seed`.
- `dataset.py` returns (image, proprio, instruction, action chunk [t:t+4]). Chunks that run past the episode end are padded with the last action and flagged by a mask. Normalization stats are stored in `stats.json`.
- `sampler.py` defines `ResumableDistributedSampler(seed, epoch, consumed)` with a `state_dict()` method. It is deterministic and skips samples already consumed.

Acceptance:
- `python -m vla.data.generate --config configs/data.yaml --num-workers 32` writes `$VLA_DATA/demos/*.h5` and `stats.json`, and reports at least 8k episodes.
- Tester test: take two samplers with the same seed and restore one from a mid-epoch `state_dict`. Both must yield the same remaining index sequence.

### Phase 3: VLA model and action tokenizer (LOCAL smoke; 0.2 GPU-h remote check)
Files: `model/{vla,action_tokenizer,prompts}.py`, `configs/model_{tiny,0p5b,1p5b}.yaml`.
- `ActionTokenizer` clips actions to [-1, 1] and maps them to 256 uniform bins, with encode and decode methods.
- The input sequence is `[196 projected image tokens][1 proprio token (MLP)][prompt tokens "In: {instruction}\nOut:"][16 action tokens]`.
- Loss is cross-entropy on the action tokens only. The attention implementation is configurable: `eager|sdpa|flash_attention_2`.
- `model_tiny.yaml` builds a 2-layer Qwen2 and a 2-layer SigLIP with `from_config`, so it needs no weight download.

Acceptance:
- The tester writes `tests/test_model_tiny.py`, which checks three things:
  - A forward/backward pass on CPU gives a finite loss.
  - Overfitting one batch for 100 steps cuts the loss by more than 90%.
  - Tokenizer round-trip error is at most 1/255.
- On the remote server, `python -m vla.model.vla --config configs/model_0p5b.yaml --count-params` prints about 0.58B.

### Phase 4: Distributed BC with FSDP2, DCP and DeepSpeed (LOCAL gloo smoke, REMOTE; about 10 GPU-h)
Files: `train/{bc,dist,checkpoint,ds_backend}.py`, `configs/{bc,bc_tiny}.yaml`, `configs/ds_zero3.json`, `scripts/resume_test.sh`.
- `dist.py` handles process-group init (nccl, or gloo on CPU), `wrap_fsdp2(model)` and `wrap_ddp(model)`.
- `bc.py`:
  - Takes `--backend fsdp|ddp|deepspeed`.
  - Optimizer: AdamW, lr 1e-4 (1e-5 for the vision encoder), cosine schedule, 500 warmup steps, grad clip 1.0.
  - Training: global batch 256, bf16, about 8 epochs.
  - Logs `loss`, `tok_acc`, `grad_norm`, `step_time`, `samples/s` and per-rank `torch.cuda.max_memory_allocated`.
- `checkpoint.py`:
  - Saves model, optimizer, scheduler, sampler, RNG state and step with DCP to `ckpt/step_N`. After the save finishes, rank 0 atomically rewrites `ckpt/latest`.
  - Saves every `ckpt_every` steps.
  - A SIGTERM/SIGUSR1 handler sets a flag, and the job saves and exits at the next step boundary.
  - `--resume` loads `latest`.
- `ds_backend.py` calls `deepspeed.initialize` with ZeRO-3 and bf16, and uses `engine.save_checkpoint` / `load_checkpoint`.
- `resume_test.sh` runs these steps in order:
  1. Run A: an uninterrupted 600-step reference run with `ckpt_every=100`.
  2. Run B: start the same config and wait until about step 350, then kill it with `kill -9`.
  3. Relaunch run B with `--resume`. It must restart from step 300.
  4. Run C: start the same config and send `kill -TERM` at about step 250. It must save at that step and resume from exactly that step.
  5. Compare the loss curves of the three runs.

Acceptance:
- LOCAL: `torchrun --nproc_per_node=2 -m vla.train.bc --config configs/bc_tiny.yaml` (gloo on CPU) runs 20 steps, saves a checkpoint and resumes from it. If the installed torch does not support FSDP2 on CPU, run this smoke test on 2 remote GPUs instead and note that.
- REMOTE resume test:
  - `bash scripts/resume_test.sh` prints `resume_from=300`.
  - Over steps 301-600, the mean relative loss difference between runs B and A is under 1%.
  - Run C resumes from the step it saved at.
- The full BC run reaches `tok_acc` above 0.8 on held-out demos. The final checkpoint goes to `$VLA_RUNS/bc_0p5b/`.
- Memory table in `results/memory.json`:
  - Per-GPU peak memory for DDP vs FSDP at 0.5B (50 steps each).
  - FSDP at 1.5B (200 steps).
  - DDP at 1.5B is attempted, and the OOM or the peak is recorded.
- DeepSpeed: run 300 steps. Loss at step 300 must be within 5% of FSDP's loss at step 300, and checkpoint save/load must work.

### Phase 5: Closed-loop eval harness and BC baseline (REMOTE; about 2 GPU-h)
Files: `eval/{rollout_eval,metrics}.py`, `infer/policy.py` (basic greedy decoding with KV-cache, with logits masked to action tokens).
- Run 200 episodes per task per split (`train` and `heldout`), batched across a vector env. Eval seeds are fixed and disjoint from the training seeds.
- Metrics:
  - Success rate with a Wilson 95% confidence interval.
  - EE jerk RMS (third finite difference of `ee_pos`).
  - Mean absolute second difference of actions.
  - Joint-limit, velocity and collision violations per episode.
- Also evaluate the scripted expert as a reference.

Acceptance:
- `python -m vla.eval.rollout_eval --ckpt $VLA_RUNS/bc_0p5b/ckpt/latest --split train,heldout --episodes 200` writes `results/eval_bc.json` with all the metric fields.
- Gate to proceed: BC success on the train split is at least 0.5. If it falls short, see the fallback under Risks.

### Phase 6: RL post-training, GRPO-style (REMOTE; about 16 GPU-h)
Files: `rl/{rollout,grpo}.py`, `configs/{rl,rl_tiny}.yaml`, final version of `sim/rewards.py`.
- Reward per episode:
  `R = 10·success − λ_j Σ‖a_t−2a_{t−1}+a_{t−2}‖² − λ_jk Σ‖Δ³ee_pos‖² − λ_lim Σ 1[q within 5% of limit] − λ_vel Σ relu(|qd|−0.8·qd_max) − λ_col Σ 1[arm–table or arm–distractor contact]`
  - Default weights: λ_j=0.5, λ_jk=50, λ_lim=0.05, λ_vel=0.1, λ_col=0.1.
  - Each term is logged separately.
- Each iteration, on each rank:
  - Rollout:
    - 8 seeds × G=8 rollouts on train-split DR, using `AsyncVectorEnv` with 64 envs at temperature 1.0.
    - Action chunks run open-loop.
    - Store the sampled tokens and their old log-probs.
  - Advantage: (R − group mean) / (group std + 1e-6), broadcast to every token of the episode.
  - Update:
    - 2 epochs of per-token PPO-clip with ε=0.2.
    - KL penalty to the frozen BC reference with β=0.02 (k3 estimator).
    - lr between 1e-6 and 5e-6, vision encoder frozen, DDP over 4 GPUs.
- Run about 100 iterations. Evaluate 50 episodes per split and save a checkpoint every 10 iterations.
- `rl_tiny.yaml` gives a CPU smoke config: 2 iterations, 4 envs, tiny model.

Acceptance:
- Local smoke: `python -m vla.rl.grpo --config configs/rl_tiny.yaml` finishes 2 iterations with finite loss and KL.
- The tester writes `tests/test_rewards.py`, which checks two things:
  - A constant action gives a zero jerk penalty.
  - A known joint-limit violation is counted.
- Remote: `metrics.jsonl` shows the mean return trending up, with KL below 0.1.
- The final eval writes `results/eval_rl.json` in the same format as the BC eval. `make_tables.py` prints the BC vs RL table for the train and held-out splits.

### Phase 7: Inference optimization benchmark (REMOTE, 1×A100; about 1 GPU-h)
Files: `infer/{policy,benchmark}.py`.
- `policy.py` decode modes:
  - (a) No cache: recompute the full sequence for every token.
  - (b) `DynamicCache`.
  - (c) `StaticCache` + `torch.compile(mode="reduce-overhead")` on the decode step.
  - The vision encoder is compiled separately.
- Benchmark rows:
  1. fp32, eager attention, no cache
  2. bf16, eager attention, no cache
  3. bf16, SDPA, KV-cache
  4. bf16, flash_attention_2, KV-cache
  5. Row 4 plus static cache and compile
- Method:
  - Batch size 1, 50 warmup queries, then 500 timed queries, with `torch.cuda.synchronize`.
  - Report p50/p95 ms per query (16 tokens), amortized ms per action (query time ÷ 4) and peak memory.
  - Report token agreement with row 2 on 200 fixed observations.
- Also report end-to-end control-loop latency (render + preprocess + policy).

Acceptance:
- `python -m vla.infer.benchmark --ckpt ... --out results/latency.json` writes all rows.
- The best row's p50 is under 100 ms per query.
- Rows 3-5 agree with row 2 on at least 99% of tokens.
- The tester writes `tests/test_kvcache_equiv.py` (tiny model, CPU), which checks that greedy decoding produces the same tokens with and without the cache.

### Phase 8: Results and README (LOCAL; 0 GPU-h)
`scripts/make_tables.py` turns `results/*.json` into markdown tables. The README holds the architecture diagram, the commands to reproduce each result, the tables and 2 GIFs (rollouts under train DR vs held-out DR).

**Total: about 30 GPU-h** (BC 7, resume/DeepSpeed/memory 3, eval 3, RL 16, benchmark 1).

## Final deliverables: README tables
1. **Policy results**, with train-DR and held-out-DR columns. Rows: Expert, BC, BC+RL. Metrics: success % ± CI, jerk RMS, violations per episode.
2. **Distributed training**:
   - Per-GPU peak memory for DDP vs FSDP at 0.5B and 1.5B.
   - Samples/s.
   - FSDP vs ZeRO-3 loss at step 300.
   - Resume-test loss difference.
3. **Latency**: the 5 benchmark rows with p50, p95, amortized per-action time and peak memory.

## Resume bullets (fill in only measured numbers)
- Built a MuJoCo tabletop simulator (Franka Panda, 3 language-conditioned tasks) with domain randomization over camera pose, lighting, textures, friction and mass. Generated [N]k expert demos headless (EGL), with a held-out randomization split as a sim-to-real proxy.
- Trained a [0.58]B VLA (SigLIP + Qwen2.5-0.5B, 256-bin action tokens, 4-step chunks) on 4×A100 with PyTorch FSDP2 (bf16, full sharding). Cut per-GPU memory from [X] GB (DDP) to [Y] GB, and added a DeepSpeed ZeRO-3 path.
- Built preemption-safe sharded checkpointing (torch.distributed.checkpoint + SIGTERM handling). A kill-and-resume test matched the uninterrupted loss curve within [Z]%.
- Fine-tuned the policy with GRPO-style RL (PPO-clip, group-relative advantages, KL to the BC policy) using jerk, joint-limit, velocity and collision penalties. Success went from [a]% to [b]% in distribution and from [c]% to [d]% on held-out randomization, and jerk dropped [e]%.
- Cut action-query latency from [p] ms to [q] ms p50 on one A100 (FlashAttention-2, KV-cache, static cache + torch.compile).

## Risks and fallbacks
- **BC success below 50%:**
  - First, generate more demos, train more epochs and check action normalization.
  - If success is still low, drop to a chunk size of H=1 or add a wrist camera.
- **RL flat or collapsing:**
  - Lower the lr, raise β or raise G to 16.
  - If the sparse reward is too weak a signal, add a dense shaping term (EE-to-cube distance).
  - If success still does not improve, report the smoothness gains honestly.
- **Latency over 100 ms:**
  - Report amortized per-action latency (query time ÷ 4) separately and honestly.
  - Next step: cut decoding to 8 tokens (H=2).
- **flash-attn wheel fails to install:** use the SDPA flash backend and label it "FlashAttention-2 kernel via PyTorch SDPA".
- **FSDP2 on CPU/gloo unsupported:** run the smoke test on 2 remote GPUs.
- **EGL fails across many subprocesses:** set `MUJOCO_EGL_DEVICE_ID` per worker, or fall back to OSMesa for data generation.
- **DeepSpeed install problems:** this path is optional. Claim it only if the Phase 4 DeepSpeed acceptance passes.

## Critical files
- src/vla/sim/env.py (+ randomization.py)
- src/vla/model/vla.py
- src/vla/train/bc.py (+ checkpoint.py)
- src/vla/rl/grpo.py
- src/vla/infer/policy.py

---

# PLAN v2 delta (Opus planner, 2026-11-01). Status: APPROVED 2026-11-01

The full v2 text is in docs/PLAN_v2.md and supersedes v1 wherever they differ. Summary of its decisions:
- Accepted: A (decoupled actor-learner RL, replaces RL-on-DDP), lm_head slicing to the 256 action rows plus a roofline analysis, C (MFU/scaling/profiling), D (seeds and ablations, trimmed), B1 (flow-matching DiT action head, BC head-to-head), E (async DCP checkpointing), F1/F3 (DTD photo-texture heldout split, true heldout light direction and kp).
- B2 (Flow-GRPO) is a gated stretch.
- Deferred: F2 (2-frame history). Rejected: Triton kernel.
- Cuts: no flash-attn build (use SDPA flash backend, labeled), no DeepSpeed training run (ZeRO-3 only in the scaling sweep), reach excluded from RL and ablations.
- Budget: about 55 GPU-h plus a 5 GPU-h reserve, 60 GPU-h or less in total.
