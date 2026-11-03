# sim2real-vla: closed-loop VLA post-training on a randomized MuJoCo tabletop

This repository trains a small vision-language-action (VLA) policy for a simulated Franka Panda, evaluates it in
closed loop, and post-trains it with reinforcement learning. The point of the project is the infrastructure around the
policy: distributed training, checkpointing, a decoupled actor-learner RL loop, and an inference path that is
analysed against the hardware limits. The policy itself is small and the tasks are simple.

Numbers in this README come from measured runs; anything not yet measured is marked `TBD`. The final tables are
rendered by `scripts/make_tables.py` from the JSON files in `results/` into `results/tables.md`. The behaviour-cloning
numbers below come from `results/eval_bc_*.json` and `results/probe_bc_*.json`.
Nothing here claims a multi-node run: launches are multi-node-ready (`torchrun` with a c10d rendezvous), but every
reported result is single-node on four A100s that were shared with other users.

## What is in the box

| Area | What it does | Code |
|---|---|---|
| Simulator | MuJoCo tabletop, Franka Panda, three coloured cubes, tasks `reach`, `lift`, `push`; 4-D end-effector delta actions at 10 Hz, damped-least-squares IK | `src/vla/sim/` |
| Domain randomization | Camera, lighting, textures, friction, mass, actuator gains; three splits (`train`, `heldout`, `heldout_photo`) | `sim/randomization.py` |
| Expert and data | Scripted waypoint expert, parallel headless generation to HDF5 shards (with per-frame cube positions and the target index), resumable sampler with early-frame oversampling | `sim/expert.py`, `src/vla/data/` |
| Model | SigLIP-base + 2-layer MLP projector + Qwen2.5-0.5B (0.58B parameters); discrete-token or flow-matching action head; optional auxiliary target-position head | `src/vla/model/` |
| BC training | FSDP2 (also DDP and a DeepSpeed ZeRO-3 path used only in the scaling sweep), bf16, sync or async DCP checkpoints | `src/vla/train/` |
| Evaluation | Batched closed-loop rollouts, Wilson intervals, success relative to the expert ceiling, smoothness and safety metrics; a first-frame grounding probe | `src/vla/eval/`, `scripts/probe_grounding.py` |
| RL | Decoupled actor-learner GRPO with truncated importance sampling; Flow-GRPO for the flow head | `src/vla/rl/` |
| Inference | Eager, static-KV-cache and CUDA-graph decode; latency benchmark with a roofline analysis | `src/vla/infer/` |
| Remote tooling | Setup, rsync, detached launch with exit-code capture, GPU snapshots, quiet-window waiting | `scripts/` |

## Architecture

```mermaid
flowchart LR
    subgraph Sim["MuJoCo tabletop (DR: camera, light, texture, friction, mass, kp)"]
        ENV[TabletopEnv] --> OBS["128x128 image + proprio(11)"]
    end
    OBS --> VIT[SigLIP-base] --> PROJ[MLP projector]
    PROP[proprio MLP] --> SEQ
    TXT["prompt 'In: {instruction}\nOut:'"] --> SEQ
    PROJ --> SEQ["sequence: prompt (left-padded) | 196 image tokens | 1 proprio"]
    SEQ --> LLM[Qwen2.5-0.5B]
    LLM -->|"aux, training only"| GH["ground head: target cube xy"]
    LLM -->|discrete head| TOK["16 action tokens, 256 bins\n(sliced 256-row head)"]
    LLM -->|flow head| DIT["DiT, adaLN on t, cross-attention\n10 Euler steps (ODE) / K SDE steps (RL)"]
    TOK --> ACT["action chunk: 4 steps x (dx, dy, dz, gripper)"]
    DIT --> ACT
    ACT --> ENV
```

Decoupled RL topology (one `torchrun` world):

```mermaid
flowchart LR
    R["rollout ranks\nstatic KV cache + CUDA graph engine\nsubprocess env workers (EGL)"] -- "batches: obs, tokens or latents,\nbehaviour log-prob, version\n(atomic files in /dev/shm)" --> L
    L["learner ranks (FSDP2)\nPPO-clip x truncated IS\nk3 KL / Gaussian KL to BC reference"] -- "bucketed broadcast, in-place copy_\n(CUDA graphs stay valid)" --> R
```

Design notes, in the order a reader is likely to ask:

* **Why a custom simulator.** It gives full control over randomization and the held-out splits and renders headless
  with EGL. The cost is that it is not a published benchmark, so success numbers are only comparable inside this repo.
* **Action tokens.** Actions in [-1, 1] are quantised into 256 bins and appended to the Qwen vocabulary. A chunk of
  4 steps is 16 tokens. The loss is cross-entropy over the 256 action rows only, which equals a logit mask to the
  action vocabulary without materialising a full-vocabulary logits tensor. At inference the head is sliced to those
  256 rows (0.23M parameters instead of about 136M).
* **Prefix order: text first.** The prompt comes before the image tokens, so under causal attention the image tokens
  are encoded already knowing which cube was asked for. The original order (image, proprio, prompt) is still the
  default in `VLAConfig.prefix_order` and in `configs/model_0p5b.yaml`; `configs/model_0p5b_txt.yaml` selects
  `txt_first`. Why this matters is the subject of the next section.
* **Flow head.** A 6-layer DiT (d=512, about 36M parameters) cross-attends to the backbone's hidden states over the
  prompt/image prefix; rectified flow with a Beta(1.5, 1) timestep density; 10 Euler steps at evaluation.
* **Splits.** `train` randomizes within the ranges in `DRConfig`. `heldout` uses disjoint light directions, a wider kp
  range, camera, friction and mass ranges that lie outside the training ranges, and procedural materials not used in
  training. `heldout_photo` adds real photo textures (DTD classes never used in training) on the table and floor.
  Success is reported both absolute and relative to the scripted expert's success on the same split, because the
  expert itself does not reach 100% on the harder splits.
* **Decoupled RL.** The rollout engine and the learner run in different processes. With staleness S=1 the rollout
  for batch k+1 overlaps the learner update on batch k, so batches are at most one policy version old. The behaviour
  log-probability is recorded at generation time and the update uses a truncated importance weight
  `min(pi_old / mu, 2)`, which corrects both the staleness and the numerical gap between the inference engine and the
  training graph. That gap is logged as `engine_mismatch`.
* **Flow-GRPO.** For the flow head the probability-flow ODE is replaced by the marginal-preserving SDE of the
  Flow-GRPO paper (`sigma_t = a * sqrt(t / (1 - t))`, Euler-Maruyama), so each denoising step is Gaussian with a
  closed-form log-probability and KL. This stage is gated on the BC results (see `docs/PLAN_v2.md`) and may not be reported.

## Colour grounding: diagnosis and fix

The first behaviour-cloning models were weak on `reach` and `lift` (15% and 12.5% train success against an expert at
100%), but scored about 50% on `push`. All three tasks name a cube by colour; `push` always starts with the same motion.

**Diagnosis.** A first-frame probe compares the policy's predicted xy motion with the expert's heading towards the
named cube, then swaps the colour word in the instruction and checks again.
* On training episodes the first-frame cosine was 0.39 (reach) and 0.48 (lift).
* Swapping the colour word changed the actions by 0.003 to 0.03 and never changed which cube the arm went to.
* The policy had learned "go to some cube" and ignored the colour word.

Three controls ruled out the data:
* A small CNN given the pixels and a one-hot colour located the named cube to 1.3 cm under full randomization
  (0.3 cm without camera randomization).
* The same VLA, trained only to predict that position, stayed at chance (13 cm) with both datasets.
* So the labels, images and camera randomization were fine; the VLA pathway was not using the word.

**What did not work.**
* An auxiliary loss that predicts the target cube's xy from the last prefix position: its output moved 0.004 across
  colour words and 0.60 across images.
* Adding counterfactual colour queries (swap the word, relabel with the other cube's xy, drop that row from the
  action loss): still at chance.

**What worked: putting the text before the image.**
* With the image first, the image tokens are encoded without the question and a single read-out position has to
  recover the target from them.
* With the text first, every image token attends to the colour word.
* In the position-only diagnostic this broke the plateau at about step 700 (2.3 cm). The image-first model at 5x the
  learning rate stayed stuck.

The final recipe (`configs/bc_ground_disc.yaml`) combines:
* the text-first prefix
* the auxiliary position loss (weight 0.2)
* counterfactual colour queries (p = 0.25)
* 3x oversampling of the first four frames of each episode

The first-frame probe on validation episodes:

| Discrete head | reach cos to expert | lift cos to expert | after colour swap: cos to the new cube / old cube | target position error |
|---|---|---|---|---|
| Before (`bc_disc`, image first, no aux loss) | 0.48 | 0.40 | 0.37 / 0.46 (reach): the swap barely moves the action | not measured |
| Flow head, warm-started (`bc_gflow6`, step 500) | 0.96 | 0.94 | 0.94 / 0.03 (reach), 0.95 / 0.03 (lift) | 1.5 cm |
| After (`bc_gdisc3`, step 1,000, discrete) | 0.96 | 0.97 | 0.95 / 0.04 (reach), 0.97 / 0.06 (lift) | 1.3 cm |

The probe gate requires a cosine of at least 0.9 on reach and lift and a colour swap that redirects the arm.

**Flow head.** Training the flow head from scratch with the same recipe did not escape the plateau by step 500,
even with the auxiliary weight raised to 0.2. Its gradient norm in the first 200 steps was 30 to 55 (clipped to 1.0),
against 5 to 20 for the discrete head. Warm-starting the flow model from the grounded discrete backbone
(`init_from`, every weight except the freshly initialised DiT) passes the gate at step 500, as the table shows.

## Setup

Local (macOS or Linux, Python 3.11, CPU is enough for the tests and smoke runs; never train on a laptop):

```bash
uv sync                                  # core + dev dependencies; the [gpu] extra is not installed
bash scripts/fetch_dtd.sh                # photo textures for the train / heldout_photo splits (~35 MB subset fallback)
uv run pytest tests -q
```

Server (GPU box, no root; see `.env.example` for the variables):

```bash
cp .env.example .env                     # set VLA_DATA, VLA_RUNS, VLA_CKPT, VLA_LOGS, CUDA_VISIBLE_DEVICES, UV_* paths
bash scripts/remote/sync.sh <host> [remote_dir]      # rsync the code (never touches .env, data, runs)
ssh <host> 'cd vla && bash scripts/remote/setup.sh'  # uv, Python 3.11, torch+cu130, EGL probe, GPU checks
bash scripts/remote/launch.sh <name> "<command>"     # detached tmux run, EXIT code and logs preserved
```

`setup.sh` renders once per MuJoCo EGL device id in a subprocess and keeps only ids whose `GL_RENDERER` names the
A100, so a software or display-card renderer cannot silently produce data. FlashAttention is not built from source
(no prebuilt wheel matched this torch/CUDA combination, and a source build was judged not worth the time); the
attention path is PyTorch SDPA.

## Reproducing each result

Every command below is run from the repository root with `.env` sourced.

1. **Expert ceiling** (`results/expert_ceiling.json`)

   ```bash
   uv run python -m vla.sim.expert --episodes 200 --split all --out results/expert_ceiling.json
   ```

2. **Demonstrations** (about 9,000 successful episodes; one shard per 50 episodes)

   ```bash
   uv run python -m vla.data.generate --config configs/data.yaml --num-workers 32
   # ablation datasets fix one randomization group at nominal values:
   uv run python -m vla.data.generate --config configs/data.yaml --num-workers 32 dr_nominal=[cam]  out_dir=$VLA_DATA/demos_nocam
   ```

3. **Behaviour cloning** (discrete and flow heads, same backbone initialisation, data and seed)

   ```bash
   # baseline (image-first, no grounding loss)
   torchrun --nproc_per_node=4 -m vla.train.bc --config configs/bc.yaml name=bc_disc
   torchrun --nproc_per_node=4 -m vla.train.bc --config configs/bc.yaml name=bc_flow head=flow
   # grounded recipe (text-first prefix, aux position loss, colour swaps, early-frame oversampling)
   torchrun --nproc_per_node=4 -m vla.train.bc --config configs/bc_ground_disc.yaml
   # flow head warm-started from the grounded discrete model
   torchrun --nproc_per_node=4 -m vla.train.bc --config configs/bc_ground_flow.yaml ground_weight=0.2 \
       init_from=$VLA_CKPT/bc_gdisc3/best.pt
   # first-frame grounding probe (pass/fail gate)
   uv run python scripts/probe_grounding.py --ckpt $VLA_RUNS/bc_gdisc3/best.pt --head discrete \
       --data $VLA_DATA/demos --model-config configs/model_0p5b_txt.yaml --n-train 300 --out results/probe_bc_gdisc3_best.json
   ```

   `stop_at=<step>` saves and exits early so a run can be gated and then resumed on the same schedule.
   `keep_best=true` keeps the checkpoint with the lowest validation action loss as `best.pt`.

   Checkpoints go to `$VLA_CKPT/<name>/ckpt` (latest two kept, DCP shards, optimizer state included);
   `$VLA_RUNS/<name>/final.pt` is weights-only bf16. A run killed at any point resumes with `--resume`.

4. **Closed-loop evaluation** (`results/eval_*.json`)

   ```bash
   uv run python -m vla.eval.rollout_eval --policy vla --ckpt $VLA_RUNS/bc_disc/final.pt \
       --model-config configs/model_0p5b.yaml --head discrete --data-dir $VLA_DATA/demos \
       --splits train,heldout,heldout_photo --tasks reach,lift,push --episodes 200 --n-envs 32 --bf16 --quiet-wait \
       --out results/eval_bc_disc.json
   # grounded model: pass the text-first model config and keep the ground head so the checkpoint loads strictly
   uv run python -m vla.eval.rollout_eval --policy vla --ckpt $VLA_RUNS/bc_gdisc3/best.pt \
       --model-config configs/model_0p5b_txt.yaml --override ground_weight=0.2 --head discrete \
       --data-dir $VLA_DATA/demos --splits train,heldout,heldout_photo --tasks reach,lift,push \
       --episodes 200 --n-envs 32 --bf16 --out results/eval_bc_gdisc3.json
   uv run python -m vla.eval.rollout_eval --policy expert --out results/eval_expert.json
   ```

5. **RL post-training**

   ```bash
   torchrun --nproc_per_node=2 -m vla.rl.grpo --config configs/rl.yaml name=rl_disc_seed0 seed=0 \
       init_ckpt=$VLA_RUNS/bc_disc/final.pt data_dir=$VLA_DATA/demos staleness=1
   ```

   `staleness=0` is the serial on-policy baseline. `head=flow` selects Flow-GRPO.

6. **Training-system measurements**

   ```bash
   uv run python scripts/bench_bc_step.py --sweep --gpus 0,1,2,4 --n-gpu 4 --quiet-wait --out results/bench_bc_step.json
   bash scripts/scaling_sweep.sh --out results/scaling.json
   uv run python scripts/profile_step.py --n-gpu 4
   torchrun --nproc_per_node=4 scripts/ckpt_stall_study.py --out results/ckpt_stall.json
   ```

7. **Inference latency** (`results/latency.json`)

   ```bash
   uv run python -m vla.infer.benchmark --ckpt $VLA_RUNS/bc_disc/final.pt --flow-ckpt $VLA_RUNS/bc_flow/final.pt \
       --data-dir $VLA_DATA/demos --quiet-wait --trace results/traces --out results/latency.json
   ```

8. **Tables and plots**

   ```bash
   uv run python scripts/make_tables.py --results results --runs runs_pulled
   ```

## Results

Measured cells are filled in; `TBD` cells are not measured yet.

### Policy results (closed loop, 200 episodes per task and split, fixed seeds)

Success rate with 95% Wilson interval. The GPUs were shared during the BC evaluations; that affects timing only, not success.
The flow head samples noise, so a re-run of the same flow checkpoint moves by up to about 0.04 (inside the interval).

| Policy | Task | Train | Held-out | Held-out photo |
|---|---|---|---|---|
| Scripted expert (ceiling) | reach | 1.00 | 1.00 | 1.00 |
| | lift | 1.00 | 0.92 | 0.92 |
| | push | 0.985 | 0.725 | 0.725 |
| BC, discrete head (baseline) | reach | 0.150 [0.11, 0.21] | 0.125 [0.09, 0.18] | 0.135 [0.09, 0.19] |
| | lift | 0.125 [0.09, 0.18] | 0.085 [0.05, 0.13] | 0.085 [0.05, 0.13] |
| | push | 0.490 [0.42, 0.56] | 0.260 [0.20, 0.32] | 0.285 [0.23, 0.35] |
| BC, flow head (baseline) | reach | 0.185 [0.14, 0.24] | 0.120 [0.08, 0.17] | 0.125 [0.09, 0.18] |
| | lift | 0.165 [0.12, 0.22] | 0.060 [0.03, 0.10] | 0.065 [0.04, 0.11] |
| | push | 0.530 [0.46, 0.60] | 0.245 [0.19, 0.31] | 0.245 [0.19, 0.31] |
| **BC, discrete head, grounded** (`bc_gdisc3`) | reach | **0.855** [0.80, 0.90] | **0.450** [0.38, 0.52] | **0.410** [0.34, 0.48] |
| | lift | **0.610** [0.54, 0.67] | **0.290** [0.23, 0.36] | **0.285** [0.23, 0.35] |
| | push | 0.460 [0.39, 0.53] | 0.205 [0.15, 0.27] | 0.245 [0.19, 0.31] |
| BC, flow head, grounded (`bc_gflow6`) | all | TBD (training) | TBD | TBD |
| BC + RL (mean ± std over TBD seeds) | all | TBD | TBD | TBD |

What the table shows:
* Grounding lifts reach and lift success 3 to 6x on every split.
* Push is unchanged within the intervals: it never depended on the colour word.
* The gap between train and the held-out splits (0.855 to 0.45 on reach) is the target for RL post-training.

End-effector jerk (RMS, lower is smoother):
* Grounded discrete model: 6.5 to 11.6 on train and 13.0 to 15.7 on the held-out splits.
* Discrete baseline: 6.2 to 17.9, with most of its successes on push.
* Safety-violation counts and success relative to the ceiling are added when the result JSONs are regenerated.

Randomization ablations (short schedule, one seed, labelled as such) and the data-scaling points: TBD.

### Distributed training

| Measurement | Value |
|---|---|
| Step throughput, baseline vs tuned configuration (samples/s, MFU) | TBD |
| 1 / 2 / 4 GPU scaling efficiency, FSDP2 vs ZeRO-3, with and without activation checkpointing | TBD |
| Per-GPU peak memory, FSDP2 vs DDP | TBD |
| Async vs sync checkpoint stall, `/dev/shm` vs NFS | TBD |
| Kill-and-resume loss difference | TBD |

### Decoupled RL

| Measurement | Value |
|---|---|
| env-steps/s, S=0 vs S=1, per topology | TBD |
| Weight-sync latency | TBD |
| Mean `engine_mismatch`, importance-weight clip fraction | TBD |
| Return, success and KL curves | TBD |

### Inference latency (p50 ms per query of one 4-step chunk)

| Row | Batch 1 | Batch 64 | Token agreement vs row 2 | Measured / HBM floor |
|---|---|---|---|---|
| 1 fp32 eager, no cache | TBD | TBD | TBD | TBD |
| 2 bf16 eager, no cache | TBD | TBD | TBD | TBD |
| 3 bf16 + KV cache | TBD | TBD | TBD | TBD |
| 4 row 3 + sliced action head | TBD | TBD | TBD | TBD |
| 5 row 4 + static cache + CUDA graph | TBD | TBD | TBD | TBD |
| 6 flow head, 10 steps, CUDA graph | TBD | TBD | not applicable | TBD |

At batch 1 the decode step is memory-bound (about 1 FLOP per byte against an A100 ridge near 150 FLOP per byte).
The benchmark reports the bytes read per token, the HBM floor at 2.0 TB/s and the measured-to-floor ratio, so the
remaining gap can be attributed to launch overhead rather than guessed at.

## Measurement protocol on shared GPUs

The four A100s used here were shared with other users' jobs, some at 60-99% utilisation. Therefore:

* Processes are pinned with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES`, and memory is reported only
  as process-local `max_memory_allocated`, never from `nvidia-smi` totals.
* Every timing result embeds a GPU snapshot before and after the run (utilisation, foreign compute processes and their
  memory) and a `contended` flag. Contended numbers are kept and marked, not discarded.
* `scripts/quiet_wait.py` waits up to 20 minutes for a quiet window before a benchmark; on timeout the run goes ahead
  and is flagged.
* A-versus-B comparisons are interleaved (ABAB) and repeated, and the tables report median and range, not a single run.
  The `contended` flag is derived from utilisation and can under-report when foreign processes are not visible
  (for example inside a container).

## Status

Done:
* Simulator, randomization splits and data pipeline.
* Distributed BC with resumable checkpoints.
* Evaluation and the grounding probe.
* Grounded discrete BC model.
* The RL, inference and benchmark code paths exist and are covered by CPU tests (295 passing). Their GPU results are not
  measured yet.

In progress:
* Flow head, warm-started from the grounded discrete backbone: passed the 500-step gate; training to 1,200 steps,
  then the 200-episode evaluation.

Fixed along the way:
* An rsync `--delete` removed the first set of result JSONs on the server. They were regenerated from the same
  checkpoints, and `sync.sh` now protects `results/`.

Next:
* Checkpoint-stall study, memory table and scaling sweep.
* Randomization ablations on the grounded recipe.
* RL post-training (3 seeds plus a penalty ablation).
* Latency benchmark.
* Demo videos.

## Limitations

* The tasks are short, scripted-expert-solvable manipulation tasks in simulation. The held-out splits probe robustness
  to randomization, not sim-to-real transfer, and no real-robot result is claimed.
* The scripted expert is not perfect on the held-out splits (low friction and heavy cubes), so absolute held-out success
  is bounded by that ceiling; success relative to the ceiling is reported next to it.
* Only the 0.5B backbone is trained end to end. FSDP2 memory savings at larger scale are not demonstrated unless a
  1.5B measurement appears in the memory table.
* RL results are from a small number of seeds on one task family; confidence intervals are Wilson intervals over
  episodes within a seed, not across seeds.
* The CUDA-graph decode path depends on the installed `transformers` version; the engine falls back to the static-cache
  eager path and reports the reason if graph capture fails.
* The grounding fix was found and tuned on the `train` split's validation episodes. The held-out splits were only
  evaluated after the recipe was fixed, so they were not used to choose it.
* The grounding recipe adds an auxiliary label (the target cube's position) that a real-robot dataset would need to
  provide, for example from a detector or from the end position of the demonstration.
* Flow-GRPO uses a coarse K-step grid whose first step starts at `t = 1 - 1/(2K)`; this is an approximation of the
  paper's setup and is noted where its results are reported.

## Repository layout

```
configs/    model, data, BC and RL configs (YAML, key=value overrides on the command line)
scripts/    remote setup/sync/launch, sweeps, profiling, benchmarks, make_tables.py
src/vla/    sim/  data/  model/  train/  rl/  eval/  infer/  utils/
tests/      CPU tests (tiny model, gloo): env and DR, data, model, training and resume, eval, RL loop, tables
results/    small committed artifacts (JSON, tables, plots, demo videos)
docs/       the build plans
```
