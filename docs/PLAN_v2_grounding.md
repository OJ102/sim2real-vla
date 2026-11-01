# PLAN v2 delta — colour-grounding fix for BC

Status: APPROVED by user 2026-11-01 (direction); plan by planner, reviewed by lead. Budget ≤ 3.8 GPU-h from the 5 h reserve. Scope: items 1–3 + probe + gate + tests. Not in scope: colour-swap negatives, instruction shuffling.

## Evidence (lead, server, bc_disc / bc_flow 4-epoch final.pt)
- Closed-loop success, train split (disc / flow): reach .15/.18, lift .125/.13, push .49/.545. Expert ceiling 1.0 / 1.0 / .985.
- First-frame probe: cosine(pred xy, expert heading) reach .39 train / .47 val, lift .48 / .41, push .98 / .98. Swapping the colour word leaves cosine unchanged (discrete action delta 0.003–0.03). Same for flow head.
- Cubes clearly visible at 128 px. Conclusion: shortcut learning; colour matters in ~3 of 21–32 frames per demo.

## Root cause
Causal attention with prefix order `[img, proprio, txt]`: image tokens never see the colour word; only text positions can compose colour with image, and the loss gives little pressure to do so. The aux loss is placed at the last prefix token (position P−1), the only position seeing both colour word and image; in discrete mode it predicts action token 1, in flow mode the DiT cross-attends to it. Reordering to `[txt, img]` is the fallback, not in scope.

## Decisions
- **Label:** target cube world xy at frame t, normalised by train per-dim mean/std (`target_xy_mean/std` in stats.json). Matches world-frame EE-delta actions; camera-DR invariant; per-frame since the cube moves in lift/push. Image-plane uv rejected (per-frame projection under jittered camera, ties features to camera).
- **Head:** `ground_head = Linear(d_llm,256) → GELU → Linear(256,2)` on `h[:, P-1]`. Masked MSE; rows without label have `ground_mask=0`; divide by `max(mask.sum(),1)`.
- **Weight:** total = main + λ_g·ground. Discrete λ_g=0.2 (CE≈1, normalised MSE≈1 at start). Flow λ_g=0.01 (flow MSE≈0.04; same ~0.2–0.25 ratio). No adaptive balancing.
- **Oversampling:** frames with t<4 get integer weight 3 (~30% of samples early vs 14% now), by integer repetition in an expanded index list — deterministic and resumable. One epoch = one pass over the expanded list. Val not reweighted.
- **Shards:** add per-frame `cube_pos f32[N,3,3]` (aligned with `images[row]`) and per-episode `target_idx i64[E]`. Dataset derives target xy.
- **Data to regenerate:** train demos + nocam/nophys/novis ablation sets (~70 s each). P5 ablations use the same recipe. Heldout demos unused by BC: skip.
- **Schedule:** 1200 total steps, 200 warmup, cosine, val every 250. Keep `best.pt` by val main_loss (exported like final.pt); eval uses best.pt. Gate = first 500 steps of the same run: stop at 500, probe, resume from DCP ckpt with the same total so the cosine schedule continues.

## Files and steps (builder)
1. `src/vla/data/generate.py`: in `run_job` append `env.privileged()["cube_pos"]` alongside each appended image (reset frame and each non-terminal step). Write `cube_pos` and `target_idx = tidx`. `compute_stats` adds `target_xy_mean/std`; warn and skip when the field is missing.
2. `src/vla/data/dataset.py`: `self.frame_t` array; read `target_idx` per episode in init; `__getitem__` returns `ground_xy` (normalised f32[2]) and `ground_mask` (1.0, or 0.0 if the shard lacks `cube_pos`); `make_collate` stacks both.
3. `src/vla/data/sampler.py`: `ResumableDistributedSampler(..., repeats: np.ndarray | None = None)`. Base list = `concat(arange(n), *[idx where repeats ≥ k] for k=2..max)`; permutation seeded exactly as now; `n` in `state_dict` = expanded length. `bc.py` builds `repeats = where(ds.frame_t < early_k, early_w, 1)`.
4. `src/vla/model/vla.py`: `VLAConfig.ground_weight: float = 0.0`; create `ground_head` only when > 0 (old checkpoints load strictly). `_discrete_logits` returns `(logits, h[:, P-1])`; update all callers incl. `token_logprobs`. Flow path takes `h[:, P-1]` from the same prefix pass. `forward` returns `loss` (combined), `main_loss`, `ground_loss`, `ground_err_cm` (de-normalised L2 ×100 over masked rows). RL and inference paths unchanged; `rl_tiny` and `rl_flow_tiny` must still pass.
5. `src/vla/train/bc.py`: fields `ground_weight`, `early_k=4`, `early_w=3`, `stop_at=0`, `keep_best=true`. Log `main_loss`, `ground_loss`. `validate()` adds `val_main_loss`, `val_ground_loss`, `val_ground_err_cm`, `val_ground_err_cm_t0` (t<early_k). `stop_at>0`: save DCP ckpt + export `stop_<step>.pt`, exit 0. `keep_best`: export `best.pt` when val main_loss improves.
6. `configs/bc_ground_disc.yaml`, `configs/bc_ground_flow.yaml`: copy of bc.yaml with `steps: 1200, warmup: 200, val_every: 250, early_k: 4, early_w: 3`, `ground_weight` 0.2 / 0.01, names `bc_gdisc` / `bc_gflow`.
7. `scripts/probe_grounding.py --ckpt X --data DIR --out results/probe_<run>.json --n-train 300 --device cuda` (any SubprocVecEnv use under `if __name__ == "__main__"`; this design needs no env). Reads t=0 frames from shards: 300 train eps + all val eps (every 20th). Raw proprio[:2] = ee xy; stored `cube_pos`. Discrete greedy; flow fixed seeded noise. pred = mean xy over chunk. Per task × part: `cos_expert` (vs mean of first 8 expert xy actions), `cos_geo` (vs unit(target_xy − ee_xy)); colour swap for both other colours (`the {c} cube` → `the {c2} cube`, image unchanged): `swap_cos_new` (vs heading to the c2 cube), `swap_cos_orig` (vs original heading), `swap_delta` (mean |a_swap − a|); `ground_err_cm_t0` if head exists. Must also run on CPU with the tiny model. Lead addition: also probe the old `bc_disc`/`bc_flow` final.pt as the README baseline (no ground head: skip `ground_err_cm_t0`; cube_pos read from regenerated shards).
8. Remote order:
   1. Regenerate the 4 datasets; verify new `actions`/`proprio` equal the old ones bit-for-bit on 3 sample shards.
   2. Data stays in /dev/shm.
   3. `bc_gdisc stop_at=500` → probe. 4. `bc_gflow stop_at=500` → probe.
   5. Each passing head: resume to 1200, probe `best.pt`, `rollout_eval` 200 eps/task × {train, heldout, heldout_photo}.
   6. `results/eval_bc_ground.json`.

## Gate (per head, step-500 probe, val part)
Pass requires all of, for reach and lift: `cos_expert ≥ 0.9`; `swap_cos_new ≥ 0.8`; `swap_cos_orig` ≥ 0.4 below unswapped `cos_expert`. `ground_err_cm_t0` reported (<3 cm expected, not gating). Push reported, not gated. Fail → no resume; report probe JSON + val_ground curves; next option for review = `[txt, img]` prefix order. Full-run report: closed-loop success vs expert ceiling, old vs new, both heads, whatever the result.

## GPU-h
| Item | GPU-h |
|---|---|
| Gate disc (500 steps ≈ 11 min × 4) | 0.7 |
| Gate flow | 0.6 |
| Resume to 1200 disc | 1.0 |
| Resume to 1200 flow | 0.9 |
| Probes + eval | 0.6 |
| **Total** | **3.8** |
If tight, cap flow at 1000 steps.

## Tests (tester-owned, CPU, tiny model)
- `test_ground_head`: ground_weight>0 → forward returns `ground_loss`, `ground_err_cm`; head out [B,2]; head grads non-zero, disc and flow. ground_weight=0 → no head; old tiny ckpt loads strictly.
- `test_ground_mask`: all mask 0 → ground_loss==0, no NaN, total==main. Mixed → only labelled rows contribute.
- `test_disc_logits_unchanged`: refactored `_discrete_logits` logits identical to before (fixed seed).
- `test_sampler_repeats`: same seed same sequence; expanded length n+Σ(w−1); mid-epoch state_dict → new sampler → identical remainder; early share ≈ w·k_frac; equal per-rank counts at world_size=2; `repeats=None` reproduces current behaviour exactly.
- `test_shard_fields`: 2-episode CPU generate writes `cube_pos` [N,3,3], `target_idx` [E]; `ds[i]["ground_xy"]` == normalised `cube_pos[row, target_idx, :2]`; legacy shard → `ground_mask=0`; stats.json has `target_xy_mean/std`.
- `test_probe_smoke`: probe runs on CPU with tiny ckpt + 2 shards; JSON has all keys per task and part.
- `test_stop_at`: `bc_tiny stop_at=3` exits 0, writes `stop_3.pt` + ckpt; resume reaches `steps` with unchanged lr schedule (lr at step 4 equals uninterrupted run).
