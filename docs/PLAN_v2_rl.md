# PLAN v2, P6 amendment: RL recipe after round 1 (planner, 2026-11-03)

## Round 1 (configs/rl.yaml, 100 iters, 8x8 eps/iter, 2 epochs x mb64 = ~38 Adam steps/iter, beta .02, token-mean loss)
T=1 rollout success .67 -> ~.35, return 5.5 -> -7.6, KL -> ~1.9-2.0, jerk ~2x. Greedy eval mixed (reach/lift up, push down).
Penalty-off run drifts the same. Infra checks clean (engine_mismatch .009, advantage alignment verified).

## Diagnosis
1. Length-biased negative gradient (main suspect): token-mean loss; failed episodes run full horizon (25 chunks, 400 tokens)
   vs successes ~8 chunks (130 tokens), so negative-advantage tokens carry ~3x the gradient mass. Pushing down sampled tokens
   without matching pull-up flattens the 256-bin distributions: entropy up, KL up, argmax flips (jerk up), T=1 success down.
2. Noise amplification: (R-mean)/(std+1e-6) on all-success / all-fail groups turns tiny penalty differences into unit advantages.
3. Step count: ~38 clipped Adam steps per iteration; beta .02 cannot counter a unit-scale advantage term.
Ruled out: bf16 rounding of updates (master params fp32).

## Builder changes (rl/algo.py, rl/learner.py, configs/rl_pilot_{a,b,c}.yaml)
- `opt_steps_per_iter: int`: 1 epoch, minibatch = ceil(n_samples / k).
- `loss_norm: token | episode`: episode weights each token by 1/(its episode's token count), normalised by the sum of weights
  over the whole minibatch (computed before micro-batching); same weights on the KL term.
- `adv_std_floor`: adv = (R-mean)/max(std, floor).
- `kl_target`, `beta_adapt`: per iteration, KL > 1.5*target -> beta *= 1.5; KL < target/1.5 -> beta /= 1.5; clamp [0.005, 1.0].
- Metrics: `entropy` (token entropy over 256 bins), `frac_uniform_groups`, `neg_weight_frac`, `beta`.
- Tester (CPU): episode norm gives equal total weight per episode regardless of length; floor applies; beta rule; uniform-logit
  entropy = ln 256; `loss_norm: token` reproduces the current loss exactly.

## Pilots (incremental ladder; seed 0, 35 iters, 64 eps/iter, T=1, lr 2e-6, engine graph, ~0.6-0.7 GPU-h each)
| Pilot | Change |
|---|---|
| A | opt_steps_per_iter 4 |
| B | A + loss_norm episode + adv_std_floor 1.0 |
| C | B + beta_adapt, kl_target 0.05/token, beta0 0.05 |
Baseline = existing rl_disc_s0, iters 0-35 (no new compute).
Primary metric: dT1 = mean T=1 success over iters 26-35 minus mean over iters 0-5 (~+-0.03 SE).
Eligible at iter 35: KL <= 0.3, entropy <= 1.1x iter-0, dT1 >= 0. Winner: highest dT1; tie-break greedy train lift+push 200 eps
and EE jerk RMS <= 1.2x BC. Proceed to final only if winner dT1 >= +0.05 and greedy push >= BC - 0.03.
Fallback if none qualifies: D = C + T 0.7 (rollout and learner) + lr 1e-6 (0.7 GPU-h), then decide.

## Final
Winner config, seeds 0/1/2 + penalty ablation (jerk_act = jerk_ee = 0) seed 0. 60 iters (80 if dT1 still rising at 35).
Report the final checkpoint (no selection on eval). Eval greedy and T=1, 200 eps x 3 tasks x 3 splits per seed; mean +- std
over seeds, per-seed Wilson CIs, KL/entropy curves, jerk RMS table.
Budget: pilots 2.1 + final 4.8 + evals ~1 = ~8 GPU-h of the ~10 left; 2 spare (fallback D or 80 iters).

## If RL still does not beat BC
Negative result with evidence: BC vs RL table (greedy and T=1, 3 seeds), KL/entropy/success curves, the ladder, the
length-bias analysis (neg_weight_frac vs drift). Report the reach gain as observed, not caused (reach not trained).
Resume: claim the async RL infra, throughput and the diagnosis, not a success gain.
P6b Flow-GRPO runs only if final shows dT1 >= +0.05 averaged over 3 seeds with bounded KL; else "not run (gate failed)".

## Addendum (lead, 2026-11-08): the dT1 metric was confounded

`dT1` compared in-training success at iterations 70-79 with iterations 0-5 of the same run. Rollout env seeds are
`seed0 + k*8 + j` for iteration k, identical in every run, so the two windows contain different scenes. A control run
with `lr=0.0` on the same schedule shows a naive dT1 of +0.026 (discrete) and +0.079 (flow) with no learning.

Corrected metric: per-iteration success minus the `lr=0` control at the same iteration, averaged over iterations 40-79.
Final discrete seeds: +0.028 / +0.022 / +0.040 (mean +0.030); the P6b gate (+0.05) is not met on the corrected metric.
Flow-GRPO had already run on the confounded reading (+0.083) and is reported as exploratory.
Sampled (T=1) evaluation on 200 fixed episodes agrees: RL minus BC is -0.052 to +0.017 per cell, within noise.
Future gates should use the paired control (or fixed-seed T=1 evaluation), never a within-run late-vs-early window.
