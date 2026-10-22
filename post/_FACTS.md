# Fact sheet (not published)

Every number the post might use, with the file it comes from. Paths are relative to the repo root.
If you quote a number that isn't here, add it here first with its source.

## Verifier

| Fact | Value | Source |
|---|---|---|
| Starter kernels that pass the full bench | 3 of 9 (H100 and L4) | `results/numbers.json` → `stage1.starters_pass_full` |
| Starters that pass quick mode | 5 of 9 on H100, 4 of 9 on L4 | `results/numbers.json` → `stage1.starters_pass_quick` |
| Why the gap | `--quick` skips the numerical-stability checks | `CLAUDE.md`, "Findings that changed the code" |
| Worst timing noise (coefficient of variation) | 1.89% | `results/numbers.json` → `stage1.max_cv_pct` |
| RMSNorm "2.9× kernel" | the 2.9× is the starter's (fusion vs 6-kernel eager PyTorch) | `results/v3/rmsnorm_remeasure.md` line 7 |
| GRPO's RMSNorm edit vs the starter | 1.004× at the primary shape; geomean 0.999× over 32 configs | `results/v3/rmsnorm_remeasure.md` line 9 |
| What the edit breaks | fp32→fp16 cast returns all-zero rows when row RMS > 4 at N=4096 | `results/v3/rmsnorm_remeasure.md` line 13 |
| Why it still passed | the bench's reference overflows the same way; its large-input test overflows too | `results/v3/rmsnorm_remeasure.md` lines 7, 13 |
| Previously passing kernels that survive bench v2 | 346 of 362 (95.6%) | `results/v3/bench_v2.md` lines 9–10 |
| Previously passing kernels with a robust speedup over the starter under bench v2 | 0; criterion: >1.05× geomean, CI-significant at every timing shape, v2 PASS | `results/v3/bench_v2.md` line 11 |
| KernelBench L1 problems the harness could score | 55/100 before the fix, 100/100 after (kb v2) | `results/v3/kb_v2.md` lines 1, 22 |
| Why 45 were broken | weights never seeded (all 35 conv problems), CPU input generation inside a 30 s timeout | `CLAUDE.md`, "Findings that changed the code" |

## Feedback

Base model, no training, 72 episodes per arm. fbv1 = tail of the bench log; fbv2 = the failing check's lines.

| Fact | fbv1 | fbv2 | Source |
|---|---|---|---|
| Episode pass rate | 30.6% | 34.7% | `results/numbers.json` → `agent_eval.<…fbv1/fbv2>.pass_rate` |
| Mean reward | 0.097 | 0.133 | same, `reward_mean` |
| Turns that crashed | 128 | 56 | same, `turn_outcomes.CRASH` |

The pass-rate gap is 3 episodes out of 72. Treat it as noise; the crash count is the clearer signal.

From `results/v3/sft_plan.md` lines 8–9 (base model):
- 31.3% of failing turns are Triton compile errors.
- After a compile failure, the next turn passes 1.2% of the time.
- 54.5% of consecutive failures repeat the same error.

## GRPO

| Fact | Value | Source |
|---|---|---|
| GRPO v1 steps | 18 | `results/numbers.json` → `grpo.grpo_v1.steps` |
| v1 reward, first 3 vs last 3 steps | 0.098 → 0.115 | `grpo.grpo_v1.reward_first3`, `reward_last3` |
| v1 pass rate, first 3 vs last 3 steps | 29.2% → 25.0% | `grpo.grpo_v1.pass_first3`, `pass_last3` |
| Groups with zero advantage (mean) | 65.4% | `grpo.grpo_v1.zero_adv_mean` |
| Share of |advantage| from noise-level groups (std < 0.02), as trained | 78.9% | `results/extra_numbers.json` → `dr_grpo_whatif.grpo_v1.abs_adv_mass_share_from_noise_groups_pct` |
| Same, without std division (Dr. GRPO) | 6.2% | same key, `dr_grpo_no_std` |
| Same, with a 0.02 std floor | 0.0% | same key, `grpo_std_floor_0.02` |

## Reward v2

| Fact | Value | Source |
|---|---|---|
| v2 steps | 10 | `results/numbers.json` → `grpo.grpo_v2.steps` |
| v2 reward, first 3 vs last 3 steps | 0.371 → 0.399 | `grpo.grpo_v2.reward_v2_first3`, `reward_v2_last3` |
| Groups with zero advantage (mean) | 37.8% (v1: 65.4%) | `grpo.grpo_v2.zero_adv_mean` |
| Kernel types ever solved | cross_entropy, reduce, rmsnorm, softmax (same as before training) | `grpo.grpo_v2.kernels_ever_passed` |
| v2 pass rate, first 3 vs last 3 steps | 31.5% → 31.9% (0.3148 → 0.3194) | `results/numbers.json` → `grpo.grpo_v2.pass_first3`, `pass_last3` |
| v2 advantage mass from noise-level groups, without the threshold | 17.4% (as trained with threshold: 0.0%) | `results/extra_numbers.json` → `dr_grpo_whatif.grpo_v2.abs_adv_mass_share_from_noise_groups_pct` |
| Groups across the run: total / update-bearing / exact ties / suppressed small variance | 90 / 56 / 21 / 13 | `results/extra_numbers.json` → `dr_grpo_whatif.grpo_v2` (`groups_total`, `zero_variance_groups`, `noise_level_groups_0_lt_std_lt_0.02`; update-bearing = 90 − 21 − 13) |
| Reward definition | `modal_app/reward_v2.py` | code |

## SFT

| Fact | Value | Source |
|---|---|---|
| Training examples | 3,022 train, 60 dev | `results/v3/sft/mix_report.json` → `splits` |
| Repair (fix-a-broken-kernel) share | 15% | same, `splits.train.repair_share` |
| Train sources | Dr. Kernel 2,707 · in-repo 244 · KernelBook 71 | same, `splits.train.by_source` |
| KernelBook rows that passed GPU verification | 69 of 1,029 | same, `subsets.kernelbook_sel.verdicts` |
| Removed as KernelBench L1 overlap | 34 (19 + 15) | same, `drops.decontam_kb_l1` |
| Training | LoRA, 378 steps (2 epochs) | `results/v3/sft_runs/sft_v1/metrics.jsonl` |
| Dev loss | 0.601 → 0.302 | same, first and last `dev_loss` |
| Validation pass@1 (62 problems) | 0.383 → 0.508 | `results/v3/phase3.json` |
| Compile errors per turn | 8.1% → 1.9% | `results/v3/phase3.json` |
| Repeat-same-error rate on validation | 0.338 → 0.283 | `results/v3/phase3.md` → validation table |
| KernelBench L1 correct + launches Triton | 0/100 → 20/100 | `results/v3/phase3.json` → `kernelbench` |
| KernelBench L1 correct + launches Triton + faster than eager PyTorch | 0/100 → 10/100 | `results/v3/phase3.md` → `fast_1_triton` |
| Geomean speedup vs starter (solving validation episodes) | 0.950× → 1.003× | `results/v3/phase3.md` → validation table |
| …on ops not seen in SFT | 0/59 → 7/59 | `results/v3/phase3.md` line 64 |
| Solved episodes beating the starter by > 5% | 2 of 126 | `results/v3/phase3.json` → `groups.sft` |
| Median solved speed vs starter | 1.00× | same, `solved_vs_starter_median` |
| Validation problems with mixed pass/fail groups | 59.7% (RL would have signal here) | same, `mixed_pass_fail` |

Data licence caveat: Dr. Kernel trajectories are distilled from GPT-5 (see `mix_report.json` → `license.drkernel`). Mention it if you discuss the data.

Evaluation budgets: validation pass@1 is per sampled episode, with up to 4 turns; KernelBench uses n=1 and up to 3 turns (`results/v3/phase3.md`, section introductions). The comparison of advantages without std division is an offline recomputation, not a trained Dr. GRPO run (`results/extra_numbers.json` → `dr_grpo_whatif`).

## Cost

| Fact | Value | Source |
|---|---|---|
| GRPO v1 step time: benchmarking / generation / update | 60.5% / 32.8% / 6.7% | `results/extra_numbers.json` → `time_breakdown.grpo_v1.share_pct` |
| Trainer GPU idle | 93.3% | same, `trainer_gpu_idle_pct` |
| GRPO v2 step time: rollout collection / policy update | 85.4% / 14.6%; generation and benchmarking not separately logged | `results/extra_numbers.json` → `time_breakdown.grpo_v2.rollout_vs_update_pct` |
| GRPO v1 cost (estimated from wall time) | $18.84, 2.38 h | `results/numbers.json` → `grpo.grpo_v1.est_cost_usd`, `wall_h` |
| GRPO v2 cost (estimated) | $11.15 | `grpo.grpo_v2.est_cost_usd` |
| Whole project (estimated) | ≈ $95 | `CLAUDE.md`, Status |
| Phase 0 (actual Modal billing) | ≈ $5.3 | `results/v3/phase0.md` § 5 |
| Bench v2 rescore | ~2× the summed job wall time in billed container time | `results/v3/bench_v2.md` line 146 |

The GRPO and $95 figures are estimates, not bills. Say so if you quote them.
