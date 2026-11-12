# CRC v2 experiments (camera-ready)

These experiments use the paper's original setup: 61-D features, 3,175 informative queries, Table 2's 5×5 CV protocol and Table 3's split. They only re-check existing claims or back sentences the camera-ready changes. No new claims depend on them.

**Status: set up, not yet run.** No numbers exist yet.

Rules for every script:
- Inputs are only read: `src/router/*_env_data*.json` and `results/*`.
- Outputs go only under `results/crc_v2/<experiment>/`.
- Each writer refuses a non-empty output directory, and there is no overwrite flag. To re-run, pass a new `--out-dir`.
- `rows.jsonl` scripts append one line per fit. `--resume` continues an interrupted run in the same directory.
- `--calibrate` times a small piece and prints an estimated total before you commit CPU.

## Run order (Windows CMD, from `efficient-rag-pipeline\`)
```bat
:: 0. Unit tests for this setup (synthetic data, seconds)
python -m pytest tests\test_crc_v2_setup.py -q

:: R0. Existing numbers still reproduce? Run before anything else.
python experiments\crc_v2\r0_reproduce_checks.py

:: R1 build. Source checks only, then write the 18 variant env files per dataset
python experiments\crc_v2\r1_build_noise_variants.py --check-only
python experiments\crc_v2\r1_build_noise_variants.py

:: R1/R2 evaluate. Time it first, then the full sweep
python experiments\crc_v2\r1_classifier_sweep.py --calibrate
python experiments\crc_v2\r1_classifier_sweep.py

:: R3. leakcheck and equality check on every arm (model-free, minutes)
python experiments\crc_v2\r3_leakcheck_sweep.py

:: R4. Apparent oracle-gap capture on Table 3's split; add split seeds for O1
python experiments\crc_v2\r4_capture.py
python experiments\crc_v2\r4_capture.py --split-seeds 7 13 99 2024 --out-dir results\crc_v2\r4_capture_o1

:: RL. Table 3 RL row with the best-validation checkpoint restored (run after other CPU jobs finish)
python experiments\crc_v2\rl_restore_rerun.py --calibrate
python experiments\crc_v2\rl_restore_rerun.py
```

## R5: per-query random noise (added after validating R1)
R1's jitter is keyed on the value, so for MRR@20 (21 distinct values) it is a lossless
relabelling. R5 repeats the sweep with independent Uniform(−s, s) noise per query and arm,
with 3 noise draws × CV seeds 0–4. It needs no env files and its built-in check compares the
s = 0 arms with R1's summary.
```bat
python experiments\crc_v2\r5_random_noise_sweep.py --calibrate
python experiments\crc_v2\r5_random_noise_sweep.py
```
Output: `results/crc_v2/r5_random_noise/` (`rows.jsonl`, `leakcheck.jsonl`, `summary.json`).

## What each experiment is

| ID | Script | Question | Arms / grid | Output |
|---|---|---|---|---|
| R0 | `r0_reproduce_checks.py` | Do Table 2's numbers reproduce from the committed env files? | Re-runs `leak_gradient.py` into a new dir and diffs against `results/leak_gradient.json`. Hashes all env inputs. Lists untraced numbers via `audit_numbers.py --unchecked`. | `r0/r0_report.json` |
| R1 | `r1_build_noise_variants.py`, `r1_classifier_sweep.py` | How much of the leak's gain survives how much noise? | 2 families (`mrr20` = what the paper leaked, `rankreward` = the true reward) × noise half-width {0, .02, .05, .10, .15, .25, .50, 1.0}, plus clipped at 0.15. Also the 5 existing Table 2 rungs as anchors. GB and LR, CV seeds 0–4. | `r1_classifier_sweep/summary.json` (accuracy, seed range, lift, retention) |
| R2 | (row `rankreward_s000` in R1) | What does a true reward copy buy at 61-D? | — | same |
| R3 | `r3_leakcheck_sweep.py` | Which arms does an equality check or leakcheck flag? | Every R1 arm, per dataset and pooled. Records verdicts, suspicions, flags outside slots 22–25, exact-match rate, abs Spearman, single-feature argmax lift. | `r3_leakcheck_sweep/leakcheck_sweep.json` |
| R4 / O1 | `r4_capture.py` | What would a leaked router appear to capture of the oracle gap, in Table 3's terms? | Anchors + `mrr20_s000/015` + `rankreward_s000/015`. Canonical `compute_supervised_baselines` (tuned GB). Split 42 = Table 3; other seeds give O1. | `r4_capture/capture.json`, `capture_rows.csv` |
| RL | `rl_restore_rerun.py` | Table 3's RL row with best-validation weights restored | 15 seeds × 3 datasets, constants imported from `canonical_evaluation`. Reports final-epoch (old protocol) and restored test reward side by side. | `rl_restore/summary.json`, `per_query.csv` |

## Variant design (R1)
- Each variant starts from the honest env file. Only slots 22–24 change:
  - slot 22 = `jitter(v_bm25, "bm25", s)`;
  - slot 23 = `jitter(v_dense, "dense", s)`;
  - slot 24 = slot 23 − slot 22;
  - slot 25 keeps its honest value.
- `v` is `mrr_at_20` for the `mrr20` family and `rank_reward` for the `rankreward` family.
- The jitter is the repo's crc32 rule with the half-width as a parameter. At s = 0.15 it reproduces the existing `leak_noisy` slots 22–23 bit for bit. The build asserts this, and a unit test checks it.
- **Difference from the paper's rungs.** The `leaky` and `leak_noisy` rungs also leak through slot 25 (variance of BM25 MRR@k). These variants hold slot 25 honest, so that only the per-arm metric and its noise vary. The anchors are evaluated alongside, so this difference is measured, not assumed.
- **Fail-closed build checks.** The build stops if any of these fails:
  - query IDs and reward tables are identical across the honest, leaky and noisy files;
  - non-leak features are identical across those files;
  - `mrr == rank_reward`;
  - the leaky file's slots 22–23 equal `mrr_at_20`;
  - the jitter reproduces `leak_noisy` at s = 0.15.

## Built-in regression checks
- **R1:** CV seed 0 for honest / `leak_noisy` / `leaky` must give 59.9 / 72.7 / 73.1 (Table 2).
- **R4:** honest gradient boosting at split 42 must give 0.1642 / 0.2000 / 0.0753 (Table 3).
- **RL:** the final-epoch numbers should land near Table 3's 0.1512 / 0.1881 / 0.0683. They are not expected to match exactly, because the old loop also evaluated test every epoch.

If a check fails, stop and investigate before using any other result.

## Decision rules (fixed before running)
- **Abstract's "retains most of that advantage":** if `rankreward_s015` keeps less than 50% of `rankreward_s000`'s gradient-boosting gain, narrow the sentence to the MRR@20 setting and state the true-reward retention.
- **"An equality-based audit does not catch it":** keep it if R3 shows exact-match rate below the 0.25 threshold for every arm with s > 0, while R1 shows a gradient-boosting gain above 2 pp for that arm.
- **§6 "leakcheck raises the jittered leak as a suspicion":** keep it only if R3 shows a suspicion on `leak_noisy` and `mrr20_s015` in the pooled audit and on every dataset. Otherwise state where it fails.
- **RL row:** report the restored numbers in Table 3 and move the old row to a footnote. If restored differs from final-epoch by more than one seed SD, say so in the text.
- **Exploratory:** retention at other noise scales and the argmax-lift-vs-gain relationship are descriptive. Neither is a primary claim.

## Runtime (estimates, not measured; use `--calibrate`)
| Experiment | Estimate |
|---|---|
| R1 | 23 arms × 5 seeds = 115 GB+LR 5×5 CV fits on 3,175 × 61; roughly 2–4 h CPU |
| R3 | minutes |
| R4 | 9 arms × 3 datasets × split seeds, each with a 20-iteration randomized GB search; roughly 15–40 min per split seed |
| RL | 45 runs × 150 epochs; earlier notes suggest several hours |

R1 and R4 use `n_jobs=-1` and will slow any other CPU job, including the 25-D sweep in a parallel run.
