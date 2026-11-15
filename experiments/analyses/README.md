# Analyses

Analyses on the paper's 61-feature tables (3,175 informative queries, the 5×5 CV protocol of
the leak ladder, and the split used for Table 3). Inputs are read from `src/router/` and
`results/`; outputs go to `results/analyses/<name>/`. A script refuses to write into a
non-empty directory, so pass a new `--out-dir` to rerun. Scripts that write `rows.jsonl` accept
`--resume`, and the slow ones accept `--calibrate` to print a time estimate first.

Run from this directory.

| Script | What it computes | Output |
|---|---|---|
| `reproduce_checks.py` | Leak-gradient numbers reproduce from the committed tables; input hashes | `reproduce_checks/` |
| `build_noise_variants.py` | Noise-scale variants of the MRR@20 leak (written to `results/analyses/env_data/`, not tracked) | env files |
| `classifier_noise_sweep.py` | Best-arm accuracy across the noise variants | `classifier_noise_sweep/` |
| `leakcheck_noise_sweep.py` | Equality check and leakcheck on every variant | `leakcheck_noise_sweep/` |
| `random_noise_sweep.py` | Same sweep with independent per-query additive noise | `random_noise_sweep/` |
| `multiplicative_noise.py` | Same sweep with multiplicative noise | `multiplicative_noise/` |
| `scaled_logreg.py` | Logistic regression on standardised features | `scaled_logreg/` |
| `oracle_gap_capture.py` | Apparent capture of the oracle gap by leaky routers | `oracle_gap_capture/` |
| `leak_capture_summary.py` | Capture per rung of the leak ladder (Figure 1b) | `leak_capture_summary/` |
| `ladder_checks.py` | Figure 1 labels, MDE bracket, p-values, shared-index arms, leaked-router test | `ladder_checks/` |
| `plot_leak_ladder.py` | Figure 1 | `results/figures/fig12_leak_ladder.pdf` |
| `detection_power.py` | What each check flags against what the leak still buys (Figure 2) | `detection_power/` |
| `split_power.py` | Minimum detectable capture and capture intervals on the test split | `split_power/` |
| `rl_checkpoint_rerun.py` | Table 3's RL row with the best-validation checkpoint restored | `rl_checkpoint_rerun/` |
| `permutation_within_dataset.py` | Label shuffle within each dataset | `permutation_within_dataset/` |
| `paper_number_checks.py` | Derived numbers quoted in the paper text | `paper_number_checks/` |
| `positive_control.py` | Synthetic task with genuine signal, with and without a leak (Supplementary S1) | `positive_control/` |
| `detector_benchmark.py` | Detectors against planted leaks and a planted group confound (Table 4) | `detector_benchmark/` |
| `queryroute_audit.py` | leakcheck on QueryRoute's released features | `queryroute_audit/` |

Order: `build_noise_variants.py` must run before the sweeps, `oracle_gap_capture.py` before
`leak_capture_summary.py`, and both summaries before `plot_leak_ladder.py`. Unit tests for the
shared helpers are in `tests/test_noise_variants.py`.
