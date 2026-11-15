# Detector benchmark

Produced by `experiments/analyses/detector_benchmark.py` with its defaults: 10 seeds, n = 2000,
n_perm = 100. The numbers are in `summary.json`.

**Synthetic setup.** Each problem has 20 honest features, 3 of which carry genuine signal. One
planted column (or block) is added per scenario.

**Flag rate on the planted column:**

| Scenario | equality | Pearson > .35 | AUC scan (bioLeak-style) | GB permutation importance | leakcheck v1 verdict | leakcheck v1 any | leakcheck v2 any |
|---|---|---|---|---|---|---|---|
| none (not a leak) | 0 | 0 | 0 | 0.2 | 0 | 0 | 0 |
| exact | 1 | 1 | 0.2 | 1 | 1 | 1 | 1 |
| truncated (MRR@5-like) | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| monotone (rank) | 0 | 1 | 0.2 | 1 | 0 | 1 | 1 |
| value-keyed ±0.15 | 0 | 1 | 0 | 1 | 0 | 1 | 1 |
| order-scrambled recoding | 0 | 0.2 | 0 | 1 | 0 | 0.8 | **1** |
| independent noise ±0.15 | 0 | 1 | 0 | 1 | 0 | 1 | 1 |
| independent noise ±0.50 | 0 | 1 | 0 | 1 | 0 | 1 | 1 |
| confound (not a leak) | 0 | 0 | 0 | 1 | 0 | 0.9 | 1, labelled GROUP CONFOUND in 10/10 |
| split across 2 columns | 0 | 1 | 0 | 1 | 0 | 1 | 1 |
| PCA-mixed over 6 columns | 0 | 1 | 0 | 1 | 0 | 1 | 1 |

**Flag rate on the other columns** (mean over scenarios):

| | equality | Pearson | AUC scan | importance | v1 verdict | v1 any | v2 any |
|---|---|---|---|---|---|---|---|
| Genuine-signal columns | 0 | 0.67 | 0 | 0.85 | 0 | 0.74 | 0.91 |
| Pure-noise columns | 0 | 0 | 0 | 0.026 | 0 | 0 | 0 |

**Reading.**
- Verdicts (exact equality, v1 verdict) are never wrong, but they only catch the exact and
  truncated copies.
- Every detector that sees perturbed leaks also flags genuinely predictive features, because
  that is what a predictive feature is. Detection is therefore triage, settled by ablation.
- What v2 adds:
  - it catches the order-scrambled recoding in 10/10 seeds, against 8/10 for v1 and 2/10 for
    Pearson;
  - it names the group confound as one;
  - it raises no flags on pure-noise columns.
- The bioLeak-style AUC scan, at its 0.90 cut, almost never fires here.

**Caveat.** The synthetic leaks copy arm 0's full reward, so they are strong. Detection rates
near 1 reflect that, and should not be read as performance on subtler leaks.

**RL router data** (pooled 61-D, groups = dataset): which of the leak slots 22–23 each version
flags.

| Matrix | v1 | v2 |
|---|---|---|
| honest | none | none (slots 26–28 flagged; label-free overlap features with real signal) |
| value-keyed ±0.15 | 22, 23 | 22, 23 |
| value-keyed ±0.50 | **none** | **22, 23** |
| value-keyed ±1.00 | **none** | **22, 23** |
| independent ±0.15 / ±0.50 | 22, 23 | 22, 23 |

No real-data flag was labelled a group confound.
