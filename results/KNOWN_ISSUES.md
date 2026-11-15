# Known issues in the result files

Defects in committed artifacts that we have documented rather than silently corrected. None
changes a number in the paper.

## 1. Pooled honest `gbt_std` in `learnability_analysis.json` does not reproduce

Two cross-validation standard deviations that describe different feature sets agree to
fifteen significant figures:

```
results/paper_claim_checks.json     claim_1 -> everything_but_leak (57 features)   std = 0.006914802645700829
results/learnability_analysis.json  pooled -> honest -> gbt_std (61 features)     std = 0.006914802645700836
```

Recomputing gives different values:

| configuration | seed 0 std | seed 7 std |
|---|---|---|
| honest, all 61 features | 0.006744571370997998 | 0.011086614173228350 |
| `everything_but_leak`, 57 features | 0.006914802645700829 | 0.008540951814389065 |

One of the two recorded figures was carried over rather than computed. The recomputed pooled
honest mean is also slightly different (0.5988031 against 0.5987402), so we cannot say which.
The paper uses neither standard deviation. Rerun `learnability_analysis.py` before using any
interval from this file.

## 2. `two_arm_ablation.json` computes the MDE with n_test = 137 on HotpotQA

`two_arm_ablation.py` derives the test size as `round(916 * 0.15) = 137`; the actual split
(as in `per_retriever_router.json`) is 641 / 137 / 138, so 137 is the validation size. At
n = 138 the three-arm MDE moves from 68.9% to 68.7%. The paper quotes the test-split MDE from
`results/analyses/split_power/summary.json` instead.

## 3. `per_retriever_router_base.json` duplicates `per_retriever_router.json`

The two files are byte-identical (sha256 `e282961eea87e766…`). Counting both inflates the
number of configurations; the paper applies Bonferroni over the 15 distinct ones
(α = 0.05/15 = 0.0033). The smallest p is 0.0293.

## 4. A traced number can still support an invalid comparison

`paper/audit_numbers.py` checks that every number in the paper appears in a results file. It
cannot check whether a comparison is controlled. Example: dropping Hybrid in the two-arm
ablation changes the lift from −3.3 / −3.1 / −6.4 to −2.0 / −2.8 / −5.0, but it also moves the
majority baseline (63.4 → 69.6, 61.7 → 69.0, 56.4 → 63.0) and the informative-query set, so the
change in lift does not show that the third arm was redundant.
