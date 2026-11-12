# Known issues in the committed result artifacts

Defects found by audit that are recorded rather than silently corrected. Each says what is
wrong, what depends on it, and what has to happen before it can.

---

## 1. `learnability_analysis.json` — pooled honest `gbt_std` is not reproducible

**Status:** open. No number in the paper depends on it today.

An external reviewer noticed that two cross-validation standard deviations in the artifacts
agree to fifteen significant figures while describing different feature sets:

```
results/paper_claim_checks.json   claim_1 → everything_but_leak (57 features)
    std = 0.006914802645700829
results/learnability_analysis.json  pooled → honest → gbt_std (61 features)
    std = 0.006914802645700836
```

Two independent 25-fold runs (5-fold x 5 repeats) over different feature sets should not
produce that. Recomputation confirms they do not:

| configuration | seed 0 std | seed 7 std |
|---|---|---|
| honest, all 61 features | **0.006744571370997998** | 0.011086614173228350 |
| `everything_but_leak`, 57 features | 0.006914802645700829 | 0.008540951814389065 |

The fresh honest run gives `0.006744…`, not the `0.006914…` recorded in
`learnability_analysis.json`, and at seed 7 the two configurations differ clearly. The
fifteen-digit agreement in the artifacts is therefore not a property of the data. One of the
two dispersion figures appears to have been inherited rather than computed.

Secondary discrepancy found while checking: the fresh pooled honest *mean* is `0.5988031`
against the recorded `0.5987402`. Small, but it means the recomputation is not bit-identical
to whatever produced the file, so the attribution above is "one of these two figures is
wrong" rather than a specific verdict on which.

**What depends on it:** nothing published. `main_v2.tex` reports no error bars on the
dose-response; the only `±` values in the paper come from `permutation_control.json`, a
different script, and those were verified separately.

**Before this can be used:** all three reviewers asked for error bars on the dose-response
figure. Do not take them from this file. Recompute `learnability_analysis.py` end to end and
confirm the pooled honest mean and std both reproduce before plotting any interval from it.

**Reproduce:** the seed comparison above is four `cross_val_score` calls over the pooled
informative-query matrix; see `experiments/router_experiments/verify_paper_claims.py` for the
protocol, which is the same one `learnability_analysis.py` uses.

---

## 1b. `two_arm_ablation.json` — the MDE divides by the wrong test-split size

**Status:** open, effect negligible. No published number changes.

`two_arm_ablation.py:131` computes the split size nominally rather than reading it:

```python
n_test = max(1, int(round(len(R) * test_ratio)))   # round(916 * 0.15) = 137
```

Every other script uses the real split, which is **138**. `per_retriever_router.json` records
`n_train=641, n_val=137, n_test=138` for HotpotQA, and 641 + 137 + 138 = 916. The 137 the MDE
divides by is the *validation* size. `two_arm_ablation.md` Table 2 therefore prints a HotpotQA
split size that `main_v2.tex` contradicts (the paper's 138 is the correct one).

**What depends on it:** the MDE, which scales as 1/sqrt(n). Recomputing at n=138 moves HotpotQA's
three-arm MDE from 68.9% to 68.7% and the two-arm from 83.2% to 82.9%. The ranges the paper
once quoted — 53–69% and 65–83% — are unchanged by this, but they are superseded: they used
whole-corpus SD and headroom while capture is normalised by test-split headroom. The camera-ready
quotes the test-split MDE from `results/crc_v2/r6_testsplit_power/summary.json` (one-sided
68/43/72% three-arm, 85/51/78% two-arm).

**Before this can be used:** if `two_arm_ablation.py` is re-run, take `n_test` from the split
function rather than recomputing it, and confirm Table 2 reads 138.

---

## 1c. `per_retriever_router_base.json` is a byte-identical duplicate

**Status:** open. Fixed in the paper; the artifact is still misleading.

```
per_retriever_router.json       sha256 e282961eea87e766…
per_retriever_router_base.json  sha256 e282961eea87e766…   identical
```

An earlier draft counted "45 configurations" across the three `per_retriever_router*` files and
applied Bonferroni at alpha = 0.05/45 = 0.0011. That denominator is 15 distinct configurations
counted three times, one of the three being an exact copy of another. `main_v2.tex` now scopes the
claim to the 15 distinct configurations at alpha = 0.0033. The conclusion is unchanged — the
smallest p is 0.0293 — but the artifact tree still invites the wrong count.

---

## 2. `audit_numbers.py` cannot see an invalid comparison

**Status:** by design, recorded so a green run is not mistaken for correctness.

`paper/audit_numbers.py` traces every numeric claim in the paper to a results file. It found
nine wrong numbers, all inherited from superseded runs. It cannot, however, detect a number
that is correctly traced but licenses an invalid inference.

The two-arm ablation is the worked example. Its lift figures (`-3.3/-3.1/-6.4` to
`-2.0/-2.8/-5.0`) trace correctly to `two_arm_ablation.json`. The claim they were used to
support — that removing the redundant arm *improved* predictability — was still wrong,
because dropping Hybrid also moves the majority baseline (63.4→69.6, 61.7→69.0, 56.4→63.0)
and changes the informative-query set. The numbers were right; the comparison was not
controlled.

No static check over (claim, artifact) pairs can catch that class. It needs someone to ask
what else moved.
