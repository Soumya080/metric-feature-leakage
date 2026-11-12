# Permutation and Random-Feature Controls

_Scale: 100K Wikipedia pool (NQ) · B=20_


Does the honest feature set carry routing signal at all? The lift in `learnability_analysis.md` cannot answer this — it compares against the majority baseline, not against chance. These two nulls compare against chance.


## Table 1 — Honest features vs. two nulls

| dataset | n inf | majority | observed | label-perm null | p | random-feat null | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| nq | 1013 | 64.7% | **61.3%** | 60.8% ± 0.6 | 0.238 | 60.9% ± 0.6 | 0.286 |

`observed` is GBT accuracy on the honest features (5-fold stratified CV × 5 repeats), identical to the GBT column of `learnability_analysis.md`. `label-perm null` shuffles the best-retriever labels across 20 replicates, preserving the class prior; `random-feat null` replaces the features with Gaussian noise and keeps the true labels. p-values are one-sided with the (1+k)/(1+B) correction, so the floor is 0.048.


## Table 2 — What each dataset licenses the paper to say

| dataset | signal above chance | verdict |
|---|---:|---|
| nq | +0.5 pts | indistinguishable from permuted labels — no detectable routing signal; the negative result is a signal problem, not a sample-size problem |

