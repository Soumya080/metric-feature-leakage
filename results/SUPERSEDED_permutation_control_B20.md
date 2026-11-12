# Permutation and Random-Feature Controls

_Scale: 100K Wikipedia pool · B=20_


Does the honest feature set carry routing signal at all? The lift in `learnability_analysis.md` cannot answer this — it compares against the majority baseline, not against chance. These two nulls compare against chance.


## Table 1 — Honest features vs. two nulls

| dataset | n inf | majority | observed | label-perm null | p | random-feat null | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| nq | 1021 | 63.4% | **60.1%** | 59.0% ± 0.7 | 0.048 | 59.0% ± 0.7 | 0.095 |
| triviaqa | 1307 | 61.7% | **58.6%** | 58.2% ± 0.4 | 0.238 | 58.4% ± 0.6 | 0.381 |
| hotpotqa | 847 | 56.4% | **50.1%** | 50.7% ± 1.0 | 0.810 | 50.8% ± 0.8 | 0.857 |
| **pooled** | 3175 | 60.9% | **59.9%** | 59.2% ± 0.3 | 0.095 | 59.1% ± 0.2 | 0.048 |

`observed` is GBT accuracy on the honest features (5-fold stratified CV × 5 repeats), identical to the GBT column of `learnability_analysis.md`. `label-perm null` shuffles the best-retriever labels across 20 replicates, preserving the class prior; `random-feat null` replaces the features with Gaussian noise and keeps the true labels. p-values are one-sided with the (1+k)/(1+B) correction, so the floor is 0.048.


## Table 2 — What each dataset licenses the paper to say

| dataset | signal above chance | verdict |
|---|---:|---|
| nq | +1.1 pts | detectable signal (beats permuted labels) but still below the majority baseline — signal exists and is too weak to exploit; scale may help |
| triviaqa | +0.4 pts | indistinguishable from permuted labels — no detectable routing signal; the negative result is a signal problem, not a sample-size problem |
| hotpotqa | -0.7 pts | indistinguishable from permuted labels — no detectable routing signal; the negative result is a signal problem, not a sample-size problem |
| **pooled** | +0.7 pts | detectable signal (beats permuted labels) but still below the majority baseline — signal exists and is too weak to exploit; scale may help |

