# Permutation and Random-Feature Controls

_Scale: 100K pool, B=1000 · B=1000_


Does the honest feature set carry routing signal at all? The lift in `learnability_analysis.md` cannot answer this — it compares against the majority baseline, not against chance. These two nulls compare against chance.


## Table 1 — Honest features vs. two nulls

| dataset | n inf | majority | observed | label-perm null | p | random-feat null | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| nq | 1021 | 63.4% | **60.1%** | 59.1% ± 0.6 | 0.066 | 59.3% ± 0.6 | 0.090 |
| triviaqa | 1307 | 61.7% | **58.6%** | 58.2% ± 0.5 | 0.181 | 58.2% ± 0.5 | 0.187 |
| hotpotqa | 847 | 56.4% | **50.1%** | 50.8% ± 1.0 | 0.740 | 51.0% ± 0.9 | 0.820 |
| **pooled** | 3175 | 60.9% | **59.9%** | 59.1% ± 0.2 | 0.003 | 59.1% ± 0.2 | 0.002 |

`observed` is GBT accuracy on the honest features (5-fold stratified CV × 5 repeats), identical to the GBT column of `learnability_analysis.md`. `label-perm null` shuffles the best-retriever labels across 1000 replicates, preserving the class prior; `random-feat null` replaces the features with Gaussian noise and keeps the true labels. p-values are one-sided with the (1+k)/(1+B) correction, so the floor is 0.001.


## Table 2 — What each dataset licenses the paper to say

| dataset | signal above chance | verdict |
|---|---:|---|
| nq | +1.0 pts | detectable signal (beats permuted labels) but still below the majority baseline — signal exists and is too weak to exploit; scale may help |
| triviaqa | +0.5 pts | indistinguishable from permuted labels — no detectable routing signal; the negative result is a signal problem, not a sample-size problem |
| hotpotqa | -0.7 pts | indistinguishable from permuted labels — no detectable routing signal; the negative result is a signal problem, not a sample-size problem |
| **pooled** | +0.7 pts | detectable signal (beats permuted labels) but still below the majority baseline — signal exists and is too weak to exploit; scale may help |

