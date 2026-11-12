# Routing Learnability Analysis

_Scale: 400 queries/dataset pilot, MRR@20, all gold types_


## Table 1 — Routing headroom and reward sparsity

| dataset | n | best fixed | oracle | headroom | all-zero | learnable |
|---|---:|---:|---:|---:|---:|---:|
| nq | 398 | Dense 0.1448 | 0.1805 | **+0.0357** (25%) | 59.8% | 35.2% |
| triviaqa | 397 | Dense 0.1599 | 0.1893 | **+0.0294** (18%) | 62.0% | 31.2% |
| hotpotqa | 400 | Dense 0.1965 | 0.2407 | **+0.0442** (22%) | 54.0% | 32.8% |
| **pooled** | 1195 | Dense 0.1671 | 0.2036 | **+0.0364** (22%) | 58.6% | 33.1% |

Headroom is the gain an oracle would get from per-query routing. `all-zero` is the share of queries where every retriever scores 0; `learnable` is the share with a strictly-best action — the only queries that carry routing signal.


## Table 2 — Can the best retriever be predicted from label-free features?

| dataset | informative n | majority | GBT (honest) | LogReg (honest) | lift | GBT (leaky) |
|---|---:|---:|---:|---:|---:|---:|
| nq | 140 | 67.9% | 56.7% ± 7.4 | 64.9% ± 4.8 | **-11.1%** | 83.1% |
| triviaqa | 124 | 70.2% | 61.8% ± 6.9 | 65.0% ± 6.0 | **-8.4%** | 84.7% |
| hotpotqa | 131 | 63.4% | 52.4% ± 10.4 | 63.1% ± 5.7 | **-11.0%** | 82.6% |
| **pooled** | 395 | 67.1% | 60.9% ± 3.5 | 66.4% ± 1.9 | **-6.2%** | 87.9% |

Accuracy predicting the strictly-best retriever (5-fold stratified CV × 5 repeats). `majority` is the always-predict-the-most-common-retriever baseline. A negative lift means the classifier does **worse than a constant policy**. The leaky column reproduces the original MRR-derived features, which are the reward itself for 2 of the 3 actions — the gap between it and the honest columns is the size of the leak.


## Feature-group breakdown (pooled)

- All 29 features: **60.9%**
- Retrieval-score features only (22–25): **58.6%**
- Cross-retriever agreement only (26–28): **66.1%**
- Majority baseline: **67.1%**

