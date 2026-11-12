# Routing Learnability Analysis

_Scale: full run_


## Table 1 — Routing headroom and reward sparsity

| dataset | n | best fixed | oracle | headroom | all-zero | learnable |
|---|---:|---:|---:|---:|---:|---:|
| nq | 1157 | Dense 0.1678 | 0.2121 | **+0.0444** (26%) | 0.0% | 88.2% |
| triviaqa | 1539 | Dense 0.2049 | 0.2659 | **+0.0611** (30%) | 0.0% | 84.9% |
| hotpotqa | 916 | Dense 0.1071 | 0.1706 | **+0.0636** (59%) | 0.0% | 92.5% |
| **pooled** | 3612 | Dense 0.1682 | 0.2245 | **+0.0564** (34%) | 0.0% | 87.9% |

Headroom is the gain an oracle would get from per-query routing. `all-zero` is the share of queries where every retriever scores 0; `learnable` is the share with a strictly-best action — the only queries that carry routing signal.


## Table 2 — Can the best retriever be predicted from label-free features?

| dataset | informative n | majority | GBT (honest) | LogReg (honest) | lift | GBT (leaky) |
|---|---:|---:|---:|---:|---:|---:|
| nq | 1021 | 63.4% | 60.1% ± 2.3 | 61.0% ± 1.6 | **-3.3%** | 72.0% |
| triviaqa | 1307 | 61.7% | 58.6% ± 1.5 | 59.6% ± 1.5 | **-3.1%** | 71.0% |
| hotpotqa | 847 | 56.4% | 50.1% ± 3.1 | 51.2% ± 3.0 | **-6.3%** | 66.9% |
| **pooled** | 3175 | 60.9% | 59.9% ± 0.7 | 60.6% ± 0.6 | **-1.0%** | 73.1% |

Accuracy predicting the strictly-best retriever (5-fold stratified CV × 5 repeats). `majority` is the always-predict-the-most-common-retriever baseline. A negative lift means the classifier does **worse than a constant policy**. The leaky column reproduces the original MRR-derived features, which are the reward itself for 2 of the 3 actions — the gap between it and the honest columns is the size of the leak.


## Table 3 — Which feature group carries the signal? (pooled)

Majority baseline: **60.9%**. Lift is accuracy minus that baseline; a negative lift means the group is worse than a constant policy.

| feature group | # features | accuracy | lift |
|---|---:|---:|---:|
| query_only (0-21) | 22 | 59.4% ± 0.8 | **-1.5%** |
| +retrieval_scores (0-25) | 26 | 59.5% ± 0.7 | **-1.3%** |
| +cross_retriever (0-28) | 29 | 59.6% ± 0.8 | **-1.2%** |
| retrieval_only (22-25) | 4 | 59.4% ± 0.7 | **-1.4%** |
| cross_only (26-28) | 3 | 60.8% ± 0.4 | **-0.1%** |
| +embeddings (all) | 61 | 59.9% ± 0.7 | **-1.0%** |
| embeddings_only (29+) | 32 | 59.7% ± 0.8 | **-1.1%** |
| _leaky reference (all)_ | — | 73.1% | +12.2% |

