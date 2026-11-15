# Cross-Retriever Agreement vs. Routing Headroom

Tests the mechanism behind the routing claim: headroom (oracle - best-fixed) should
concentrate where BM25 and Dense **disagree**, and vanish where they agree. Where the
two retrievers return the same passages they earn nearly the same reward, so the
routing decision cannot matter — every point of achievable gain must come from the
queries where they diverge.

Agreement = top-5 retrieved-passage overlap (`cross_ret_overlap_5`, feature 26),
computed from passage identity only and never from the gold answer.


## NQ

- Queries analysed: **174**
- Best-fixed retriever (global): **dense**
- Spearman(agreement, headroom) = **-0.018** (p = 8.14e-01)
  - **No significant relationship.** The hypothesis that headroom concentrates where the retrievers disagree is not supported on this dataset, in either direction.

| agreement (top-5 overlap) | n | best-fixed | oracle | headroom | RL router | GBT |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 (disjoint) | 66 | 0.080 | 0.115 | **+0.036** | 0.075 | 0.080 |
| 0.2 | 50 | 0.185 | 0.230 | **+0.045** | 0.155 | 0.180 |
| 0.4 | 42 | 0.182 | 0.210 | **+0.028** | 0.166 | 0.182 |
| 0.6 | 12 | 0.464 | 0.485 | **+0.021** | 0.473 | 0.464 |
| 0.8-1.0 (agree) | 4 | 0.269 | 0.270 | **+0.001** | 0.251 | 0.269 |

## TRIVIAQA

- Queries analysed: **231**
- Best-fixed retriever (global): **dense**
- Spearman(agreement, headroom) = **-0.034** (p = 6.07e-01)
  - **No significant relationship.** The hypothesis that headroom concentrates where the retrievers disagree is not supported on this dataset, in either direction.

| agreement (top-5 overlap) | n | best-fixed | oracle | headroom | RL router | GBT |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 (disjoint) | 81 | 0.139 | 0.212 | **+0.073** | 0.136 | 0.139 |
| 0.2 | 97 | 0.220 | 0.296 | **+0.076** | 0.208 | 0.220 |
| 0.4 | 38 | 0.200 | 0.268 | **+0.069** | 0.194 | 0.200 |
| 0.6 | 13 | 0.390 | 0.395 | **+0.005** | 0.300 | 0.390 |
| 0.8-1.0 (agree) | 2 | 0.501 | 0.501 | **+0.000** | 0.501 | 0.501 |

## HOTPOTQA

- Queries analysed: **138**
- Best-fixed retriever (global): **dense**
- Spearman(agreement, headroom) = **-0.090** (p = 2.95e-01)
  - **No significant relationship.** The hypothesis that headroom concentrates where the retrievers disagree is not supported on this dataset, in either direction.

| agreement (top-5 overlap) | n | best-fixed | oracle | headroom | RL router | GBT |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 (disjoint) | 64 | 0.056 | 0.106 | **+0.050** | 0.064 | 0.056 |
| 0.2 | 45 | 0.111 | 0.134 | **+0.024** | 0.084 | 0.111 |
| 0.4 | 25 | 0.065 | 0.084 | **+0.018** | 0.055 | 0.065 |
| 0.6 | 3 | 0.059 | 0.059 | **+0.000** | 0.059 | 0.059 |
| 0.8-1.0 (agree) | 1 | 0.002 | 0.002 | **+0.000** | 0.002 | 0.002 |
