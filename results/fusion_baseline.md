# Fusion Baselines: route, or just combine the lists?

At decision time all three ranked lists already exist. This asks whether selecting one beats merging them.

`RRF (Hybrid)` is not a new baseline — Hybrid **is** reciprocal rank fusion over BM25 and Dense across the full corpus, so it has been in Table 1 all along. `Convex` is the addition: `alpha * norm(BM25) + (1-alpha) * norm(Dense)`, with alpha chosen on the **training** split and reported on test.

| dataset | n test | best arm | RRF (Hybrid) | Convex (alpha) | Oracle | Convex - best arm |
|---|---:|---|---:|---:|---:|---:|
| nq | 174 | dense 0.1655 | 0.1464 | 0.1647 (a=0.1) | 0.2001 | **-0.0008** |
| triviaqa | 231 | dense 0.2000 | 0.1907 | 0.2049 (a=0.1) | 0.2694 | **+0.0048** |
| hotpotqa | 138 | dense 0.0753 | 0.0572 | 0.0598 (a=0.3) | 0.1096 | **-0.0155** |

A positive final column means fusion beats committing to the single best retriever. If it is positive while every routing method is negative, the honest recommendation from this work is to fuse rather than route — and that belongs in the paper regardless of how it reflects on the original method.

## Selected alpha, and what the sweep looks like on test

**nq** — alpha selected on train: **0.1**

| alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test MRR | 0.1655 | 0.1647 | 0.1519 | 0.1434 | 0.1379 | 0.1325 | 0.1234 | 0.1161 | 0.1071 | 0.1091 | 0.1029 |

**triviaqa** — alpha selected on train: **0.1**

| alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test MRR | 0.2000 | 0.2049 | 0.1951 | 0.1926 | 0.1880 | 0.1799 | 0.1719 | 0.1631 | 0.1586 | 0.1538 | 0.1501 |

**hotpotqa** — alpha selected on train: **0.3**

| alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test MRR | 0.0753 | 0.0750 | 0.0675 | 0.0598 | 0.0578 | 0.0537 | 0.0537 | 0.0512 | 0.0532 | 0.0473 | 0.0468 |

_alpha = 0 is pure Dense, alpha = 1 is pure BM25; both reproduce the corresponding arm exactly, which is the implementation's self-check._
