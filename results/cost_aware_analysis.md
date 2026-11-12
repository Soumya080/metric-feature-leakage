# Cost-Aware Routing: is the decision non-degenerate?

For each lambda, the ORACLE action per query under `MRR - lambda * cost`, and how
concentrated that oracle choice is. `majority` is the accuracy of always picking the
single most common optimal action — the baseline a router has to beat. A large drop
in `majority` as lambda grows means cost genuinely creates a routing decision; a flat
`majority` means it does not, and cost-aware routing is as degenerate as the
quality-only version.


## nq  (n=1157, metric=mrr_at_20, cost=structural)

| lambda | BM25 | Dense | Hybrid | majority | best fixed | headroom | oracle quality | oracle cost |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0.000 | 66% | 26% | 8% | **65.7%** | dense | +0.0451 | 0.2049 | 0.442 |
| 0.005 | 66% | 26% | 8% | **65.7%** | dense | +0.0466 | 0.2049 | 0.442 |
| 0.009 | 66% | 26% | 8% | **65.7%** | dense | +0.0478 | 0.2049 | 0.442 |
| 0.015 | 66% | 26% | 8% | **65.9%** | dense | +0.0497 | 0.2049 | 0.440 |
| 0.026 | 66% | 26% | 8% | **66.2%** | dense | +0.0530 | 0.2049 | 0.438 |
| 0.044 | 67% | 26% | 7% | **66.6%** | dense | +0.0588 | 0.2048 | 0.435 |
| 0.076 | 68% | 26% | 6% | **67.7%** | dense | +0.0690 | 0.2043 | 0.427 |
| 0.131 | 72% | 23% | 5% | **72.0%** | bm25 | +0.0797 | 0.2016 | 0.402 |
| 0.226 | 80% | 17% | 3% | **79.7%** | bm25 | +0.0674 | 0.1943 | 0.360 |
| 0.390 | 85% | 13% | 2% | **85.1%** | bm25 | +0.0521 | 0.1853 | 0.330 |
| 0.673 | 91% | 8% | 1% | **91.2%** | bm25 | +0.0345 | 0.1680 | 0.297 |
| 1.160 | 95% | 5% | 0% | **94.8%** | bm25 | +0.0155 | 0.1481 | 0.276 |
| 2.000 | 100% | 0% | 0% | **100.0%** | bm25 | +0.0000 | 0.1020 | 0.250 |

majority at lambda=0: 65.7% -> minimum 65.7% at lambda=0.000 (drop 0.0%) => **DEGENERATE**

_Decided by: majority never falls below 65% (best 65.7%) and cost moves it only 0.0%._


## triviaqa  (n=1539, metric=mrr_at_20, cost=structural)

| lambda | BM25 | Dense | Hybrid | majority | best fixed | headroom | oracle quality | oracle cost |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0.000 | 65% | 26% | 10% | **64.7%** | dense | +0.0618 | 0.2586 | 0.451 |
| 0.005 | 65% | 26% | 9% | **64.7%** | dense | +0.0633 | 0.2586 | 0.450 |
| 0.009 | 65% | 26% | 9% | **64.7%** | dense | +0.0644 | 0.2586 | 0.450 |
| 0.015 | 65% | 26% | 9% | **65.0%** | dense | +0.0663 | 0.2586 | 0.448 |
| 0.026 | 65% | 26% | 9% | **65.2%** | dense | +0.0695 | 0.2585 | 0.447 |
| 0.044 | 65% | 26% | 9% | **65.4%** | dense | +0.0752 | 0.2585 | 0.445 |
| 0.076 | 67% | 26% | 7% | **66.9%** | dense | +0.0851 | 0.2578 | 0.434 |
| 0.131 | 71% | 23% | 6% | **71.5%** | bm25 | +0.0934 | 0.2549 | 0.407 |
| 0.226 | 79% | 17% | 4% | **78.7%** | bm25 | +0.0807 | 0.2479 | 0.366 |
| 0.390 | 82% | 15% | 3% | **82.5%** | bm25 | +0.0637 | 0.2418 | 0.345 |
| 0.673 | 88% | 10% | 1% | **88.4%** | bm25 | +0.0413 | 0.2236 | 0.312 |
| 1.160 | 94% | 6% | 0% | **94.0%** | bm25 | +0.0173 | 0.1938 | 0.281 |
| 2.000 | 100% | 0% | 0% | **100.0%** | bm25 | +0.0000 | 0.1409 | 0.250 |

majority at lambda=0: 64.7% -> minimum 64.7% at lambda=0.000 (drop 0.0%) => **NON-DEGENERATE**

_Decided by: majority 64.7% is at or below the 65% bar on its own, independently of cost._


## hotpotqa  (n=916, metric=mrr_at_20, cost=structural)

| lambda | BM25 | Dense | Hybrid | majority | best fixed | headroom | oracle quality | oracle cost |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0.000 | 73% | 18% | 9% | **73.0%** | dense | +0.0644 | 0.1621 | 0.408 |
| 0.005 | 73% | 18% | 9% | **73.0%** | dense | +0.0661 | 0.1621 | 0.408 |
| 0.009 | 73% | 18% | 9% | **73.1%** | dense | +0.0673 | 0.1621 | 0.407 |
| 0.015 | 73% | 18% | 9% | **73.5%** | dense | +0.0695 | 0.1621 | 0.404 |
| 0.026 | 74% | 18% | 8% | **73.7%** | dense | +0.0732 | 0.1620 | 0.403 |
| 0.044 | 74% | 18% | 8% | **73.8%** | bm25 | +0.0712 | 0.1620 | 0.401 |
| 0.076 | 75% | 18% | 7% | **75.0%** | bm25 | +0.0665 | 0.1614 | 0.392 |
| 0.131 | 78% | 16% | 6% | **78.4%** | bm25 | +0.0591 | 0.1594 | 0.373 |
| 0.226 | 85% | 11% | 3% | **85.0%** | bm25 | +0.0494 | 0.1524 | 0.334 |
| 0.390 | 89% | 9% | 2% | **89.3%** | bm25 | +0.0381 | 0.1452 | 0.309 |
| 0.673 | 93% | 5% | 1% | **93.3%** | bm25 | +0.0247 | 0.1332 | 0.286 |
| 1.160 | 96% | 3% | 0% | **96.4%** | bm25 | +0.0102 | 0.1165 | 0.269 |
| 2.000 | 100% | 0% | 0% | **100.0%** | bm25 | +0.0000 | 0.0841 | 0.250 |

majority at lambda=0: 73.0% -> minimum 73.0% at lambda=0.000 (drop 0.0%) => **DEGENERATE**

_Decided by: majority never falls below 65% (best 73.0%) and cost moves it only 0.0%._


## How to read the verdict

NON-DEGENERATE means some lambda leaves the majority baseline at or below 65% — no single retriever is optimal often enough to coast on, so a router has real work to do and real headroom to capture. It does NOT mean the choice is predictable; that is the separate question the learnability analysis answers.

'NON-DEGENERATE (via cost)' means the majority baseline was high at lambda=0 but fell by at least 10% once compute was charged — i.e. cost is what created the routing decision.

DEGENERATE means no lambda produces a mixed decision, and the cost-aware reframing should be dropped rather than tuned.


One algebraic caveat worth checking against the table: for any two actions, `r(a) - r(b) = [MRR(a) - MRR(b)] - lambda*[cost(a) - cost(b)]`, so two retrievers with EQUAL cost have a lambda-invariant comparison. If the BM25/Dense shares are frozen across the whole sweep, that is this identity, not a property of the data.

