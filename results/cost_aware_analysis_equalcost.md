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
| 0.000 | 66% | 26% | 8% | **65.7%** | dense | +0.0451 | 0.2049 | 0.541 |
| 0.005 | 66% | 26% | 8% | **65.7%** | dense | +0.0449 | 0.2049 | 0.541 |
| 0.009 | 66% | 26% | 8% | **65.7%** | dense | +0.0447 | 0.2049 | 0.541 |
| 0.015 | 66% | 26% | 8% | **65.7%** | dense | +0.0445 | 0.2049 | 0.541 |
| 0.026 | 66% | 26% | 8% | **65.9%** | dense | +0.0441 | 0.2049 | 0.538 |
| 0.044 | 66% | 27% | 7% | **66.3%** | dense | +0.0434 | 0.2048 | 0.535 |
| 0.076 | 67% | 27% | 6% | **66.6%** | dense | +0.0423 | 0.2046 | 0.532 |
| 0.131 | 67% | 27% | 5% | **67.3%** | dense | +0.0407 | 0.2040 | 0.527 |
| 0.226 | 68% | 28% | 4% | **68.4%** | dense | +0.0386 | 0.2027 | 0.519 |
| 0.390 | 69% | 29% | 2% | **69.1%** | dense | +0.0360 | 0.2003 | 0.511 |
| 0.673 | 70% | 29% | 2% | **69.6%** | dense | +0.0333 | 0.1983 | 0.508 |
| 1.160 | 70% | 29% | 1% | **70.0%** | dense | +0.0304 | 0.1933 | 0.503 |
| 2.000 | 70% | 30% | 0% | **70.4%** | dense | +0.0295 | 0.1893 | 0.500 |

majority at lambda=0: 65.7% -> minimum 65.7% at lambda=0.000 (drop 0.0%) => **DEGENERATE**

_Decided by: majority never falls below 65% (best 65.7%) and cost moves it only 0.0%._


## triviaqa  (n=1539, metric=mrr_at_20, cost=structural)

| lambda | BM25 | Dense | Hybrid | majority | best fixed | headroom | oracle quality | oracle cost |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0.000 | 65% | 26% | 10% | **64.7%** | dense | +0.0618 | 0.2586 | 0.548 |
| 0.005 | 65% | 26% | 10% | **64.7%** | dense | +0.0616 | 0.2586 | 0.548 |
| 0.009 | 65% | 26% | 9% | **64.7%** | dense | +0.0614 | 0.2586 | 0.547 |
| 0.015 | 65% | 26% | 9% | **64.7%** | dense | +0.0611 | 0.2586 | 0.546 |
| 0.026 | 65% | 26% | 9% | **64.9%** | dense | +0.0606 | 0.2585 | 0.545 |
| 0.044 | 65% | 26% | 9% | **64.9%** | dense | +0.0598 | 0.2585 | 0.543 |
| 0.076 | 65% | 27% | 8% | **65.2%** | dense | +0.0585 | 0.2583 | 0.540 |
| 0.131 | 67% | 27% | 6% | **66.6%** | dense | +0.0565 | 0.2575 | 0.532 |
| 0.226 | 68% | 28% | 4% | **67.8%** | dense | +0.0541 | 0.2558 | 0.522 |
| 0.390 | 68% | 28% | 3% | **68.5%** | dense | +0.0510 | 0.2538 | 0.515 |
| 0.673 | 69% | 29% | 2% | **68.7%** | dense | +0.0473 | 0.2517 | 0.511 |
| 1.160 | 69% | 30% | 1% | **69.5%** | dense | +0.0432 | 0.2445 | 0.504 |
| 2.000 | 70% | 30% | 0% | **69.9%** | dense | +0.0419 | 0.2386 | 0.500 |

majority at lambda=0: 64.7% -> minimum 64.7% at lambda=0.000 (drop 0.0%) => **NON-DEGENERATE**

_Decided by: majority 64.7% is at or below the 65% bar on its own, independently of cost._


## hotpotqa  (n=916, metric=mrr_at_20, cost=structural)

| lambda | BM25 | Dense | Hybrid | majority | best fixed | headroom | oracle quality | oracle cost |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0.000 | 73% | 18% | 9% | **73.0%** | dense | +0.0644 | 0.1621 | 0.546 |
| 0.005 | 73% | 18% | 9% | **73.0%** | dense | +0.0642 | 0.1621 | 0.546 |
| 0.009 | 73% | 18% | 9% | **73.1%** | dense | +0.0640 | 0.1621 | 0.545 |
| 0.015 | 73% | 18% | 9% | **73.3%** | dense | +0.0637 | 0.1621 | 0.544 |
| 0.026 | 73% | 18% | 8% | **73.5%** | dense | +0.0633 | 0.1620 | 0.542 |
| 0.044 | 74% | 18% | 8% | **73.7%** | dense | +0.0625 | 0.1620 | 0.539 |
| 0.076 | 74% | 19% | 7% | **74.1%** | dense | +0.0613 | 0.1617 | 0.535 |
| 0.131 | 75% | 20% | 5% | **74.9%** | dense | +0.0596 | 0.1608 | 0.527 |
| 0.226 | 76% | 20% | 4% | **75.7%** | dense | +0.0574 | 0.1600 | 0.521 |
| 0.390 | 77% | 21% | 3% | **76.9%** | dense | +0.0546 | 0.1572 | 0.513 |
| 0.673 | 78% | 21% | 2% | **77.5%** | dense | +0.0516 | 0.1545 | 0.508 |
| 1.160 | 78% | 21% | 1% | **77.8%** | dense | +0.0484 | 0.1512 | 0.504 |
| 2.000 | 78% | 22% | 0% | **78.3%** | dense | +0.0472 | 0.1449 | 0.500 |

majority at lambda=0: 73.0% -> minimum 73.0% at lambda=0.000 (drop 0.0%) => **DEGENERATE**

_Decided by: majority never falls below 65% (best 73.0%) and cost moves it only 0.0%._


## How to read the verdict

NON-DEGENERATE means some lambda leaves the majority baseline at or below 65% — no single retriever is optimal often enough to coast on, so a router has real work to do and real headroom to capture. It does NOT mean the choice is predictable; that is the separate question the learnability analysis answers.

'NON-DEGENERATE (via cost)' means the majority baseline was high at lambda=0 but fell by at least 10% once compute was charged — i.e. cost is what created the routing decision.

DEGENERATE means no lambda produces a mixed decision, and the cost-aware reframing should be dropped rather than tuned.


One algebraic caveat worth checking against the table: for any two actions, `r(a) - r(b) = [MRR(a) - MRR(b)] - lambda*[cost(a) - cost(b)]`, so two retrievers with EQUAL cost have a lambda-invariant comparison. If the BM25/Dense shares are frozen across the whole sweep, that is this identity, not a property of the data.

