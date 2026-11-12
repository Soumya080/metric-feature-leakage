# Learned Cross-Scoring Cascade

Run the cheap retriever, inspect what it returned, escalate to Hybrid only when
the evidence says it failed. Quality is MRR@k; cost is in structural units
(BM25=1, Dense=3), so escalating costs 4.

`vs matched random` is the number that decides whether this works: the learned
policy compared against escalating at random AT THE SAME RATE. Any policy that
escalates more will look better than always-accept; only beating matched random
shows the post-retrieval evidence carried information.

| dataset | probe | esc. rate | quality | cost | vs always-accept | vs matched random | p |
|---|---|---:|---:|---:|---:|---:|---:|
| nq | bm25 | 28.7% | 0.0986 | 1.86 | +0.0029 | -0.0093 | 0.900 |
| nq | dense | 0.6% | 0.1570 | 3.01 | +0.0000 | +0.0001 | 0.816 |
| triviaqa | bm25 | 8.7% | 0.1436 | 1.26 | +0.0008 | -0.0030 | 0.835 |
| triviaqa | dense | 28.1% | 0.1822 | 3.28 | -0.0102 | -0.0080 | 0.851 |
| hotpotqa | bm25 | 0.7% | 0.0341 | 1.02 | -0.0036 | -0.0037 | 1.000 |
| hotpotqa | dense | 0.0% | 0.0658 | 3.00 | +0.0000 | -0.0000 | 1.000 |

## Reference points

| dataset | probe | always-accept | always-escalate (Hybrid) | oracle cascade |
|---|---|---|---|---|
| nq | bm25 | 0.0956 @ 1.00 | 0.1381 @ 4.00 | 0.1503 @ 1.78 |
| nq | dense | 0.1570 @ 3.00 | 0.1381 @ 4.00 | 0.1837 @ 3.16 |
| triviaqa | bm25 | 0.1428 @ 1.00 | 0.1846 @ 4.00 | 0.1991 @ 1.81 |
| triviaqa | dense | 0.1925 @ 3.00 | 0.1846 @ 4.00 | 0.2521 @ 3.23 |
| hotpotqa | bm25 | 0.0378 @ 1.00 | 0.0467 @ 4.00 | 0.0616 @ 1.50 |
| hotpotqa | dense | 0.0658 @ 3.00 | 0.0467 @ 4.00 | 0.0884 @ 3.12 |

## Reading this

A positive `vs matched random` with p < 0.05 means the cascade is targeting its escalations, not just buying quality with compute. That is the claim.

`vs always-escalate` near zero at a much lower cost is also a win, and often the more useful one in practice: same quality as always running Hybrid, for a fraction of the compute.

If `vs matched random` is ~0, the escalation signal is not informative and the cascade reduces to spending more — which should be reported as such rather than tuned until it looks positive.

