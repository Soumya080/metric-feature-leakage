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
| nq | bm25 | 15.5% | 0.0930 | 1.47 | -0.0026 | -0.0095 | 0.970 |
| nq | dense | 35.6% | 0.1611 | 3.36 | +0.0041 | +0.0109 | 0.046 |
| triviaqa | bm25 | 0.9% | 0.1428 | 1.03 | +0.0000 | -0.0004 | 0.887 |
| triviaqa | dense | 7.4% | 0.1875 | 3.07 | -0.0049 | -0.0042 | 0.835 |
| hotpotqa | bm25 | 1.4% | 0.0388 | 1.04 | +0.0011 | +0.0009 | 0.103 |
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

