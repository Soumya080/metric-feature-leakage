> **Superseded (2026-11-09):** the minimum-detectable-effect section below uses whole-corpus SD and headroom. The paper quotes the test-split values from `results/analyses/split_power/summary.json` (one-sided 68/43/72% three-arm, 85/51/78% two-arm). The lift and majority-baseline numbers in this file are unaffected.

# Two-Arm Ablation and Statistical Power

_Restricted action space: bm25+dense_


Hybrid is RRF of BM25 and Dense, so the three-arm action space contains a redundant arm. If routing fails only because one of three arms is a function of the other two, that is a setup problem rather than a finding. This re-runs the core measurement on two genuinely distinct retrievers.


## Table 1 — Three arms vs. two arms

| dataset | arms | n | best fixed | oracle | headroom | tie rate | informative n | majority | GBT | lift |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nq | bm25+dense+hybrid | 1157 | dense 0.1678 | 0.2121 | +0.0444 | 11.8% | 1021 | 63.4% | 60.1% ± 2.3 | **-3.3%** |
| nq | bm25+dense | 1157 | dense 0.1678 | 0.1968 | +0.0290 | 5.5% | 1093 | 69.6% | 67.6% ± 1.9 | **-2.0%** |
| triviaqa | bm25+dense+hybrid | 1539 | dense 0.2049 | 0.2659 | +0.0611 | 15.1% | 1307 | 61.7% | 58.6% ± 1.4 | **-3.1%** |
| triviaqa | bm25+dense | 1539 | dense 0.2049 | 0.2463 | +0.0414 | 7.9% | 1418 | 69.0% | 66.2% ± 1.8 | **-2.8%** |
| hotpotqa | bm25+dense+hybrid | 916 | dense 0.1071 | 0.1706 | +0.0636 | 7.5% | 847 | 56.4% | 50.1% ± 3.0 | **-6.4%** |
| hotpotqa | bm25+dense | 916 | dense 0.1071 | 0.1537 | +0.0467 | 3.6% | 883 | 63.0% | 58.0% ± 2.8 | **-5.0%** |

`lift` is GBT accuracy minus the majority-class baseline. A negative lift on BOTH action spaces means the negative result is not an artifact of Hybrid's redundancy: it holds between two retrievers with no functional dependence on each other.


## Table 2 — Minimum detectable effect

| dataset | arms | n test | sd of per-query gain | MDE | headroom | must capture at least |
|---|---|---:|---:|---:|---:|---:|
| nq | bm25+dense+hybrid | 174 | 0.1414 | 0.0300 | 0.0444 | **68%** of headroom |
| nq | bm25+dense | 174 | 0.1133 | 0.0241 | 0.0290 | **83%** of headroom |
| triviaqa | bm25+dense+hybrid | 231 | 0.1763 | 0.0325 | 0.0611 | **53%** of headroom |
| triviaqa | bm25+dense | 231 | 0.1468 | 0.0271 | 0.0414 | **65%** of headroom |
| hotpotqa | bm25+dense+hybrid | 137 | 0.1831 | 0.0438 | 0.0636 | **69%** of headroom |
| hotpotqa | bm25+dense | 137 | 0.1622 | 0.0388 | 0.0467 | **83%** of headroom |

Smallest mean per-query improvement over the best fixed retriever that a test split of this size could detect at alpha = 0.05 with 80% power, in a paired design. The last column is the MDE as a fraction of available headroom, and it is the number the limitations section should quote: **a router capturing less than that share of the oracle headroom cannot be distinguished from the best fixed retriever on a test split this size, whether or not it is genuinely better.**


This is why the column is expressed as a share rather than as a yes/no verdict. Asking whether a *perfect* oracle router would be detectable is not informative — the oracle gain is by construction the largest effect available, so the answer is almost always yes. The binding question is how modest a real improvement could be before this design goes blind to it, and here that threshold is high: roughly half to four fifths of all available headroom.


Note also that the difference series used is the per-query oracle gain, the largest difference any router could produce. A real router differs from best-fixed on fewer queries and by less, so its per-query variance is lower but its mean effect is much lower too; the share above is therefore an optimistic floor.

