# Oracle Inflation Analysis

_Seed 0 · 200 random arm-splits · noise sweep averaged over 50 draws_


Oracle headroom is the motivation for routing, and it is computed as a max over three noisy arms — which is upward-biased by Jensen's inequality. These three measurements say how much of the headroom is real.


## Table 1 — Headroom, and the selection bias in choosing an arm

| dataset | n | best fixed | oracle | headroom | same-split | cross-split | selection bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| nq | 1157 | dense 0.1678 | 0.2121 | **+0.0444** (26%) | 0.1689 | 0.1666 | +0.0022 |
| triviaqa | 1539 | dense 0.2049 | 0.2659 | **+0.0611** (30%) | 0.2044 | 0.2054 | -0.0010 |
| hotpotqa | 916 | dense 0.1071 | 0.1706 | **+0.0636** (59%) | 0.1087 | 0.1045 | +0.0042 |

`same-split` picks the best fixed arm on one half of the queries and scores it on that same half; `cross-split` picks on one half and scores on the other. Their difference is how much of `best fixed` is a selection artifact. A small bias here means arm selection is stable; it does not by itself de-bias the per-query oracle, which cannot be cross-validated from a single reward draw.


## Table 2 — Oracle under injected zero-mean noise

| dataset | +0% sd | +5% sd | +10% sd | +25% sd | +50% sd | +100% sd | inflation slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| nq | 0.0444 | 0.0498 | 0.0573 | 0.0832 | 0.1323 | 0.2357 | **+431%** |
| triviaqa | 0.0611 | 0.0674 | 0.0758 | 0.1053 | 0.1591 | 0.2753 | **+351%** |
| hotpotqa | 0.0636 | 0.0679 | 0.0739 | 0.0954 | 0.1354 | 0.2199 | **+246%** |

Headroom after adding noise of the stated fraction of the reward table's own standard deviation. Real per-query differences are insensitive to small noise; headroom built on near-ties inflates steeply, because the max has more noise to select from. The slope is the increase in headroom from 0% to 100% sd.


**Read this as a sensitivity, not a proof of inflation.** A max over three arms inflates under *any* added noise, so a positive slope is expected and is not by itself evidence that the reported headroom is an artifact. What the column licenses is a conditional: if the reward table carries measurement noise of a given size — and it does, since which passage the answer string matches depends on the distractor draw — then that column says how much of the headroom that noise accounts for. Quote the +10% column, not the +100% one.


## Table 3 — Which queries the headroom comes from

| dataset | margin <= 1% | margin <= 5% | margin <= 10% | margin <= 25% |
|---|---:|---:|---:|---:|
| nq | 19.2% (337q) | 20.7% (586q) | 22.5% (702q) | 26.7% (840q) |
| triviaqa | 22.2% (505q) | 23.4% (831q) | 24.4% (960q) | 28.2% (1133q) |
| hotpotqa | 15.6% (203q) | 16.4% (402q) | 17.4% (513q) | 19.7% (620q) |

Share of total oracle headroom contributed by queries where the winning retriever beat the runner-up by no more than the stated fraction of the best-fixed mean. Headroom concentrated in tiny margins is headroom decided by measurement noise rather than by retriever quality — a router cannot learn to predict it because there is no stable signal to predict.

