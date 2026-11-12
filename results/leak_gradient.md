# Leakage Dose-Response: how much contamination does it take?

_Scale: full run, real_wikipedia gold. Accuracy predicting the strictly-best retriever, 5-fold stratified CV x 5 repeats, pooled over nq, triviaqa, hotpotqa._

Every arm differs from the others **only in features 22-25**. Reward tables are identical across arms (asserted, not assumed), so the accuracy difference is attributable to feature construction and nothing else.

| arm | what leaks | accuracy | lift vs majority | vs honest |
|---|---|---:|---:|---:|
| `honest_minimal` | raw top-5 score mean/std (label-free) | 59.9% ± 0.7 | **-1.0%** | _baseline_ |
| `leak_noisy` | MRR + deterministic jitter (evades exact-match detection) | 72.7% ± 1.2 | **+11.8%** | +12.8 pts |
| `leak_hit` | Hit@k instead of MRR (sign, not value) | 69.0% ± 1.1 | **+8.1%** | +9.1 pts |
| `leak_one_arm` | BM25's MRR only; Dense honest | 65.1% ± 1.3 | **+4.2%** | +5.2 pts |
| `leaky` | both arms' MRR -- the original bug | 73.1% ± 1.0 | **+12.2%** | +13.2 pts |

Majority-class baseline: **60.9%** (n = 3175 informative queries).

## What the noisy arm means

`leak_noisy` buys **+12.8 points** over honest features while matching no reward value exactly -- jitter breaks equality, so a check of the form "no feature equals any reward" does not see it. Checking for exact equality is therefore **not a sufficient defence**: a leak can be an imprecise proxy and still inflate results, which is what an accidental leak usually is.

Honest and leaky arms reproduce `results/learnability_analysis.json`: **yes**.
