# Headroom: how much is captured, and where it lives

## 1. Share of oracle headroom captured

`capture = (method - best_fixed) / (oracle - best_fixed)`. 100% is perfect routing; 0% means you may as well have committed to one retriever; negative is worse than that.

| method | nq (honest) | triviaqa (honest) | hotpotqa (honest) |
|---|---|---|---|
| Heuristic (score gap) | +0% | -5% | +0% |
| Logistic Regression | -5% | -1% | -35% |
| Random Forest | -4% | -5% | -0% |
| Ridge Regression | -18% | -3% | -1% |
| Gradient Boosting (tuned) | -4% | +0% | +0% |
| RL router | -42% | -17% | -21% |

Best-fixed and oracle per dataset:

| dataset | regime | best fixed | oracle | headroom |
|---|---|---:|---:|---:|
| nq | honest | dense 0.1655 | 0.2001 | +0.0346 |
| triviaqa | honest | dense 0.2000 | 0.2694 | +0.0693 |
| hotpotqa | honest | dense 0.0753 | 0.1096 | +0.0343 |

## 2. Where the headroom lives

The oracle-minus-best-fixed gap, split by the rank at which the winning retriever placed the gold passage. If the headroom sat in the deep ranks it would not matter much that no method captures it.

| dataset | rank 1-5 | rank 6-20 | rank 21-100 | rank >100 |
|---|---|---|---|---|
| nq | 85.2% | 12.3% | 2.2% | 0.3% |
| triviaqa | 90.8% | 7.4% | 1.6% | 0.2% |
| hotpotqa | 85.8% | 11.8% | 2.2% | 0.2% |

Ranks 1-5 hold 85-91% of the headroom across datasets — the regime a user would actually notice. The opportunity is real and concentrated where it matters, which is what makes failing to capture it a result rather than a technicality.
