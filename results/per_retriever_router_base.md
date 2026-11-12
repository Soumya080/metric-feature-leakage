# Per-Retriever Routing

One regressor per retriever predicting its own reward, routed by argmax, versus
the single 3-way argmax classifier it replaces. Reported metric is MRR@k on the
held-out test split; `p` is a paired bootstrap against Best Fixed.

| dataset | n_test | best fixed | oracle | method | MRR | Δ vs best-fixed | p |
|---|---:|---:|---:|---|---:|---:|---:|
| nq | 174 | 0.1570 | 0.1926 | per_retriever_gbt | 0.1477 | -0.0093 | 0.857 |
| nq | 174 | 0.1570 | 0.1926 | per_retriever_gbt_noembed | 0.1433 | -0.0137 | 0.939 |
| nq | 174 | 0.1570 | 0.1926 | per_retriever_ridge | 0.1483 | -0.0087 | 0.910 |
| nq | 174 | 0.1570 | 0.1926 | per_retriever_ridge_noembed | 0.1492 | -0.0078 | 0.831 |
| nq | 174 | 0.1570 | 0.1926 | argmax_classifier | 0.1487 | -0.0083 | 0.941 |
| triviaqa | 231 | 0.1925 | 0.2637 | per_retriever_gbt | 0.1716 | -0.0209 | 0.940 |
| triviaqa | 231 | 0.1925 | 0.2637 | per_retriever_gbt_noembed | 0.1785 | -0.0140 | 0.917 |
| triviaqa | 231 | 0.1925 | 0.2637 | per_retriever_ridge | 0.1906 | -0.0019 | 0.564 |
| triviaqa | 231 | 0.1925 | 0.2637 | per_retriever_ridge_noembed | 0.1889 | -0.0036 | 0.702 |
| triviaqa | 231 | 0.1925 | 0.2637 | argmax_classifier | 0.1955 | +0.0030 | 0.231 |
| hotpotqa | 138 | 0.0467 | 0.0999 | per_retriever_gbt | 0.0419 | -0.0048 | 0.736 |
| hotpotqa | 138 | 0.0467 | 0.0999 | per_retriever_gbt_noembed | 0.0430 | -0.0037 | 0.671 |
| hotpotqa | 138 | 0.0467 | 0.0999 | per_retriever_ridge | 0.0674 | +0.0207 | 0.036 |
| hotpotqa | 138 | 0.0467 | 0.0999 | per_retriever_ridge_noembed | 0.0708 | +0.0241 | 0.029 |
| hotpotqa | 138 | 0.0467 | 0.0999 | argmax_classifier | 0.0640 | +0.0173 | 0.085 |

A positive Δ with p < 0.05 means routing genuinely beats committing to the single best retriever. Δ ≤ 0 means it does not — which, stated plainly, is the result rather than something to tune away.

