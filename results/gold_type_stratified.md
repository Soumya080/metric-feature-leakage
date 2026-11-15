# Results Stratified by Gold-Passage Provenance

`constructed` = template-generated filler from `construct_gold()` (the answer string is restated verbatim in a short, keyword-dense passage). `real_wikipedia` = a genuine Wikipedia snippet that contains the answer. **Headline the `real_wikipedia` column** — it is the harder, honest condition.


## NQ

_Source: mode A (from saved per-query scores)._

Test split: 174 queries (174 real_wikipedia, 0 constructed = 0% synthetic)

| Method | All | real_wikipedia | constructed | real - constructed |
|---|---|---|---|---|
| Always BM25 | 0.1029 | 0.1029 | n/a | n/a |
| Always Dense | 0.1655 | 0.1655 | n/a | n/a |
| Always Hybrid | 0.1464 | 0.1464 | n/a | n/a |
| Heuristic (Score Gap) | 0.1657 | 0.1657 | n/a | n/a |
| Logistic Regression | 0.1639 | 0.1639 | n/a | n/a |
| Random Forest | 0.1640 | 0.1640 | n/a | n/a |
| Ridge Regression | 0.1594 | 0.1594 | n/a | n/a |
| Gradient Boosting (Tuned) | 0.1642 | 0.1642 | n/a | n/a |
| RL (REINFORCE) | 0.1512 | 0.1512 | n/a | n/a |
| Oracle | 0.2001 | 0.2001 | n/a | n/a |

**Significance on real_wikipedia only (n=174):** RL router vs. tuned GBT delta=-0.0130 (p=0.9136); RL router vs. Best-Fixed (dense) delta=-0.0144 (p=0.9303).

## TRIVIAQA

_Source: mode A (from saved per-query scores)._

Test split: 231 queries (231 real_wikipedia, 0 constructed = 0% synthetic)

| Method | All | real_wikipedia | constructed | real - constructed |
|---|---|---|---|---|
| Always BM25 | 0.1501 | 0.1501 | n/a | n/a |
| Always Dense | 0.2000 | 0.2000 | n/a | n/a |
| Always Hybrid | 0.1907 | 0.1907 | n/a | n/a |
| Heuristic (Score Gap) | 0.1968 | 0.1968 | n/a | n/a |
| Logistic Regression | 0.1990 | 0.1990 | n/a | n/a |
| Random Forest | 0.1968 | 0.1968 | n/a | n/a |
| Ridge Regression | 0.1981 | 0.1981 | n/a | n/a |
| Gradient Boosting (Tuned) | 0.2000 | 0.2000 | n/a | n/a |
| RL (REINFORCE) | 0.1881 | 0.1881 | n/a | n/a |
| Oracle | 0.2694 | 0.2694 | n/a | n/a |

**Significance on real_wikipedia only (n=231):** RL router vs. tuned GBT delta=-0.0119 (p=0.8781); RL router vs. Best-Fixed (dense) delta=-0.0119 (p=0.8781).

## HOTPOTQA

_Source: mode A (from saved per-query scores)._

Test split: 138 queries (138 real_wikipedia, 0 constructed = 0% synthetic)

| Method | All | real_wikipedia | constructed | real - constructed |
|---|---|---|---|---|
| Always BM25 | 0.0468 | 0.0468 | n/a | n/a |
| Always Dense | 0.0753 | 0.0753 | n/a | n/a |
| Always Hybrid | 0.0572 | 0.0572 | n/a | n/a |
| Heuristic (Score Gap) | 0.0753 | 0.0753 | n/a | n/a |
| Logistic Regression | 0.0635 | 0.0635 | n/a | n/a |
| Random Forest | 0.0753 | 0.0753 | n/a | n/a |
| Ridge Regression | 0.0750 | 0.0750 | n/a | n/a |
| Gradient Boosting (Tuned) | 0.0753 | 0.0753 | n/a | n/a |
| RL (REINFORCE) | 0.0683 | 0.0683 | n/a | n/a |
| Oracle | 0.1096 | 0.1096 | n/a | n/a |

**Significance on real_wikipedia only (n=138):** RL router vs. tuned GBT delta=-0.0070 (p=0.8353); RL router vs. Best-Fixed (dense) delta=-0.0070 (p=0.8353).
