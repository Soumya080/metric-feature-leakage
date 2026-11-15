# Within-dataset permutation null

`summary.json` was produced by `experiments/analyses/permutation_within_dataset.py` with:
- GB defaults and 5×5 CV, CV seed 0
- labels shuffled within each dataset
- `default_rng(5000 + b)`, B = 60

Rerunning the script into a new `--out-dir` regenerates these numbers.

| | Accuracy (%) |
|---|---|
| Observed (pooled, honest 61-D, n = 3,175) | **59.87**, matching `perm_B1000` |
| Within-dataset null (B = 60) | **59.18 ± 0.23**, max 59.65 |
| Across-dataset null (`perm_B1000`, B = 1000) | 59.15 ± 0.24 |

All 60 within-dataset draws fall below the observed value. That gives p = 1/61 = 0.016, which is
the smallest p this B can produce; for comparison, the observed value sits about 3 SD above the
null.

**Reading.** Keeping the link between dataset and class prior moves the null centre by 0.03
points. So the pooled signal in Figure 2 is not dataset identity, and the paper's sentence "the
honest features do carry signal" stands. The paper does not quote this p-value: B = 60 caps it
at 0.016, while the published pooled test is at B = 1000.
