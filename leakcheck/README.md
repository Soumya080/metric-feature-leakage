# leakcheck

Checks whether a router's input features encode its own reward.

If you train a router, model selector or contextual bandit from logged evaluation data, it is
easy to pass the model a function of the reward as context. Feature descriptions rarely reveal
it: "retrieval score statistics" describes both the clean features and the leaked ones. In the
setting this package comes from, four leaked columns out of 61 moved a router from below a
constant baseline to apparently beating every fixed retriever.

```python
import leakcheck

report = leakcheck.audit(X, R)     # X: (n, d) features, R: (n, k) per-arm rewards
print(report)
assert not report.leaked
```

The core audit needs only numpy: no model fitting and no train/test split.

Install from this repository with `pip install -e .` (Python 3.8+).

## Demo

```
python examples/leak_demo.py
```

The demo builds a routing problem where the best arm is drawn independently of the features,
so anything above the majority rate comes from the leak the script injects:

| feature set | accuracy | vs majority | report |
|---|---|---|---|
| honest | 50.7% | −2.3 | clean |
| direct | 96.6% | +43.7 | verdict: direct |
| single arm | 66.0% | +13.1 | verdict: single_arm |
| monotone derived | 65.8% | +12.9 | suspicion: monotone_derived |
| jittered | 78.2% | +25.2 | suspicion: monotone_derived |
| correlated proxy | 67.4% | +14.5 | suspicion: monotone_derived |

`direct` and `jittered` leak the same arms into the same columns and differ only by noise. The
jittered leak keeps 58% of the advantage and matches no reward value, so an equality check
misses it.

## Finding types

| type | meaning | reported as |
|---|---|---|
| `direct` | a feature equals an arm's reward | verdict |
| `monotone_derived` | a feature is a monotone function of a reward (rank, rescaling, log) | suspicion |
| `jittered` | a reward plus noise, or a value-keyed recoding of it | suspicion |
| `single_arm` | exactly one arm's reward is copied | verdict |
| `correlated_proxy` | not derived from the reward, but strongly predicts it | suspicion |

Only exact agreement produces a verdict. Everything else is a suspicion with its evidence
attached.

An earlier version also issued verdicts on strong rank correlation. On RouterBench (36,497
queries, 11 LLMs, 86 benchmarks) it flagged features computed from the prompt text alone, at
|ρ| = 0.881. The association is real: RouterBench pools benchmarks with different prompt
formats, so a feature such as the number of question marks identifies the benchmark, and the
benchmark predicts which model wins. A confound through a shared cause and a monotone leak
produce the same statistics (high rank correlation, no exact agreement). Telling them apart is
a causal question, so the tool does not attempt it.

Rank-based findings also report arm selectivity: how much more the feature correlates with its
top arm than with the next one. Measured values: RouterBench confounds 0.001–0.031, real leaks
0.053–0.078. The gap is real but too narrow to use as a threshold, so it is reported, not
acted on.

## Decodability test (`audit_v2`, since 0.2.0)

`audit` asks whether a column equals a reward or orders rows like one. A recoding that keeps
each value's identity but scrambles its order passes both checks, yet a model can still decode
it: on our data, a value-keyed offset of ±50% raised no finding while the leak still bought
12.3 accuracy points. `audit_v2` adds a held-out test:

```python
report = leakcheck.audit_v2(X, R, groups=dataset_id, n_perm=200)
```

For each (column, arm) pair it fits a held-out lookup (exact values for columns with at most 64
distinct values, quantile bins otherwise), reports the held-out R² of the reward and a
permutation p-value (Benjamini–Hochberg at 5%), and, if `groups` is given, how much the column
adds within groups. A column that only identifies the group is labelled a group confound.
Findings are suspicions, never verdicts, because genuinely predictive features are decodable
too. Results from `experiments/analyses/detector_benchmark.py`:

| | v1 | v2 |
|---|---|---|
| Synthetic leaks flagged (9 types × 10 seeds) | 98% (order-scrambled recoding 80%) | 100% |
| Planted group confound labelled as a confound | – | 10 / 10 |
| Pure-noise columns flagged | 0% | 0% |
| Genuine-signal columns flagged | 74% | 91% |
| Value-keyed leak at ±50% / ±100% | not flagged | flagged |
| Honest overlap features 26–28 | not flagged | flagged (real signal) |

## Reading a clean report

`report.leaked == False` means no feature exactly encodes any arm's reward. It is not proof of
absence: a noised or monotone leak shows up only as a suspicion.

The verdict rule (0.2.1):

- A feature must agree with a reward (relative tolerance 1e-6, so float32 copies count) on more
  than 25% of the rows where that reward is non-zero, with at least two distinct agreeing
  values. Honest columns on our data reach at most 6.1%; the MRR@20 leak reaches 26–45% against
  the untruncated reward. A copy of a metric truncated more tightly than the reward (for
  example MRR@5 against full-rank reciprocal rank) can fall below 25%.
- For a binary reward, such as per-model correctness in LLM routing, the feature must equal the
  reward on every row. Partial matches are left to the rank and decodability tests.

Check `report.suspect_features` and resolve them by ablation: retrain without those columns and
see whether the result holds.

## Thresholds

Set from data with known ground truth. Honest features reach at most |ρ| = 0.229 against the
training reward; the weakest true leak is 0.525; RouterBench confounds reach 0.881. The
measurements and how to recalibrate on your own data are in [CALIBRATION.md](CALIBRATION.md).

## As a regression test

```python
def test_no_reward_leak():
    X, R = load_my_features(), load_my_rewards()
    report = leakcheck.audit(X, R)
    assert not report.leaked, str(report)
```

Pair it with a positive control: audit a deliberately leaked copy of your features and assert
that the audit fires. A test that passes on both proves nothing. `tests/test_no_feature_reward_leak.py`
in this repository does both.

## Citation

Sawant and Joshi, *Metric-as-Feature Leakage: A Feature-Side Failure Mode in Retriever Routing*,
NewInML Workshop at NeurIPS 2026. BibTeX in the [repository README](../README.md#citation).
