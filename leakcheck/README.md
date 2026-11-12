# leakcheck

Find the evaluation metric hiding in your router's input features.

If you train a router, a model selector, or any contextual bandit from logged evaluation data,
you can accidentally hand the model a function of its own reward as context. It is a one-line
mistake. It is invisible in how papers describe features — *"we use retrieval score
statistics"* describes both the clean version and the leaked one, so a reader cannot tell which
you did. And it inverts conclusions: on the corpus this package was extracted from, **four
leaked columns out of sixty-one moved a router from below a constant baseline to apparently
working.**

```python
import leakcheck

report = leakcheck.audit(X, R)     # X: (n, d) features, R: (n, k) per-arm rewards
print(report)
assert not report.leaked
```

No model to fit, no train/test split, no dependency beyond numpy.

## See it in about three minutes

```
python examples/leak_demo.py
```

Builds a routing problem where the best arm is drawn **independently of the features**, so the
target is unlearnable by construction and anything above the majority-class rate is leakage the
demo introduced on purpose:

| feature set | accuracy | vs majority | verdict |
|---|---|---|---|
| honest (no leak) | 50.7% | −2.3% | clean |
| direct | 96.6% | +43.7% | LEAK direct |
| single arm | 66.0% | +13.1% | LEAK single_arm |
| monotone derived | 65.8% | +12.9% | suspect monotone_derived |
| jittered | 78.2% | +25.2% | suspect monotone_derived |
| correlated proxy | 67.4% | +14.5% | suspect monotone_derived |

`direct` and `jittered` leak the same arms into the same columns — the only difference is
noise. **The leak retains 58% of its advantage after being jittered.** That is the finding: a
leak does not need to be a clean copy of the reward to invert a conclusion, and a noised copy
is what an accident actually produces. An exact copy is trivial to catch with an equality
check; a noised one is invisible to equality checks and still does most of the damage.

## The five rungs

| rung | what it is | reported as |
|---|---|---|
| 1 · `direct` | feature **equals** an arm's reward | **verdict** |
| 2 · `monotone_derived` | feature is a monotone function of a reward (rank, rescale, log) | suspicion |
| 3 · `jittered` | feature is a reward plus noise, or a value-keyed recoding of it (the paper's "Recoded" rung) | suspicion |
| 4 · `single_arm` | exactly one arm's reward leaked, exactly | **verdict** |
| 5 · `correlated_proxy` | not derived from the reward but strongly predicts it | suspicion |

**Only exact agreement is a verdict.** Everything else is a suspicion carrying its evidence.

That split was forced by data, not chosen out of caution. An earlier version called a verdict on
strong rank correlation. Run against RouterBench (36,497 queries × 11 LLMs, 86 benchmarks), it
flagged features built from **prompt text alone** — which cannot encode a model's correctness
score — at |rho| = 0.881. The association is real, not a tie artifact: RouterBench pools
86 benchmarks with different prompt formats, so "number of question marks" is effectively a
benchmark identifier, and benchmark identity predicts which model wins.

A confound through a shared upstream cause produces **exactly** the signature of a monotone
leak — high rank correlation, zero exact agreement. The difference between them is causal; the
data is correlational. No threshold can separate them, so the tool does not pretend to.

Each rank-based finding therefore reports **arm selectivity**: how much more the feature
correlates with its top arm than with the next one. A leak of one arm's reward is arm-specific;
a confound correlates with everything at once. Measured: RouterBench confounds 0.001–0.031,
real leaks 0.053–0.078. It separates, but too narrowly to carry a decision — so it is evidence
for you, not a threshold in the code.

## Optional: the decodability test (`audit_v2`, since 0.2.0)

`audit` checks whether a column EQUALS a reward (verdict) or ORDERS rows like one (suspicion).
A recoding that keeps each value's identity but scrambles its order passes both, while a model
can still decode it: on the RL-Route corpora, a value-keyed offset at +/-50% raised no v1
finding while the leak still bought 12.3 accuracy points. `audit_v2` adds the question a model
actually answers:

```python
report = leakcheck.audit_v2(X, R, groups=dataset_id, n_perm=200)   # v1 findings unchanged
```

For each (column, arm) it fits a held-out lookup (exact values if the column has <= 64 of them,
else quantile bins) and reports the held-out R^2 of the reward, a permutation p-value
(Benjamini-Hochberg at 5% across the tested pairs), and, if `groups` is given, how much the
column adds WITHIN groups. A column that only identifies the group (the RouterBench confound)
is labelled `GROUP CONFOUND`. Findings are `DECODABLE` suspicions, never verdicts: a genuinely
predictive feature is decodable too. Measured behaviour (`experiments/crc_v2/h_detector_benchmark.py`):

| | v1 | v2 |
|---|---|---|
| Synthetic leaks flagged (9 types x 10 seeds) | 98% (order-scrambled recoding 80%) | 100% |
| Planted group confound labelled as a confound | -- | 10 / 10 |
| Pure-noise columns flagged | 0% | 0% |
| Genuine-signal columns flagged | 74% | 91% |
| RL-Route value-keyed leak at +/-50% / +/-100% | not flagged | flagged |
| RL-Route honest overlap features 26-28 | not flagged | flagged (real signal, not a leak) |

## What a clean report does and does not mean

`report.leaked == False` means no feature *exactly* encodes any arm's reward. It is evidence,
not proof — a noised or monotone leak will show up only as a suspicion.

What "exactly" means, precisely (0.2.1):

- A verdict needs agreement (relative tolerance 1e-6, so float32 copies count) on **more than
  25% of the rows where the reward is non-zero**, with at least two distinct agreeing values.
  Honest columns on the RL-Route corpora reach at most 6.1%; the paper's real MRR@20 leak
  reaches 26–45% against the untruncated reward. An exact copy of a metric truncated more
  tightly than the reward (e.g. MRR@5 against full-rank reciprocal rank) can fall below 25% and
  is then only a suspicion.
- For a **binary** reward (e.g. per-model correctness in LLM routing) a verdict needs the
  feature to equal the reward on **every** row. Anything short of a full copy is left to the
  rank and decodability tests (`audit_v2`).

**Check `report.suspect_features`, and resolve by ablation:** retrain without those columns and
see whether the result survives. That is the only conclusive test, and no amount of statistics
substitutes for it.

## Thresholds

Measured on real corpora with known ground truth, not chosen by intuition. Honest features top
out at |rho| = 0.229 against the training reward; the weakest true leak is 0.525; RouterBench
confounds reach 0.881. Full measurements, the falsifying case, and how to recalibrate against
your own data: [CALIBRATION.md](CALIBRATION.md).

## Use it as a regression test

The most valuable place for this is a test that runs on every commit, so a leak cannot come
back after you fix it:

```python
def test_no_reward_leak():
    X, R = load_my_features(), load_my_rewards()
    report = leakcheck.audit(X, R)
    assert not report.leaked, str(report)
```

Pair it with a **positive control** — audit a deliberately-leaked variant of your features and
assert the detector *does* fire. A test that passes on both clean and leaked data proves
nothing, and its green result is worse than no test at all.

## Provenance

Extracted from the RL-Route project, where the leak described above was found, quantified, and
fixed. The graded leak arms that calibrate these thresholds exist because the original
finding needed a dose–response, and they turned out to be an unusually good calibration set:
same pipeline, same corpus, ground truth known by construction.
