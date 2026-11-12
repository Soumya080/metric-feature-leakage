# Calibration

Every threshold in `leakcheck` comes from measured separation on real data, not from
intuition. This file records the measurements so the numbers can be re-derived, disputed, or
recalibrated against a different corpus.

## The data

Three QA datasets (NQ, TriviaQA, HotpotQA), 1,157 / 1,539 / 916 queries, three retrieval arms
(BM25, Dense, Hybrid-RRF) over 500-passage corpora mined from a 100,000-passage Wikipedia
pool. Rewards are untruncated reciprocal rank of the first answer-bearing passage.

Five feature variants of the same corpus, built by the same pipeline, differing only in what
was leaked into feature columns 22-25:

| variant | what was leaked |
|---|---|
| `honest` | nothing — retrieval-score features built from raw label-free scores |
| `leaky` | BM25 and Dense metric values, exactly |
| `one-arm` | BM25's metric value only |
| `hit-based` | Hit@k instead of MRR — a coarser monotone function of the same thing |
| `jittered` | BM25 and Dense metric values with a value-keyed offset of up to ±0.15 (a deterministic recoding, not independent noise; the paper calls this rung "Recoded") |

This is an unusually good calibration set: the ground truth is known by construction, the
honest and leaked variants are otherwise identical, and the leak arms span four of the five
taxonomy rungs.

## Exact-match threshold — `EXACT_MATCH_THRESHOLD = 0.25`

Share of non-zero-reward rows on which a feature column equals a reward column.

| variant | worst match rate |
|---|---|
| `leaky` | 30.8% – 44.2% |
| `one-arm` | 27.4% – 32.9% |
| `honest` | 0% (≤ 6.1% before the two false-positive guards) |

The honest residue before the guards was small-integer and small-rational collision — a
lexical count colliding with an integer gold rank, and a top-5 overlap fraction (multiples of
0.2) colliding with 1/rank at rank 5. The two guards in `detectors.exact_match` (score only
non-zero-reward rows; require ≥ 2 distinct agreeing values) remove it entirely.

Note the leaked variants sit at 31–44%, not 100%. The leak arms were built from a *truncated*
metric (MRR@20) while the training reward is untruncated MRR, so the two disagree wherever the
gold passage ranks below 20. **A leak of a related-but-not-identical metric is still
devastating** — this one was worth +11.8 accuracy points — which is exactly why the rank-based
detector below is necessary and why 0.25 is set well below the observed leak floor.

## Rank-correlation bands — `0.35` suspect, `0.50` strong (both are suspicions, never verdicts)

Maximum |Spearman rho| over every (feature, arm) pair, audited against both the training
reward and the truncated metric the leaks were built from:

| variant | vs training reward | vs MRR@20 |
|---|---|---|
| `honest` | **0.118 – 0.229** | **0.194 – 0.315** |
| `leaky` | 0.818 – 0.909 | 1.000 |
| `one-arm` | 0.786 – 0.835 | 1.000 |
| `hit-based` | 0.800 – 0.861 | 0.438 – 0.582 |
| `jittered` | 0.525 – 0.689 | 0.839 – 0.917 |

Audited against the training reward, honest features top out at **0.229** and the weakest true
leak is **0.525** (jittered, HotpotQA). A cut at **0.50** separates every variant
perfectly with 0.27 of margin below it. The suspicion band starts at **0.35**, above the 0.315
honest maximum measured against the truncated metric.

### Why rank correlation is never a verdict — the falsifying case

An earlier version of this file used 0.50 as a **verdict** cut, on the strength of the
separation above. That was wrong, and external data proved it.

Audited against RouterBench (36,497 queries × 11 LLMs, 86 benchmarks), features derived from
the **prompt text alone** — character count, word count, punctuation counts, which cannot
possibly encode a model's correctness score — reach:

| prompt-only feature | max &#124;rho&#124; vs a model's reward | arm selectivity |
|---|---:|---:|
| `n_question_marks` | **0.881** | 0.014 |
| `n_digits` | 0.812 | 0.001 |
| `n_upper` | 0.755 | 0.029 |
| `n_chars` | 0.458 | — |

The association is real, not a tie artifact. The cause is that
RouterBench pools 86 benchmarks whose prompts differ in format and difficulty, so "number of
question marks" is effectively a *benchmark identifier*, and benchmark identity strongly
predicts which model wins.

**A confound through a shared upstream cause produces exactly the signature we were calling a
monotone leak** — high rank correlation, zero exact agreement. No threshold can separate them,
because the difference is causal and the data is correlational.

So the design changed structurally rather than numerically:

* **Only exact agreement yields a verdict.** It is a fact about the arrays.
* **Everything rank-based is a suspicion**, carrying its evidence, resolved by ablation.

### Arm selectivity — reported, not thresholded

A leak of arm *a*'s reward is specific to arm *a*; a confound correlates with every arm at once.
Measured:

| | selectivity |
|---|---:|
| RouterBench confounds | 0.001 – 0.031 |
| RL-Route honest | 0.033 |
| RL-Route real leaks | 0.053 – 0.078 |

It separates — but only just, because RL-Route's own arms are correlated (Hybrid is RRF of the
other two), which suppresses selectivity for genuine leaks. The margin between 0.033 and 0.053
is too thin to carry a decision, so selectivity is reported as supporting evidence for a human
and never used as a threshold.

## Proxy threshold — `ARGMAX_LIFT_THRESHOLD = 0.10`

Accuracy of recovering the winning arm from one quantile-binned feature, minus the
majority-class rate. Not calibrated against measured separation, because the rung it targets
is by definition not mechanically decidable — it is a screening sensitivity, chosen so that a
feature carrying a tenth of the label's information gets surfaced for a human to judge.
Findings at this rung are never verdicts.

## Validation result

With these thresholds, auditing all fifteen (variant × dataset) combinations against the
training reward:

* all three `honest` combinations: **no verdicts and no suspicions**
* all twelve leaked combinations: **findings** on feature columns 22–25; the exact-copy
  variants (`leaky`, `one-arm`) are **verdicts**, while `hit-based` and `jittered` are
  **suspicions**, because rank evidence cannot support a verdict (see the falsifying case above)
* the `one-arm` variant is correctly distinguished as `single_arm` rather than `direct`

Reproduce with `python -m pytest leakcheck/tests/ -k calibration`, or against your own env
data by pointing `RLROUTE_ENV_DIR` at it.

## Recalibrating for your own data

Do not simply lower the thresholds until your leak is caught. Measure separation first:

1. Build an honest variant and a deliberately-leaked variant of your feature matrix.
2. Compute max |rho| over all (feature, arm) pairs for each.
3. Put the verdict cut in the gap. If there is no gap, your features and your reward are too
   entangled for a static threshold and the only sound test is an ablation.

Step 3's failure case is real and is not a defect in this tool. It is the honest answer for
setups where the routing signal genuinely lives in the same space as the metric.
