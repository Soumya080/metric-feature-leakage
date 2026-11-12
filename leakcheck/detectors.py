"""Detectors for each mechanically-detectable rung of reward leakage.

Three detectors, in increasing order of what they can catch and of what they cost:

  exact_match     feature == reward, elementwise. Cheap, exact, zero false negatives on the
                  DIRECT rung and blind to every other rung.

  rank_agreement  Spearman correlation between a feature column and a reward column. Catches
                  MONOTONE_DERIVED and JITTERED, which exact matching cannot see. This is the
                  detector that matters: the rung it catches is nearly as damaging as a direct
                  copy and is the one an accident actually produces.

  argmax_recovery How well a single feature column recovers the identity of the winning arm,
                  as lift over the majority-class rate (no permutation null is computed;
                  the opt-in decode test in decode.py has one). Catches CORRELATED_PROXY. Reported as a
                  suspicion rather than a verdict, because a legitimately excellent
                  query-difficulty feature would also score here -- distinguishing "this
                  predicts the reward because it is the reward" from "because it is a good
                  feature" is not decidable from the data alone.

Two false-positive guards run in every detector, both learned the hard way on real data:

  1. Score only rows where the reward is non-zero. Retrieval metrics are zero on most
     queries (the gold document is not retrieved), and several legitimate structural features
     are near-constant zero -- so a naive equality check reports ~70% "agreement" between two
     columns of zeros.

  2. Require at least MIN_DISTINCT distinct agreeing values. A feature that tracks a reward
     carries its variation; agreement on a single repeated value is a collision, not a leak.

Thresholds are calibrated against measured separation on real data, not chosen by intuition.
See `leakcheck/CALIBRATION.md` for the numbers and the datasets they came from.
"""

from __future__ import annotations

import numpy as np

# Elementwise equality tolerance. Rewards are floats that survived a JSON round-trip, so
# exact == is too strict. The tolerance is relative as well as absolute: an absolute 1e-12 alone
# missed an exact copy stored as float32 (PyTorch's and many feature stores' default), whose
# rounding error is ~1e-8 relative. rtol=1e-6 is still far below any real feature's resolution.
EQ_TOL = 1e-12
EQ_RTOL = 1e-6

# Share of non-zero-reward rows that must agree before a DIRECT leak is called. Measured
# separation on the RL-Route corpora: honest arms <= 6.1% (small-integer collisions between
# lexical counts and integer gold ranks); the paper's real MRR@20 leak against the untruncated
# reward agrees on 26.3-44.8% of rows (it equals the reward only where the gold ranks in the top
# 20), and against MRR@20 itself on 100%. 0.25 sits above every honest arm, but an exact copy of
# a metric truncated much more tightly than the reward (e.g. MRR@5 against full-rank reciprocal
# rank) can agree on fewer rows and is then reported only as a suspicion.
EXACT_MATCH_THRESHOLD = 0.25

# |Spearman rho| bands for the monotone/jittered rungs, in TWO levels rather than one,
# because the measured separation does not support a single clean cut.
#
# Measured max |rho| over every (feature, arm) pair on the RL-Route corpora, against both the
# training reward (untruncated MRR) and the truncated MRR@20 the leak arms were built from:
#
#                     vs reward      vs MRR@20
#   honest            0.118-0.229    0.194-0.315     <- no leak present
#   leaky (exact)     0.818-0.909    1.000
#   single-arm        0.786-0.835    1.000
#   hit-based         0.800-0.861    0.438-0.582
#   jittered          0.525-0.689    0.839-0.917
#
# Audited against the training reward, honest features top out at 0.229 and the weakest true
# leak is 0.525. An earlier version of this file therefore set a VERDICT cut at 0.50.
#
# That was wrong, and running the detector on external public data proved it. On RouterBench
# (36,497 queries x 11 LLMs, 86 benchmarks), features derived from the prompt text alone --
# which cannot possibly encode a model's correctness score -- reach |rho| = 0.881 against a
# model's reward. The association is real, not a tie artifact: the
# benchmarks differ in prompt format and in difficulty, so a feature like "number of question
# marks" is effectively a benchmark identifier, and benchmark identity strongly predicts which
# model wins. A confound through a shared upstream cause produces exactly the signature we had
# been calling a monotone leak -- high rank correlation with zero exact agreement.
#
# The correct conclusion is structural, not a new number: RANK CORRELATION CAN NEVER BE A
# VERDICT. Only exact agreement can, because only exact agreement is a fact about the arrays
# rather than an inference about causation. Everything rank-based is reported as a suspicion
# with its evidence, and resolved by ablation.
#
# Both constants are retained: findings are reported above SUSPECT, and _THRESHOLD marks the
# band where a leak is more likely than a confound -- reported as stronger evidence, never as a
# verdict.
RANK_CORRELATION_THRESHOLD = 0.50
RANK_CORRELATION_SUSPECT = 0.35

# Arm selectivity: how much more a feature correlates with its top arm than with the next one.
# A leak of arm a's reward is specific to arm a; a confound correlates with every arm at once.
# Measured: RouterBench confounds 0.001-0.031, RL-Route honest 0.033, RL-Route real leaks
# 0.053-0.078. It separates, but only just, because RL-Route's own arms are correlated (Hybrid
# is RRF of the other two) which suppresses selectivity for genuine leaks. Reported as
# supporting evidence for a human, deliberately NOT used as a threshold -- the margin is too
# thin to carry a decision.
SELECTIVITY_SUGGESTS_LEAK = 0.04

# Accuracy lift over the majority-class rate, from one feature alone, above which a proxy is
# flagged for human review.
ARGMAX_LIFT_THRESHOLD = 0.10

MIN_DISTINCT = 2


def _nonzero_mask(reward_col):
    return np.asarray(reward_col, dtype=float) != 0.0


def _agrees(f, r):
    return np.isclose(f, r, rtol=EQ_RTOL, atol=EQ_TOL)


def exact_match(feature_col, reward_col):
    """Share of non-zero-reward rows where feature == reward, plus the distinct-value guard.

    Returns (rate, n_agree, n_eligible, n_distinct). Rate is 0.0 when the distinct-value
    guard fails, so a column of one repeated value can never trip the threshold.

    Binary rewards (at most two distinct values, e.g. per-model correctness in LLM routing)
    get a separate rule: the non-zero rows all carry the same value, so the distinct-value guard
    could never pass and an exact copy would go unflagged. For them the rate is 1.0 only when
    the feature equals the reward on EVERY row (an exact copy, a fact about the arrays), and
    0.0 otherwise; anything short of a full copy is left to the rank and decode tests.
    """
    f = np.asarray(feature_col, dtype=float)
    r = np.asarray(reward_col, dtype=float)
    fin = np.isfinite(r)
    if fin.any() and len(np.unique(r[fin])) == 2:
        agree_all = _agrees(f, r)
        n_agree = int(agree_all.sum())
        rate = 1.0 if n_agree == len(r) else 0.0
        return rate, n_agree, len(r), 2
    m = _nonzero_mask(r)
    n_elig = int(m.sum())
    if n_elig == 0:
        return 0.0, 0, 0, 0
    agree = _agrees(f[m], r[m])
    n_agree = int(agree.sum())
    n_distinct = int(len(np.unique(r[m][agree]))) if n_agree else 0
    if n_distinct < MIN_DISTINCT:
        return 0.0, n_agree, n_elig, n_distinct
    return n_agree / n_elig, n_agree, n_elig, n_distinct


def _spearman(a, b):
    """Spearman rho without a scipy dependency: Pearson on ranks, average ranks for ties."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3:
        return 0.0

    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        # Average ranks within tied groups, otherwise heavy ties (zero-inflated rewards)
        # distort rho by an amount that depends on input order.
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    ra, rb = ranks(a), ranks(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def rank_agreement(feature_col, reward_col):
    """|Spearman rho| between feature and reward, on non-zero-reward rows.

    Restricted to non-zero rows for the same reason as exact_match: the zero mass is large
    and shared, and including it manufactures correlation between any two zero-inflated
    columns regardless of whether they are related.
    """
    f = np.asarray(feature_col, dtype=float)
    r = np.asarray(reward_col, dtype=float)
    m = _nonzero_mask(r)
    if m.sum() < 3 or len(np.unique(f[m])) < MIN_DISTINCT:
        return 0.0, int(m.sum())
    return abs(_spearman(f[m], r[m])), int(m.sum())


def argmax_recovery(feature_col, R, n_bins=8):
    """Accuracy of predicting the winning arm from one feature alone, minus the majority rate.

    Deliberately model-free: bin the feature into quantiles and predict each bin's modal
    winner. A single decision-tree split is what a real model would find first, and binning
    approximates it without importing a learner or needing a train/test protocol. Restricted
    to rows with a strictly-best arm, since on ties there is no correct label.
    """
    f = np.asarray(feature_col, dtype=float)
    R = np.asarray(R, dtype=float)
    strict = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    if strict.sum() < n_bins * 2:
        return 0.0, 0.0, int(strict.sum())
    f, y = f[strict], R[strict].argmax(axis=1)
    majority = float(np.bincount(y).max() / len(y))

    edges = np.quantile(f, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    bins = np.digitize(f, edges)
    correct = 0
    for b in np.unique(bins):
        sel = bins == b
        if sel.sum():
            correct += int(np.bincount(y[sel]).max())
    acc = correct / len(y)
    return acc, acc - majority, int(strict.sum())
