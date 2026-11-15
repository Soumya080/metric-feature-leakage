"""The five rungs of reward-into-feature leakage, and synthesizers for each.

A router (or any contextual bandit trained from logged data) picks an arm from a context
vector and is scored by a reward. Leakage is any case where the context contains information
about the reward it is about to be scored by. It is easy to introduce and hard to see, because
the features that *ought* to be most predictive of "which arm is best" occupy the same space
as the metric that defines "best".

The rungs below are ordered by how much of the reward survives into the feature, which is the
opposite of how hard they are to detect: the top rung is the most damaging and the easiest to
catch, the bottom rungs are nearly as damaging and evade exact-match detection entirely.

    1. DIRECT             feature == reward for some arm.
                          The textbook case. An exact-equality check finds it.

    2. MONOTONE_DERIVED   feature == g(reward) for monotone g -- a rescaling, a rank, a
                          difference of two arms' rewards, a log. Carries the full ordering,
                          so a tree-based model exploits it as effectively as the direct
                          case, but equality checks see nothing.

    3. JITTERED           feature == reward + noise. The important rung: on the RL router
                          data, 15% multiplicative jitter retained almost all of the
                          advantage of an exact leak (72.7% vs 73.2% accuracy, against a
                          61.4% majority baseline). A leak does not need to be clean to
                          invert a conclusion, which is precisely what makes an accidental
                          one dangerous.

    4. SINGLE_ARM         only one arm's reward leaks. Still worth 5+ accuracy points on
                          three arms, because knowing one arm's reward exactly tells you
                          when NOT to pick it.

    5. CORRELATED_PROXY   feature is not derived from the reward but is strongly predictive
                          of it through a shared upstream cause -- e.g. a feature computed
                          from the same ranked list the metric is computed from. The hardest
                          rung, and the one where "is this leakage?" becomes a judgement
                          call rather than a fact.

Rungs 1-4 are mechanical: given the reward table, they can be detected. Rung 5 cannot be
settled by a detector, only flagged for a human -- so `leakcheck` reports it as a suspicion
with the evidence attached, never as a verdict.

The synthesizers exist so the effect can be reproduced from nothing, with no dependence on
any particular corpus. `examples/leak_demo.py` uses them.
"""

from __future__ import annotations

import zlib

import numpy as np

DIRECT = "direct"
MONOTONE_DERIVED = "monotone_derived"
JITTERED = "jittered"
SINGLE_ARM = "single_arm"
CORRELATED_PROXY = "correlated_proxy"
#: Not a synthesizer rung: the finding type of the opt-in decodability test (audit_v2). A
#: feature from which an arm's reward can be recovered on held-out rows, whatever the mapping.
DECODABLE = "decodable"

#: Taxonomy order -- rungs 1-5 as described above, ordered by how much of the reward survives.
RUNGS = (DIRECT, MONOTONE_DERIVED, JITTERED, SINGLE_ARM, CORRELATED_PROXY)

#: Severity order, which is NOT the taxonomy order. A single-arm leak is an exact copy of one
#: reward column and belongs next to DIRECT when ranking findings, even though it is listed
#: fourth in the taxonomy (where the ordering is by how much of the reward is present, not by
#: how confident a detector can be). Used for sorting reports.
SEVERITY = (DIRECT, SINGLE_ARM, MONOTONE_DERIVED, JITTERED, DECODABLE, CORRELATED_PROXY)

DESCRIPTIONS = {
    DIRECT: "feature equals an arm's reward exactly",
    MONOTONE_DERIVED: "feature is a monotone function of an arm's reward",
    JITTERED: "feature is an arm's reward plus noise",
    SINGLE_ARM: "exactly one arm's reward is present in the features",
    CORRELATED_PROXY: "feature is not derived from the reward but strongly predicts it",
    DECODABLE: "an arm's reward is recoverable from the feature on held-out rows (any mapping)",
}


def _deterministic_jitter(values, scale, salt="leakcheck"):
    """Reproducible per-value noise, derived from the value itself rather than from an RNG.

    Using crc32 of the value means the same input produces the same jitter on every machine
    and in every process, so a demo of this rung is byte-identical everywhere. That matters
    for a teaching artifact: a reader who gets different numbers than the text cannot tell
    whether they have found a bug or a seed difference.
    """
    out = np.empty_like(values, dtype=float)
    for i, v in enumerate(np.asarray(values, dtype=float).ravel()):
        h = zlib.crc32(f"{salt}:{float(v)!r}".encode("utf-8"))
        out.flat[i] = v + ((h / 2 ** 32) - 0.5) * 2.0 * scale
    return out


def make_synthetic(n=2000, n_arms=3, n_features=20, best_arm_share=0.60, seed=0):
    """A routing problem in which the best arm is genuinely unpredictable from the features.

    This is the control the demo needs: features drawn independently of the labels, so any
    accuracy above the majority-class rate can only come from leakage that the demo itself
    introduces. Rewards are built so that one arm wins `best_arm_share` of the time, matching
    the class imbalance real routing data has.

    Returns (X, R): features (n, n_features) and rewards (n, n_arms).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))

    # Choose the winning arm per query independently of X -- this is what makes the target
    # unlearnable by construction, which is the whole point of a control.
    p = np.full(n_arms, (1.0 - best_arm_share) / (n_arms - 1))
    p[0] = best_arm_share
    winner = rng.choice(n_arms, size=n, p=p)

    # Rewards resemble reciprocal-rank data: mostly small, heavy zero mass, winner higher.
    R = rng.gamma(shape=1.2, scale=0.06, size=(n, n_arms))
    R[rng.random((n, n_arms)) < 0.45] = 0.0
    R[np.arange(n), winner] += rng.gamma(shape=2.0, scale=0.05, size=n) + 0.02
    return X, R


def inject(X, R, rung, arm=0, columns=None, noise_scale=0.15, seed=0):
    """Return a copy of X with the named rung of leakage introduced.

    `columns` selects which feature columns to overwrite; by default the last one or two.
    The reward table R is never modified -- a leak is a change to the features only, and
    keeping R fixed is what makes accuracy across rungs comparable.
    """
    X = np.asarray(X, dtype=float).copy()
    R = np.asarray(R, dtype=float)
    n_arms = R.shape[1]

    if rung == DIRECT:
        # EVERY arm's reward, which is what makes this rung 1 -- the whole reward table is
        # present in the context. Leaking only one arm is the SINGLE_ARM rung and is
        # materially weaker, so the two must not be synthesized identically.
        cols = (columns if columns is not None
                else list(range(X.shape[1] - n_arms, X.shape[1])))
        for j, a in zip(cols, range(n_arms)):
            X[:, j] = R[:, a]

    elif rung == MONOTONE_DERIVED:
        # A rank transform: carries the complete ordering of one arm's reward while sharing
        # no VALUES with it, so exact matching is structurally blind to it while a rank
        # correlation sees it perfectly.
        #
        # A difference of two arms' rewards would be the more natural illustration, but it is
        # a bad one: wherever the second arm's reward is zero -- which for zero-inflated
        # retrieval metrics is most rows -- the difference equals the first arm's reward
        # exactly, so the exact-match detector fires and the case is indistinguishable from a
        # direct leak. Ranks have no such degenerate overlap.
        cols = columns if columns is not None else [X.shape[1] - 1]
        order = np.argsort(np.argsort(R[:, 0], kind="mergesort"), kind="mergesort")
        X[:, cols[0]] = order.astype(float)

    elif rung == JITTERED:
        # Same columns and same arms as DIRECT, deliberately: the only difference between the
        # two rungs is the noise. That makes DIRECT vs JITTERED a clean isolation of "how much
        # does the leak lose when it stops being an exact copy?", which is the single most
        # important number in the taxonomy. Leaking a different number of arms here would
        # confound noise with coverage.
        cols = (columns if columns is not None
                else list(range(X.shape[1] - n_arms, X.shape[1])))
        for j, a in zip(cols, range(n_arms)):
            X[:, j] = _deterministic_jitter(R[:, a], noise_scale)

    elif rung == SINGLE_ARM:
        cols = columns if columns is not None else [X.shape[1] - 1]
        X[:, cols[0]] = R[:, arm]

    elif rung == CORRELATED_PROXY:
        # Shares an upstream cause with the reward without being derived from it: a noisy
        # observation of the winning arm's identity, which is what a feature computed from
        # the same ranked list as the metric looks like in practice.
        rng = np.random.default_rng(seed)
        cols = columns if columns is not None else [X.shape[1] - 1]
        X[:, cols[0]] = R.argmax(axis=1) + rng.normal(0.0, 0.8, size=len(R))

    else:
        raise ValueError(f"unknown rung {rung!r}; expected one of {RUNGS}")

    return X
