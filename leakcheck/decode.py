"""Decodability: how much of an arm's reward can be RECOVERED from one feature, held out.

Why this exists. The v1 detectors ask two questions: does a feature EQUAL a reward (exact
match), and does it ORDER rows like a reward (Spearman). A recoding that keeps each value's
identity but scrambles its order answers "no" to both while leaving the reward perfectly
recoverable. The paper measured exactly that blind spot: under a value-keyed offset at +/-50%,
leakcheck v1 raised no suspicion while gradient boosting still gained 12.3 accuracy points.

Decodability asks the question a model actually answers: given only this column, how well can
the reward be predicted on rows not used to build the predictor? The predictor is a lookup:

  * a column with few distinct values (<= max_codes) is looked up value by value, so ANY
    recoding of a discrete reward -- monotone or not -- is decoded exactly;
  * a continuous column is cut into quantile bins and each bin predicts its mean reward.

Score: held-out R^2 of that lookup against the reward, over K folds. Significance: the same
score with the reward column permuted, B times, giving p = (1 + k) / (1 + B). Optional
`groups` (e.g. dataset or benchmark id) adds the within-group score: the gain of
code-within-group over group alone. A feature that only identifies the group -- the
RouterBench confound -- scores high overall and near zero within groups.

Like every rank-based finding, decodability is a SUSPICION, never a verdict: a genuinely good
routing feature is also decodable. What it adds over v1 is (a) recodings and truncations are
no longer invisible, (b) a calibrated p-value instead of a fixed cutoff, and (c) a direct test
for the confound that forced v1's verdict/suspicion split. NumPy only.
"""

from __future__ import annotations

import numpy as np

MAX_CODES = 64        # columns with at most this many distinct values are looked up exactly
N_BINS = 20           # quantile bins for continuous columns
N_SPLITS = 5
MIN_R2 = 0.02         # below this a pair is not worth a permutation test


def encode(col, max_codes=MAX_CODES, n_bins=N_BINS):
    """Integer codes for a feature column: exact values if few, else quantile bins."""
    col = np.asarray(col, dtype=float)
    uniq, inv = np.unique(col, return_inverse=True)
    if len(uniq) <= max_codes:
        return inv
    edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.digitize(col, edges)


def _folds(n, n_splits, seed):
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, n_splits)


def _heldout_r2(codes, y, folds, groups=None):
    """Held-out R^2 of a per-code mean lookup (optionally within groups) against y."""
    n_codes = int(codes.max()) + 1
    sse = sst = 0.0
    for test in folds:
        train = np.ones(len(y), dtype=bool)
        train[test] = False
        mu = y[train].mean()
        if groups is None:
            s = np.bincount(codes[train], weights=y[train], minlength=n_codes)
            c = np.bincount(codes[train], minlength=n_codes)
            pred = np.where(c[codes[test]] > 0, s[codes[test]] / np.maximum(c[codes[test]], 1), mu)
        else:
            key = groups * n_codes + codes
            n_keys = int(key.max()) + 1
            s = np.bincount(key[train], weights=y[train], minlength=n_keys)
            c = np.bincount(key[train], minlength=n_keys)
            gs = np.bincount(groups[train], weights=y[train], minlength=int(groups.max()) + 1)
            gc = np.bincount(groups[train], minlength=int(groups.max()) + 1)
            gmean = np.where(gc > 0, gs / np.maximum(gc, 1), mu)
            kt = key[test]
            pred = np.where(c[kt] > 0, s[kt] / np.maximum(c[kt], 1), gmean[groups[test]])
        sse += float(((y[test] - pred) ** 2).sum())
        sst += float(((y[test] - mu) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else 0.0


def decodability(feature_col, reward_col, groups=None, n_perm=0, n_splits=N_SPLITS, seed=0):
    """Held-out R^2 of reward from one feature, with optional permutation p and group check.

    Returns dict(r2, p_value, n_perm, r2_group_only, r2_within_group_gain).
    p_value is NaN when n_perm == 0; the group fields are NaN when groups is None.
    """
    y = np.asarray(reward_col, dtype=float)
    codes = encode(feature_col)
    folds = _folds(len(y), n_splits, seed)
    r2 = _heldout_r2(codes, y, folds)
    out = {"r2": r2, "p_value": float("nan"), "n_perm": int(n_perm),
           "r2_group_only": float("nan"), "r2_within_group_gain": float("nan")}
    if n_perm:
        rng = np.random.default_rng([seed, 7])
        k = sum(_heldout_r2(codes, y[rng.permutation(len(y))], folds) >= r2 for _ in range(n_perm))
        out["p_value"] = (1 + k) / (1 + n_perm)
    if groups is not None:
        g = np.unique(np.asarray(groups), return_inverse=True)[1]
        zero = np.zeros(len(y), dtype=int)
        r2_g = _heldout_r2(zero, y, folds, groups=g)
        out["r2_group_only"] = r2_g
        out["r2_within_group_gain"] = _heldout_r2(codes, y, folds, groups=g) - r2_g
    return out
