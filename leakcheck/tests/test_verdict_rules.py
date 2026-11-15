"""Regression tests for the float-tolerance and binary-reward verdict rules (0.2.1).

1. An exact copy of a binary (0/1) reward used to get no verdict: every non-zero row carries
   the same value, so the distinct-value guard could never pass.
2. An exact copy stored as float32 used to lose its verdict to an absolute 1e-12 tolerance.
"""
import numpy as np

import leakcheck


def _honest(n, d, rng):
    return rng.normal(size=(n, d))


def test_exact_copy_of_binary_reward_is_a_verdict():
    rng = np.random.default_rng(0)
    n = 2000
    R = (rng.random((n, 4)) < 0.6).astype(float)
    X = np.hstack([_honest(n, 5, rng), R[:, [1]]])
    report = leakcheck.audit(X, R)
    assert report.leaked
    assert any(f.feature == 5 and f.arm == 1 for f in report.verdicts)


def test_independent_binary_flag_is_not_a_verdict():
    rng = np.random.default_rng(1)
    n = 2000
    R = (rng.random((n, 4)) < 0.6).astype(float)
    flag = (rng.random((n, 1)) < 0.6).astype(float)
    X = np.hstack([_honest(n, 5, rng), flag])
    assert not leakcheck.audit(X, R).leaked


def test_near_copy_of_binary_reward_is_not_a_verdict():
    rng = np.random.default_rng(2)
    n = 2000
    R = (rng.random((n, 2)) < 0.5).astype(float)
    col = R[:, 0].copy()
    col[:5] = 1.0 - col[:5]
    X = np.hstack([_honest(n, 3, rng), col[:, None]])
    assert not leakcheck.audit(X, R).leaked


def test_float32_exact_copy_keeps_its_verdict():
    rng = np.random.default_rng(3)
    n = 2000
    ranks = rng.integers(1, 50, size=(n, 3))
    R = 1.0 / ranks
    X = np.hstack([_honest(n, 5, rng), R[:, [0, 1]]])
    X32 = X.astype(np.float32).astype(float)
    assert leakcheck.audit(X32, R).leaked
