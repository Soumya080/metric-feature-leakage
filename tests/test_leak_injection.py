"""The external leak-injection experiment must be a working instrument before it is trusted.

This does not test a claim about RouterBench -- it tests that the code would detect a leak if one
were present, and would not invent one if it were not. Both directions matter: an injection script
that silently produces a flat curve is indistinguishable from a genuine negative result, and a
genuine negative here would bound the paper's central claim. We must not be in a position where a
bug and a finding look the same.

Settings are reduced (1 repeat, small n) so the suite stays fast; the science runs use the module
defaults.
"""

import numpy as np
import pytest

from experiments.router_experiments import leak_injection_external as M


def _synthetic(n=500, k=3, d=8, seed=0, signal=False):
    """Honest features with no information about the winner, unless signal=True."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    R = rng.random((n, k))
    if signal:
        # Make arm 0 win exactly when the first feature is large, so honest features CAN predict.
        R[:, 0] += 2.0 * (X[:, 0] > 0)
    strict = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    X, R = X[strict], R[strict]
    return X, R, R.argmax(axis=1)


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(M, "N_REPEATS", 1)


def test_planted_leak_is_detected_and_ordered():
    """With no honest signal, each rung of leakage must beat the one below it."""
    X, R, y = _synthetic()
    arms = M.build_arms(X, R, k=2, seed=0)
    acc = {n: M.evaluate(Xa, y, seed=0, n_jobs=1)["acc"] for n, Xa in arms.items()}

    assert acc["honest"] < acc["leak_one"] < acc["leaky"], acc
    assert acc["leak_one"] < acc["leak_noisy"], acc
    # Honest features are random noise here, so they must not beat the majority class by much.
    maj = np.bincount(y).max() / len(y)
    assert acc["honest"] < maj + 0.06, f"honest={acc['honest']:.3f} vs majority={maj:.3f}"


def test_jitter_retains_most_of_the_premium():
    """The paper's headline: a noised leak keeps most of its value. If this inverts, say so."""
    X, R, y = _synthetic()
    arms = M.build_arms(X, R, k=2, seed=0)
    base = M.evaluate(arms["honest"], y, seed=0, n_jobs=1)["acc"]
    exact = M.evaluate(arms["leaky"], y, seed=0, n_jobs=1)["acc"]
    noisy = M.evaluate(arms["leak_noisy"], y, seed=0, n_jobs=1)["acc"]

    assert exact - base > 0.05, "no exact-leak premium to compare against"
    assert (noisy - base) / (exact - base) > 0.5


def test_guard_fires_on_an_exact_column():
    """A jittered arm that stayed exact would make the headline comparison vacuous."""
    X, R, _ = _synthetic()
    arms = M.build_arms(X, R, k=2, seed=0)
    M.assert_no_exact_leak(arms, R)          # the real jittered arm must pass

    arms["leak_noisy"] = np.hstack([X, R[:, :2]])
    with pytest.raises(SystemExit, match="not noisy"):
        M.assert_no_exact_leak(arms, R)


def test_no_leak_means_no_premium():
    """Honest features alone must not produce a premium -- otherwise the instrument invents one."""
    X, R, y = _synthetic()
    a = M.evaluate(X, y, seed=0, n_jobs=1)["acc"]
    b = M.evaluate(np.hstack([X, np.random.default_rng(1).standard_normal((len(X), 2))]),
                   y, seed=0, n_jobs=1)["acc"]
    assert abs(a - b) < 0.08, f"appending pure noise moved accuracy {a:.3f}->{b:.3f}"


def test_jitter_is_deterministic():
    """A reader must be able to regenerate the perturbed values to confirm no column is exact."""
    v = np.linspace(0.1, 1.0, 50).reshape(-1, 1)
    assert np.array_equal(M.jitter(v, seed=3), M.jitter(v, seed=3))
    assert not np.array_equal(M.jitter(v, seed=3), M.jitter(v, seed=4))
