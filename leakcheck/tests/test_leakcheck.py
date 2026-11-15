"""Tests for leakcheck: synthetic ground truth, plus calibration against real corpora.

The synthetic tests are the ones that must always pass -- they construct leakage deliberately
and assert it is found, and construct clean data and assert it is not. The calibration tests
skip when the RL router env data is not present, since that data is a generated artifact that
does not live in every working tree.
"""

import glob
import json
import os

import numpy as np
import pytest

import leakcheck
from leakcheck import (CORRELATED_PROXY, DIRECT, JITTERED, MONOTONE_DERIVED, SINGLE_ARM,
                       inject, make_synthetic)

ACTIONS = ["bm25", "dense", "hybrid"]
ENV_DIR = os.environ.get("ENV_DATA_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "router")


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic(n=600, n_arms=3, n_features=12, seed=0)


def test_clean_synthetic_is_clean(synthetic):
    """Features drawn independently of the labels must produce no verdict.

    This is the false-positive test, and it is the one that matters most: a detector that
    fires on clean data is worse than no detector, because it trains people to ignore it.
    """
    X, R = synthetic
    report = leakcheck.audit(X, R)
    assert not report.leaked, f"false positive on independent features:\n{report}"
    assert report.leaked_features == []


@pytest.mark.parametrize("rung", [DIRECT, SINGLE_ARM, MONOTONE_DERIVED, JITTERED,
                                  CORRELATED_PROXY])
def test_every_rung_is_surfaced(synthetic, rung):
    """Every rung must produce at least one finding. Detection is separate from verdicts."""
    X, R = synthetic
    report = leakcheck.audit(inject(X, R, rung), R)
    assert report.findings, f"{rung} produced no findings at all:\n{report}"


@pytest.mark.parametrize("rung", [DIRECT, SINGLE_ARM])
def test_exact_rungs_yield_verdicts(synthetic, rung):
    """Exact agreement is a fact about the arrays, so it may be called."""
    X, R = synthetic
    report = leakcheck.audit(inject(X, R, rung), R)
    assert report.leaked, f"{rung} should be a verdict:\n{report}"


@pytest.mark.parametrize("rung", [MONOTONE_DERIVED, CORRELATED_PROXY])
def test_inferred_rungs_are_never_verdicts(synthetic, rung):
    """Rank-based and proxy findings must be surfaced but never called.

    Forced by external data: on RouterBench, prompt-length features reach |rho| = 0.88 against
    a model's correctness score, because they identify which benchmark a query came from and
    benchmark identity predicts which model wins. A confound through a shared upstream cause is
    indistinguishable from a monotone leak by correlation alone, so calling a verdict here would
    be a claim the arrays cannot support. This test pins that design decision.
    """
    X, R = synthetic
    report = leakcheck.audit(inject(X, R, rung), R)
    assert report.suspicions, f"{rung} not surfaced:\n{report}"
    assert not report.leaked, (
        f"{rung} was reported as a verdict, but rank/proxy evidence cannot support one:\n"
        f"{report}")


def test_rank_findings_carry_selectivity(synthetic):
    """A rank finding without selectivity is not interpretable, so it must always be present."""
    X, R = synthetic
    report = leakcheck.audit(inject(X, R, MONOTONE_DERIVED), R)
    rank_findings = [f for f in report.findings if f.rung in (MONOTONE_DERIVED, JITTERED)]
    assert rank_findings, f"no rank-based findings to check:\n{report}"
    assert all(not np.isnan(f.selectivity) for f in rank_findings)


def test_direct_leak_names_the_right_columns(synthetic):
    """The report must point at the columns that were actually overwritten."""
    X, R = synthetic
    injected_cols = [X.shape[1] - 3, X.shape[1] - 2, X.shape[1] - 1]
    report = leakcheck.audit(inject(X, R, DIRECT), R)
    assert set(injected_cols).issubset(set(report.leaked_features)), (
        f"expected {injected_cols} among leaked columns, got {report.leaked_features}")


def test_exact_match_ignores_shared_zero_mass():
    """Two columns of zeros must not read as agreement.

    Retrieval metrics are zero on most queries and several legitimate structural features are
    near-constant zero, so a naive equality check reports ~70% agreement between them. This
    was a real false positive found by running an earlier detector against known-clean data.
    """
    n = 200
    R = np.zeros((n, 2))
    R[:20, 0] = np.linspace(0.1, 1.0, 20)   # reward non-zero on 10% of rows
    X = np.zeros((n, 1))                     # a legitimately all-zero feature
    report = leakcheck.audit(X, R, check_proxies=False)
    assert not report.leaked, f"fired on shared zero mass:\n{report}"


def test_exact_match_requires_two_distinct_values():
    """Agreement on a single repeated value is a collision, not a leak."""
    n = 200
    R = np.full((n, 2), 0.5)
    X = np.full((n, 1), 0.5)   # matches everywhere, but carries no variation
    report = leakcheck.audit(X, R, check_proxies=False)
    assert not report.leaked, f"fired on a constant column:\n{report}"


def test_jitter_is_deterministic_across_calls(synthetic):
    """The JITTERED rung must be reproducible, so a teaching demo is stable everywhere."""
    X, R = synthetic
    a = inject(X, R, JITTERED)
    b = inject(X, R, JITTERED)
    np.testing.assert_array_equal(a, b)


def test_shape_validation():
    with pytest.raises(ValueError):
        leakcheck.audit(np.zeros((10, 3)), np.zeros((9, 2)))
    with pytest.raises(ValueError):
        leakcheck.audit(np.zeros(10), np.zeros((10, 2)))


# ---------------------------------------------------------------------------------------
# Calibration against the real corpora. Skipped when the env data is absent.
# ---------------------------------------------------------------------------------------

def _load_env(path, metric="mrr"):
    with open(path, "r", encoding="utf-8") as f:
        env = json.load(f)
    queries = env.get("queries", env)
    X = np.array([v["features"] for v in queries.values()], dtype=float)
    R = np.array([[float(v["retrievers"][a][metric]) for a in ACTIONS]
                  for v in queries.values()], dtype=float)
    return X, R


def _env_paths(suffix):
    return sorted(p for p in glob.glob(os.path.join(ENV_DIR, f"*_env_data{suffix}.json"))
                  if "_layout" not in p and (suffix or "_" not in
                                             os.path.basename(p)[len("xx_env_data"):]))


@pytest.mark.parametrize("suffix", ["_leaky", "_leak1", "_leakhit", "_leaknoisy"])
def test_calibration_real_leak_arms_are_surfaced(suffix):
    """Every known leak arm must produce findings, on the columns that were leaked (22-25)."""
    paths = _env_paths(suffix)
    if not paths:
        pytest.skip(f"no *_env_data{suffix}.json under {ENV_DIR}")
    for path in paths:
        X, R = _load_env(path)
        report = leakcheck.audit(X, R, arm_names=ACTIONS, check_proxies=False)
        assert report.findings, (
            f"{os.path.basename(path)} is a known leak arm but produced nothing:\n{report}")
        flagged = set(report.leaked_features) | set(report.suspect_features)
        assert flagged & {22, 23, 24, 25}, (
            f"{os.path.basename(path)} flagged {sorted(flagged)}, none of which are the "
            f"retrieval-score columns 22-25 that were actually leaked:\n{report}")


@pytest.mark.parametrize("suffix", ["_leaky", "_leak1"])
def test_calibration_exact_leak_arms_yield_verdicts(suffix):
    """The arms that copied a metric value verbatim are exact, so they may be called.

    Deliberately does NOT include _leaknoisy or _leakhit. Those leak through noise and through
    a coarser monotone function respectively, so they produce rank evidence rather than exact
    agreement -- and rank evidence cannot support a verdict. They are covered by the
    surfaced-only test above. Pinning that distinction here is the point: the tool must not
    quietly regain confidence it is not entitled to.
    """
    paths = _env_paths(suffix)
    if not paths:
        pytest.skip(f"no *_env_data{suffix}.json under {ENV_DIR}")
    for path in paths:
        X, R = _load_env(path)
        report = leakcheck.audit(X, R, arm_names=ACTIONS, check_proxies=False)
        assert report.leaked, f"{os.path.basename(path)} should be a verdict:\n{report}"


def test_calibration_real_honest_arms_are_clean():
    paths = [p for p in glob.glob(os.path.join(ENV_DIR, "*_env_data.json"))
             if "_layout" not in p]
    if not paths:
        pytest.skip(f"no honest env data under {ENV_DIR}")
    for path in paths:
        X, R = _load_env(path)
        report = leakcheck.audit(X, R, arm_names=ACTIONS, check_proxies=False)
        assert not report.findings, (
            f"{os.path.basename(path)} is the honest arm but produced findings -- either the "
            f"thresholds have drifted or a leak has been reintroduced:\n{report}")
