"""Tests for the opt-in decodability test (leakcheck.audit_v2, leakcheck.decode).

Synthetic ground truth only, so these always run. The properties that matter:
  * v1 behaviour is unchanged by v2 (same findings, same verdicts);
  * a recoding v1 cannot see is found;
  * independent features stay clean (the false-positive guard);
  * a pure group confound is labelled as one.
"""

import numpy as np
import pytest

import leakcheck
from leakcheck import DECODABLE, decode, make_synthetic


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic(n=1200, n_arms=3, n_features=10, seed=3)


def _discrete_rewards(n, seed=0):
    """Reciprocal-rank-like rewards with few distinct values, like MRR@20."""
    rng = np.random.default_rng(seed)
    ranks = rng.integers(1, 30, size=(n, 3))
    return np.where(ranks <= 20, 1.0 / ranks, 0.0)


def test_v2_leaves_v1_findings_unchanged(synthetic):
    X, R = synthetic
    X = leakcheck.inject(X, R, leakcheck.DIRECT)
    v1, v2 = leakcheck.audit(X, R), leakcheck.audit_v2(X, R, n_perm=50)
    assert v1.leaked == v2.leaked
    assert v1.leaked_features == v2.leaked_features
    v1_pairs = {(f.feature, f.arm, f.rung) for f in v1.findings}
    assert v1_pairs <= {(f.feature, f.arm, f.rung) for f in v2.findings}


def test_clean_features_stay_clean(synthetic):
    X, R = synthetic
    rep = leakcheck.audit_v2(X, R, n_perm=100)
    assert not rep.leaked
    assert not [f for f in rep.findings if f.rung == DECODABLE], str(rep)


def test_order_scrambling_recoding_found_by_v2_only():
    n = 2000
    R = _discrete_rewards(n)
    X = np.random.default_rng(1).standard_normal((n, 6))
    uniq, inv = np.unique(R[:, 0], return_inverse=True)
    # A value-keyed recoding whose order carries almost nothing: pick the first permutation of
    # the distinct values that leaves |Spearman rho| under v1's suspicion band.
    for s in range(200):
        col = (np.random.default_rng(s).permutation(len(uniq)) * 0.37 + 5.0)[inv]
        if leakcheck.detectors.rank_agreement(col, R[:, 0])[0] < 0.2:
            break
    Xr = np.hstack([X, col[:, None]])

    v1 = leakcheck.audit(Xr, R, check_proxies=False)
    assert not [f for f in v1.findings if f.feature == 6 and f.arm == 0], "v1 should be blind here"

    v2 = leakcheck.audit_v2(Xr, R, n_perm=100, check_proxies=False)
    hits = [f for f in v2.findings if f.feature == 6 and f.arm == 0]
    assert hits and hits[0].rung == DECODABLE and not hits[0].is_verdict
    assert hits[0].statistic > 0.95  # a lookup decodes a discrete recoding exactly


def test_group_confound_is_labelled():
    n = 3000
    rng = np.random.default_rng(4)
    groups = rng.integers(0, 4, size=n)
    R = np.column_stack([0.2 * groups + rng.normal(0, 0.05, n) for _ in range(3)])
    feature = groups + rng.normal(0, 0.01, n)          # identifies the group, nothing more
    X = np.column_stack([rng.standard_normal(n), feature])
    rep = leakcheck.audit_v2(X, R, groups=groups, n_perm=50, check_proxies=False)
    f = [x for x in rep.findings if x.feature == 1]
    assert f, str(rep)
    assert all(x.within_group_gain < decode.MIN_R2 for x in f)
    assert "GROUP CONFOUND" in f[0].detail


def test_decodability_p_value_floor():
    R = _discrete_rewards(500)
    out = decode.decodability(R[:, 0] * 2.0 + 1.0, R[:, 0], n_perm=20)
    assert out["r2"] > 0.99
    assert out["p_value"] == pytest.approx(1 / 21)
