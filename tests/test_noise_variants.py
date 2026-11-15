"""Unit tests for the helpers in experiments/analyses/. Synthetic data only;
nothing here reads or writes the real env files or results."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "analyses"))

import common as C  # noqa: E402
from build_noise_variants import build_entry  # noqa: E402
from oracle_gap_capture import capture  # noqa: E402


def _entry():
    feats = [float(i) for i in range(61)]
    rets = {"bm25": {"mrr_at_20": 0.0, "rank_reward": 1 / 37, "mrr": 1 / 37},
            "dense": {"mrr_at_20": 1 / 3, "rank_reward": 1 / 3, "mrr": 1 / 3},
            "hybrid": {"mrr_at_20": 0.5, "rank_reward": 0.5, "mrr": 0.5}}
    return {"features": feats, "retrievers": rets}


def test_jitter_matches_repo_rule_at_paper_scale():
    from src.router.prepare_env_data import LEAK_NOISE_SCALE, _jitter
    for v in (0.0, 1 / 37, 1 / 3, 0.5, 1.0):
        for salt in ("bm25", "dense"):
            assert C.jitter(v, salt, LEAK_NOISE_SCALE) == _jitter(v, salt)


def test_jitter_zero_scale_is_identity_and_bounded_otherwise():
    for v in (0.0, 0.2, 1.0):
        assert C.jitter(v, "bm25", 0.0) == v
        for s in C.NOISE_SCALES:
            assert abs(C.jitter(v, "dense", s) - v) <= s + 1e-15


def test_build_entry_slot_contract_and_untouched_features():
    e = _entry()
    out = build_entry(e, "rankreward", 0.0, clipped=False)
    f = out["features"]
    assert f[22] == 1 / 37 and f[23] == 1 / 3 and f[24] == f[23] - f[22]
    assert f[25] == 25.0 and f[:22] == e["features"][:22] and f[26:] == e["features"][26:]
    assert out["retrievers"] == e["retrievers"]
    assert e["features"][22] == 22.0  # source entry not mutated
    m = build_entry(e, "mrr20", 0.0, clipped=False)["features"]
    assert m[22] == 0.0 and m[23] == 1 / 3  # truncated metric differs from reward for bm25


def test_clipped_variant_stays_in_unit_interval():
    f = build_entry(_entry(), "rankreward", 1.0, clipped=True)["features"]
    assert 0.0 <= f[22] <= 1.0 and 0.0 <= f[23] <= 1.0


def test_variant_tags_are_unique_and_complete():
    tags = C.all_variant_tags()
    assert len(tags) == len(set(tags)) == 2 * len(C.NOISE_SCALES) + 2
    assert "rankreward_s000" in tags and "mrr20_s015" in tags and "rankreward_s015_clip" in tags


def test_prepare_out_dir_refuses_non_empty_and_protected(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    (d / "x.txt").write_text("keep")
    with pytest.raises(SystemExit):
        C.prepare_out_dir(str(d))
    assert (d / "x.txt").read_text() == "keep"
    with pytest.raises(SystemExit):
        C.prepare_out_dir(C.ROUTER_DIR)


def test_r5_noise_is_per_query_not_per_value():
    import numpy as np
    from random_noise_sweep import leak_matrix, noise
    n = 200
    u = noise(n, draw=0)
    assert u.shape == (n, 2) and np.all(np.abs(u) <= 1.0)
    assert np.array_equal(u, noise(n, draw=0))            # reproducible
    assert not np.array_equal(u, noise(n, draw=1))        # draws differ
    X = np.zeros((n, 61))
    vals = np.full((n, 2), 0.5)                           # every query has the SAME value
    Xl = leak_matrix(X, vals, 0.15, u)
    assert len(np.unique(Xl[:, 22])) == n                 # ...but gets its own offset
    assert np.all(np.abs(Xl[:, 22] - 0.5) <= 0.15 + 1e-12)
    assert np.allclose(Xl[:, 24], Xl[:, 23] - Xl[:, 22])
    assert np.all(Xl[:, 25] == 0.0) and np.all(X[:, 22] == 0.0)  # slot 25 untouched, input not mutated
    assert np.array_equal(leak_matrix(X, vals, 0.0, u)[:, 22], vals[:, 0])


def test_capture_formula():
    assert capture(0.15, 0.10, 0.20) == pytest.approx(0.5)
    assert capture(0.05, 0.10, 0.20) == pytest.approx(-0.5)
    assert capture(0.10, 0.10, 0.10) is None
