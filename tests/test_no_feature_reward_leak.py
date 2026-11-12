"""Guard against the reward-leakage bug returning to the router's input features.

The original defect: `prepare_env_data.py` built `avg_bm25`/`avg_dense` (features 22 and
23) from each retriever's MRR. MRR is computed from the gold answer and is *literally the
reward* for actions 0 and 1, so two of three arms' rewards were being handed to the policy
as inputs. It was invisible because the field is named `avg_bm25`, and because stage 2
writes raw scores and evaluation metrics into the same per-query record -- reading the
wrong key is a one-line error with no type distinction between them.

These tests make the fix permanent. The paper's central claim is that this class of leak
inflates routing accuracy, so the honest pipeline must be provably free of it rather than
free of it by inspection.

Detection is by *systematic* agreement, not by a single coincidence: a feature that equals
some retriever's reward on one query in a thousand is a float collision, while one that
equals it on half the corpus is the bug. The leaky variant trips this at roughly 50%;
honest data sits at 0%.
"""

import glob
import json
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Env data is a generated artifact and is not always present in a working tree -- it may
# live on a results branch, or on the machine that ran the pipeline. RLROUTE_ENV_DIR lets
# the guard run against wherever it actually is, without copying artifacts into the repo.
ENV_DIR = os.environ.get("RLROUTE_ENV_DIR") or os.path.join(REPO_ROOT, "src", "router")

# Every per-retriever field that is, or is derived from, the gold answer. Any of these
# appearing as a feature is leakage regardless of which one the reward happens to use.
REWARD_KEYS = ("mrr", "mrr_at_20", "rank_reward", "hit_at_5", "gold_rank")

# Calibrated against the pool20k data rather than guessed. Measured separation:
#
#   *_env_data_leaky.json   100.0%  (features[22|23] == bm25|dense.mrr_at_20, 20 values)
#   triviaqa_env_data.json    6.1%  (features[6] == dense.gold_rank, 5 values)
#   nq_env_data.json          5.5%  (features[27] == hybrid.mrr_at_20, 3 values)
#   hotpotqa_env_data.json    4.6%  (features[27] == hybrid.mrr_at_20, 3 values)
#
# The honest residue is small-integer and small-rational coincidence: feature 6 is a
# lexical count colliding with an integer gold_rank, and feature 27 is a top-5 overlap
# fraction (multiples of 0.2) colliding with 1/rank at rank 5. A 16x gap separates that
# from a real leak, so 0.25 sits clear of both.
#
# Limitation worth knowing: this detects *exact* agreement. A leak passed through noise or
# a monotone transform would evade it, which is why the honest/leaky ablation -- not this
# test -- is the paper's evidence. This guards against the specific regression.
MAX_MATCH_RATE = 0.25


def _env_files(pattern):
    return sorted(glob.glob(os.path.join(ENV_DIR, pattern)))


def _worst_match_rate(env_path):
    """Highest share of queries on which any (feature, retriever, reward-key) triple agrees.

    Returns (rate, description). Scanning every triple rather than the historically
    affected indices 22/23 means a leak reintroduced at a different position is still
    caught.

    Two guards against false positives, both found by running this against known-clean
    data. `hit_at_5` and `mrr_at_20` are zero on most queries (the gold is not in the top
    k), and several structural features are near-constant zero -- so a naive equality
    check reports ~70% "agreement" between two columns of zeros. We therefore:

      1. count only queries where the reward is non-zero, and score the rate over those
      2. require the agreeing values to take at least two distinct values

    A feature that tracks a reward carries its *variation*; a column of zeros carries
    nothing and is indistinguishable from a legitimately-zero feature.
    """
    with open(env_path, "r", encoding="utf-8") as f:
        env = json.load(f)
    queries = env.get("queries", env)
    if not queries:
        return 0.0, "no queries"

    hits, eligible, seen_values = {}, {}, {}
    for entry in queries.values():
        features = entry.get("features") or []
        retrievers = entry.get("retrievers") or {}
        for r_name, metrics in retrievers.items():
            for key in REWARD_KEYS:
                if key not in metrics:
                    continue
                target = float(metrics[key])
                if target == 0.0:
                    continue
                for idx, value in enumerate(features):
                    triple = (idx, r_name, key)
                    eligible[triple] = eligible.get(triple, 0) + 1
                    if abs(float(value) - target) < 1e-12:
                        hits[triple] = hits.get(triple, 0) + 1
                        seen_values.setdefault(triple, set()).add(target)

    rates = {t: n / eligible[t] for t, n in hits.items()
             if eligible.get(t) and len(seen_values.get(t, ())) >= 2}
    if not rates:
        return 0.0, "no systematic matches"
    triple = max(rates, key=rates.get)
    idx, r_name, key = triple
    return rates[triple], (f"features[{idx}] == retrievers.{r_name}.{key} on "
                           f"{hits[triple]}/{eligible[triple]} non-zero-reward queries "
                           f"({len(seen_values[triple])} distinct values)")


def test_honest_env_data_has_no_reward_leak():
    """No feature in the shipped env data may encode any arm's reward."""
    paths = [p for p in _env_files("*_env_data.json") if "_leaky" not in p]
    if not paths:
        pytest.skip(f"No env data under {ENV_DIR} -- run prepare_env_data.py first.")

    offenders = []
    for path in paths:
        rate, detail = _worst_match_rate(path)
        if rate > MAX_MATCH_RATE:
            offenders.append(f"{os.path.basename(path)}: {rate:.1%} -- {detail}")

    assert not offenders, (
        "Reward leakage detected in features:\n  " + "\n  ".join(offenders) +
        "\n\nThe retrieval-score features must come from raw label-free scores "
        "(mean_score_top5 / std_score_top5), never from metrics_at_k. "
        "See prepare_env_data.py::_retrieval_features."
    )


def test_leak_detector_fires_on_the_leaky_variant():
    """The detector must have teeth.

    A test that passes on both honest and leaky data proves nothing. The `leaky` feature
    mode reproduces the original bug deliberately so the ablation can quantify it, which
    makes it the natural positive control: if this does *not* trip, the check above is
    vacuous and its green result is meaningless.
    """
    paths = _env_files("*_env_data_leaky.json")
    if not paths:
        pytest.skip("No leaky env data -- generate with prepare_env_data.py "
                    "--feature-mode leaky --suffix _leaky")

    for path in paths:
        rate, detail = _worst_match_rate(path)
        assert rate > MAX_MATCH_RATE, (
            f"{os.path.basename(path)} was generated with --feature-mode leaky but the "
            f"detector did not trip (worst rate {rate:.1%}). Either the leaky mode no "
            f"longer reproduces the original bug, or the detector is broken -- in both "
            f"cases the honest-data test above is not actually guarding anything. "
            f"Worst triple: {detail}"
        )
