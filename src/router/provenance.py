"""Make every results artifact say which experiment produced it.

The problem this solves, concretely. After the 2026-08-04 run, "NQ, Always-Dense" appeared
across the repository as four different numbers:

    0.2163   canonical_results.json        real-Wikipedia-filtered env, metric 'mrr'
    0.2072   canonical_results_all.json    all gold types,              metric 'mrr'
    0.2265   gold_type_stratified.md       real-Wikipedia SUBSET of the all-gold split
    0.1807   cost_aware_analysis.md        real-Wikipedia-filtered env, metric 'mrr_at_20'

None of those is wrong. They answer different questions. But nothing in the files says so,
so a reader -- including the authors, and including a careful reader who has all four open
at once -- cannot tell which is which. Two axes vary silently:

    dataset variant   all gold types / real-Wikipedia-filtered / real subset of all
    metric            'mrr' (untruncated 1/rank) / 'mrr_at_20' (zero past rank 20)

A paper assembled from these will contain numbers that contradict each other, which is the
defect that made the previous draft indefensible. The fix is not to pick one combination
and delete the rest -- the variants are all legitimate -- but to make each artifact state
which combination it is.

Usage:

    from src.router.provenance import stamp
    payload = stamp(payload, env_path=ds_path, metric=REWARD_TYPE, split_seed=SPLIT_SEED)
    json.dump(payload, f, indent=2)

Then `verify_provenance.py` can check that any two artifacts sharing a variant and metric
agree on the quantities they both report.
"""

import json
import os
import subprocess
from datetime import datetime, timezone

# Reward keys that appear in env data, and what each one actually is. Recorded alongside
# the metric name because "mrr" in this codebase is not MRR@5 -- a reader who assumes the
# textbook definition will misread every table.
METRIC_SEMANTICS = {
    "mrr": "reciprocal rank of the first answer-bearing passage, untruncated over the "
           "full per-query corpus",
    "mrr_at_20": "as 'mrr' but zero when the gold falls past rank 20",
    "rank_reward": "alias of 'mrr' as written by prepare_env_data.py --reward rank",
    "hit_at_5": "1 if an answer-bearing passage appears in the top 5, else 0",
}

# Filename suffix -> which feature construction the env data used. The suffix is the only
# record of this; nothing is embedded in the env JSON itself.
VARIANT_SUFFIXES = {
    "_leaky": ("leaky", "retrieval-score features built from metrics_at_k MRR -- "
                        "the reward for actions 0 and 1. Deliberate control, not a bug."),
    "_all": ("honest_minimal", "all gold types, including template-constructed passages"),
    "_xscore": ("honest_minimal", "honest features plus learned cross-retriever scoring"),
    "": ("honest_minimal", "retrieval-score features built from raw label-free scores"),
}


def _git_commit(repo_root):
    """Short commit of the code that produced the artifact, or None outside a checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=repo_root, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _classify_env(env_path):
    """(variant_suffix, feature_mode, description) inferred from the env-data filename."""
    base = os.path.basename(env_path)
    stem = base[:-len(".json")] if base.endswith(".json") else base
    for suffix, (mode, desc) in VARIANT_SUFFIXES.items():
        if suffix and stem.endswith(f"_env_data{suffix}"):
            return suffix, mode, desc
    mode, desc = VARIANT_SUFFIXES[""]
    return "", mode, desc


def _env_facts(env_path):
    """Query count and feature dimensions, read from the env data and its layout sidecar.

    Counted rather than assumed: the gold-type filter changes n between runs, and n is the
    single most useful number for telling two artifacts apart at a glance.
    """
    facts = {"n_queries": None, "feature_dim": None, "layout": None}
    if not env_path or not os.path.exists(env_path):
        return facts
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            env = json.load(f)
        queries = env.get("queries", env)
        facts["n_queries"] = len(queries)
        first = next(iter(queries.values()), None)
        if isinstance(first, dict) and isinstance(first.get("features"), list):
            facts["feature_dim"] = len(first["features"])
    except (OSError, ValueError, StopIteration):
        pass

    layout_path = env_path.replace(".json", "_layout.json")
    if os.path.exists(layout_path):
        try:
            with open(layout_path, "r", encoding="utf-8") as f:
                facts["layout"] = json.load(f)
        except (OSError, ValueError):
            pass
    return facts


def build(env_path=None, metric=None, split_seed=None, gold_type=None,
          pool_size=None, run_tag=None, repo_root=None, extra=None):
    """Assemble the provenance block for one artifact.

    Every field is either measured from the inputs or passed by the caller -- nothing is
    defaulted to a value that might be wrong, because a confidently wrong provenance block
    is worse than an absent one.
    """
    repo_root = repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    suffix, feature_mode, variant_desc = _classify_env(env_path or "")
    facts = _env_facts(env_path)

    block = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(repo_root),
        "dataset_variant": {
            "env_data": os.path.basename(env_path) if env_path else None,
            "suffix": suffix or "(none)",
            "feature_mode": feature_mode,
            "description": variant_desc,
            "gold_type": gold_type,
            "n_queries": facts["n_queries"],
            "feature_dim": facts["feature_dim"],
            "wiki_pool_size": pool_size,
        },
        "metric": {
            "name": metric,
            "means": METRIC_SEMANTICS.get(metric, "unrecognised metric key"),
        },
        "split_seed": split_seed,
        "run_tag": run_tag,
    }
    if facts["layout"]:
        block["dataset_variant"]["feature_layout"] = facts["layout"]
    if extra:
        block.update(extra)
    return block


def stamp(payload, **kwargs):
    """Attach provenance to a results payload in place, and return it.

    `_provenance` sorts to the front of most JSON viewers and is skipped by the `k.startswith("_")`
    guards the analysis scripts already use, so existing readers are unaffected.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"stamp() expects a dict payload, got {type(payload).__name__}")
    payload["_provenance"] = build(**kwargs)
    return payload


def summarize(block):
    """One-line human-readable form, for markdown headers and console output."""
    if not block:
        return "(no provenance)"
    dv = block.get("dataset_variant", {})
    parts = [
        dv.get("env_data") or "?",
        f"n={dv.get('n_queries')}" if dv.get("n_queries") else None,
        f"gold={dv.get('gold_type')}" if dv.get("gold_type") else None,
        f"metric={(block.get('metric') or {}).get('name')}",
        f"pool={dv.get('wiki_pool_size')}" if dv.get("wiki_pool_size") else None,
        f"split_seed={block.get('split_seed')}" if block.get("split_seed") is not None else None,
    ]
    # ASCII separator on purpose: this string gets printed to Windows consoles on the lab
    # machine, where a non-cp1252 character raises UnicodeEncodeError mid-pipeline.
    return " | ".join(p for p in parts if p)
