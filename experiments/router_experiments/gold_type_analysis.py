#!/usr/bin/env python3
"""
gold_type_analysis.py - Stratify every routing method's MRR@5 by whether the query's gold
passage is a REAL Wikipedia snippet or a template-generated one.

Why this matters: 01_prepare_data.py falls back to construct_gold() when it
can't find a real Wikipedia passage containing the answer, emitting formulaic filler text
like "<answer> is widely recognized in this context. According to various sources...". On
NQ this is ~56% of queries. That text repeats the answer string verbatim in a short, clean,
keyword-dense passage — which is close to a best case for BM25 and changes what "retrieval
difficulty" even means. If the headline numbers are driven by the constructed half, a
reviewer who notices will discount the whole result.

The defensible move is not to hide this: report both strata, and headline the
real_wikipedia subset (the honest, harder, genuinely-Wikipedia condition). This script
produces exactly that table, plus a paired significance test on the real subset alone, so
"RL-Route still wins where it counts" is a tested claim rather than an assertion.

Two modes, chosen automatically:
  A. Preferred - results/canonical_results.json contains the `per_query` block written by
     canonical_evaluation.py. Every method (including RL-Route) is stratified with zero
     retraining, straight from the saved per-query scores.
  B. Fallback - no `per_query` block (older results file, or the run predates that change).
     Static/heuristic/supervised baselines are recomputed directly from env data, which is
     cheap; RL-Route is reported as unavailable, since reproducing it would need retraining.

Usage:
    python experiments/router_experiments/gold_type_analysis.py
    python experiments/router_experiments/gold_type_analysis.py --results path/to/canonical_results.json
"""
import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from src.router.routing_env import RouterEnvironment, create_train_val_test_split

DEFAULT_RESULTS = os.path.join(PROJECT_ROOT, "results", "canonical_results.json")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "results", "gold_type_stratified.json")
OUTPUT_MD = os.path.join(PROJECT_ROOT, "results", "gold_type_stratified.md")

# Display order / labels for the report.
METHOD_LABELS = [
    ("bm25", "Always BM25"),
    ("dense", "Always Dense"),
    ("hybrid", "Always Hybrid"),
    ("score_gap", "Heuristic (Score Gap)"),
    ("logistic_regression", "Logistic Regression"),
    ("random_forest", "Random Forest"),
    ("ridge", "Ridge Regression"),
    ("gradient_boosting", "Gradient Boosting (Tuned)"),
    ("rl_route", "RL-Route (Ours)"),
    ("oracle", "Oracle"),
]


def paired_bootstrap_test(scores_a, scores_b, n_resamples=10000, seed=42):
    """One-sided paired bootstrap: is A > B? Seeded for reproducibility."""
    rng = np.random.RandomState(seed)
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    diffs = a - b
    observed = diffs.mean()
    shifted = diffs - observed  # impose the null (mean difference = 0)
    idx = rng.randint(0, len(diffs), size=(n_resamples, len(diffs)))
    null_means = shifted[idx].mean(axis=1)
    p_value = float((null_means >= observed).mean())
    return float(observed), p_value


def stratify_from_per_query(ds_block):
    """Mode A: use the saved per-query scores + gold_type."""
    pq = ds_block["per_query"]
    gold = np.array(pq["gold_type"])
    out = {}
    for method, scores in pq["scores"].items():
        s = np.asarray(scores, dtype=float)
        out[method] = {
            "all": float(s.mean()),
            "real_wikipedia": float(s[gold == "real_wikipedia"].mean()) if (gold == "real_wikipedia").any() else None,
            "constructed": float(s[gold == "constructed"].mean()) if (gold == "constructed").any() else None,
        }
    counts = {
        "all": int(len(gold)),
        "real_wikipedia": int((gold == "real_wikipedia").sum()),
        "constructed": int((gold == "constructed").sum()),
    }
    return out, counts, pq


def stratify_fallback(ds_name, ds_path):
    """Mode B: recompute the non-RL methods directly from env data (no retraining)."""
    from canonical_evaluation import (
        SPLIT_SEED, REWARD_TYPE, compute_static_baselines,
        compute_heuristic_baseline, compute_supervised_baselines,
    )

    env_train, _env_val, env_test = create_train_val_test_split(ds_path, split_seed=SPLIT_SEED)
    _static_means, static_pq = compute_static_baselines(env_test, REWARD_TYPE)
    _h_mean, heuristic_pq = compute_heuristic_baseline(env_train, env_test, REWARD_TYPE)
    _sup_means, sup_pq = compute_supervised_baselines(env_train, env_test, REWARD_TYPE)

    scores = {**static_pq, "score_gap": heuristic_pq, **sup_pq}
    gold = np.array([env_test.get_gold_type(q) for q in env_test.query_pool])

    out = {}
    for method, s in scores.items():
        s = np.asarray(s, dtype=float)
        out[method] = {
            "all": float(s.mean()),
            "real_wikipedia": float(s[gold == "real_wikipedia"].mean()) if (gold == "real_wikipedia").any() else None,
            "constructed": float(s[gold == "constructed"].mean()) if (gold == "constructed").any() else None,
        }
    counts = {
        "all": int(len(gold)),
        "real_wikipedia": int((gold == "real_wikipedia").sum()),
        "constructed": int((gold == "constructed").sum()),
    }
    pq = {"gold_type": gold.tolist(), "scores": {k: list(map(float, v)) for k, v in scores.items()}}
    return out, counts, pq


def main():
    parser = argparse.ArgumentParser(description="Stratify routing results by gold-passage provenance.")
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    from canonical_evaluation import DATASETS

    results_data = {}
    if os.path.exists(args.results):
        with open(args.results) as f:
            results_data = json.load(f)
    else:
        print(f"NOTE: {args.results} not found — falling back to recomputing from env data.")

    report = {}
    lines = ["# Results Stratified by Gold-Passage Provenance\n"]
    lines.append(
        "`constructed` = template-generated filler from `construct_gold()` (the answer string is "
        "restated verbatim in a short, keyword-dense passage). `real_wikipedia` = a genuine "
        "Wikipedia snippet that contains the answer. **Headline the `real_wikipedia` column** — "
        "it is the harder, honest condition.\n"
    )

    for ds_name, ds_path in DATASETS.items():
        block = results_data.get(ds_name, {})
        if "per_query" in block:
            mode = "A (from saved per-query scores)"
            strat, counts, pq = stratify_from_per_query(block)
        else:
            if not os.path.exists(ds_path):
                print(f"SKIP {ds_name}: no env data at {ds_path}")
                continue
            mode = "B (recomputed; RL-Route unavailable without retraining)"
            print(f"[{ds_name}] no per_query block in results — recomputing non-RL methods...")
            strat, counts, pq = stratify_fallback(ds_name, ds_path)

        report[ds_name] = {"mode": mode, "counts": counts, "by_method": strat}

        lines.append(f"\n## {ds_name.upper()}\n")
        lines.append(f"_Source: mode {mode}._\n")
        lines.append(
            f"Test split: {counts['all']} queries "
            f"({counts['real_wikipedia']} real_wikipedia, {counts['constructed']} constructed "
            f"= {counts['constructed']/max(1,counts['all'])*100:.0f}% synthetic)\n"
        )
        lines.append("| Method | All | real_wikipedia | constructed | real - constructed |")
        lines.append("|---|---|---|---|---|")
        for key, label in METHOD_LABELS:
            if key not in strat:
                continue
            r = strat[key]
            rw = r["real_wikipedia"]
            cs = r["constructed"]
            gap = (rw - cs) if (rw is not None and cs is not None) else None
            lines.append(
                f"| {label} | {r['all']:.4f} | "
                f"{'n/a' if rw is None else f'{rw:.4f}'} | "
                f"{'n/a' if cs is None else f'{cs:.4f}'} | "
                f"{'n/a' if gap is None else f'{gap:+.4f}'} |"
            )

        # Significance on the real_wikipedia subset alone — the claim that actually matters.
        if "rl_route" in pq["scores"] and "gradient_boosting" in pq["scores"]:
            gold = np.array(pq["gold_type"])
            mask = gold == "real_wikipedia"
            if mask.sum() > 10:
                rl = np.asarray(pq["scores"]["rl_route"], dtype=float)[mask]
                gbt = np.asarray(pq["scores"]["gradient_boosting"], dtype=float)[mask]
                best_fixed_key = max(("bm25", "dense", "hybrid"),
                                     key=lambda k: np.asarray(pq["scores"][k], dtype=float)[mask].mean())
                bf = np.asarray(pq["scores"][best_fixed_key], dtype=float)[mask]

                d_gbt, p_gbt = paired_bootstrap_test(rl, gbt)
                d_bf, p_bf = paired_bootstrap_test(rl, bf)
                report[ds_name]["significance_real_wikipedia_only"] = {
                    "n": int(mask.sum()),
                    "rl_vs_gbt": {"delta": d_gbt, "p_value": p_gbt},
                    "rl_vs_best_fixed": {"method": best_fixed_key, "delta": d_bf, "p_value": p_bf},
                }
                lines.append(
                    f"\n**Significance on real_wikipedia only (n={int(mask.sum())}):** "
                    f"RL-Route vs. tuned GBT delta={d_gbt:+.4f} (p={p_gbt:.4f}); "
                    f"RL-Route vs. Best-Fixed ({best_fixed_key}) delta={d_bf:+.4f} (p={p_bf:.4f})."
                )

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    text = "\n".join(lines)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(text)
    print(f"\n\nSaved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
