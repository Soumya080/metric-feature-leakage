#!/usr/bin/env python3
"""
leakage_ablation.py - Quantify how much the MRR-derived ("leaky") retrieval-score features
inflated every router, and whether a scale-free calibrated design recovers a real margin.

See PLAN_leakage_harness.md for the full background. Short version: 4 of the router's 29
features (avg_bm25/avg_dense/score_gap/bm25_variance, indices 22-25) were computed from MRR, which
is exactly the reward for 2 of the 3 actions — this script builds three env-data variants
that differ ONLY in those 4 features, runs the identical evaluation pipeline on each, and
reports the deltas.

    leaky              - MRR-derived. Reproduces the bug on purpose to quantify it.
    honest_minimal     - raw top-5 score mean/std. The shipped, label-free fix. This is
                          also exactly today's DEFAULT behavior (env-suffix=""), so if
                          results/canonical_results.json already exists (e.g. from the
                          main pipeline), it IS this arm's data and is reused as-is.
    honest_calibrated  - scale-free within-retriever confidence signals (margin, entropy).

This script does NOT run retrieval and does NOT touch the paper source. It only orchestrates
prepare_env_data.py (if the suffixed env files don't exist) and canonical_evaluation.py
(if the suffixed results don't exist), then compares.

Usage:
    python experiments/router_experiments/leakage_ablation.py --smoke   # tiny, safe, structure-only
    python experiments/router_experiments/leakage_ablation.py           # full run (hours)
    python experiments/router_experiments/leakage_ablation.py --skip-run  # results already exist
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from run_full_pipeline import run_stage  # reuse the exact subprocess/skip-if-exists pattern
from canonical_evaluation import DATASETS, get_dataset_paths

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ROUTER_DIR = os.path.join(PROJECT_ROOT, "src", "router")
RE_DIR = os.path.dirname(__file__)

# Arm name -> (prepare_env_data.py --feature-mode, filename suffix).
# honest_minimal's suffix is "" ON PURPOSE: it is today's DEFAULT behavior, so this arm's
# env data and canonical_results.json are exactly what the main pipeline already produces.
ARMS = {
    "leaky": ("leaky", "_leaky"),
    "honest_minimal": ("honest_minimal", ""),
    "honest_calibrated": ("honest_calibrated", "_cal"),
    # Graded leak arms. honest-vs-leaky establishes that contamination matters; these
    # establish how much of it is needed, which is the question a reader auditing their own
    # router actually has. leak_noisy matters most: it evades exact-equality leak detection
    # entirely, so if it still buys accuracy, "I checked that no feature equals the reward"
    # is not a sufficient defence.
    "leak_one_arm": ("leak_one_arm", "_leak1"),
    "leak_hit": ("leak_hit", "_leakhit"),
    "leak_noisy": ("leak_noisy", "_leaknoisy"),
}
# Display order. "leaky" stays first because ARM_ORDER[0] is the reference arm every
# comparison below is drawn against; the rest run from no leak to full leak so the table
# reads as a dose-response curve.
ARM_ORDER = ["leaky", "honest_minimal", "honest_calibrated",
             "leak_noisy", "leak_hit", "leak_one_arm"]

METHOD_LABELS = [
    ("bm25", "Always BM25"),
    ("dense", "Always Dense"),
    ("hybrid", "Always Hybrid"),
    ("score_gap", "Heuristic (Score Gap)"),
    ("logistic_regression", "Logistic Regression"),
    ("random_forest", "Random Forest"),
    ("ridge", "Ridge Regression"),
    ("gradient_boosting", "Gradient Boosting (Tuned)"),
    ("rl_route_mean", "RL-Route (Ours)"),
    ("oracle", "Oracle"),
]
# These never touch `features` — compute_static_baselines reads env.retrievers directly —
# so they MUST be identical across all three arms. Displayed as a built-in sanity check.
FEATURE_INDEPENDENT = {"bm25", "dense", "hybrid", "oracle"}


def env_path(dataset, suffix):
    return get_dataset_paths(suffix)[dataset]


def results_path(suffix):
    name = "canonical_results.json" if not suffix else f"canonical_results{suffix}.json"
    return os.path.join(RESULTS_DIR, name)


# =====================================================================
# 3a. Integrity assertions — run BEFORE any comparison. If these fail, the arms are not
# a clean ablation (something other than features 22:26 differs), and reporting numbers
# would be actively misleading. Exit non-zero without a report in that case.
# =====================================================================

def _load_env(dataset, suffix):
    p = env_path(dataset, suffix)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Missing env data for dataset={dataset!r} suffix={suffix!r}: {p}\n"
            f"Run: python -m src.router.prepare_env_data --feature-mode {ARMS_BY_SUFFIX[suffix]} "
            f"--suffix {suffix!r}"
        )
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


ARMS_BY_SUFFIX = {suffix: mode for mode, suffix in ARMS.values()}


def available_arms(dataset: str):
    """Arms whose env data actually exists, in ARM_ORDER.

    An arm can be legitimately absent -- run_full_pipeline generates the leak gradient and
    honest_minimal but not honest_calibrated, and a user may generate any subset. That is
    different from an arm being CORRUPT, which the integrity checks below still refuse to
    tolerate. Hard-failing on absence would mean adding an arm to ARMS breaks every run
    that has not regenerated env data yet, which is how a diagnostic ends up switched off.
    """
    present, missing = [], []
    for arm in ARM_ORDER:
        (_, suffix) = ARMS[arm]
        (present if os.path.exists(env_path(dataset, suffix)) else missing).append(arm)
    return present, missing


def check_integrity(dataset: str) -> None:
    """Raises AssertionError with a specific message on the first violation."""
    present, missing = available_arms(dataset)
    if missing:
        print(f"[{dataset}] arms without env data, skipped: {', '.join(missing)}")
    if len(present) < 2:
        raise SystemExit(
            f"[{dataset}] only {len(present)} arm(s) present - nothing to compare. "
            f"Generate at least the honest and leaky arms:\n"
            f"  python -m src.router.prepare_env_data --feature-mode honest_minimal --suffix ''\n"
            f"  python -m src.router.prepare_env_data --feature-mode leaky --suffix '_leaky'")
    envs = {arm: _load_env(dataset, ARMS[arm][1]) for arm in present}

    # 1. Identical query-ID sets.
    qid_sets = {arm: set(e.keys()) for arm, e in envs.items()}
    order = [a for a in ARM_ORDER if a in envs]
    ref_arm = order[0]
    for arm in order[1:]:
        assert qid_sets[arm] == qid_sets[ref_arm], (
            f"[{dataset}] query-ID sets differ between {ref_arm!r} and {arm!r} "
            f"({len(qid_sets[ref_arm])} vs {len(qid_sets[arm])}) — arms were built from "
            f"different retrieval runs, this is not a clean ablation."
        )
    qids = sorted(qid_sets[ref_arm])

    # 2. Identical reward tables (mrr per retriever per query).
    for arm in order[1:]:
        for qid in qids:
            for ret in ("bm25", "dense", "hybrid"):
                r_ref = envs[ref_arm][qid]["retrievers"][ret]["mrr"]
                r_arm = envs[arm][qid]["retrievers"][ret]["mrr"]
                assert r_ref == r_arm, (
                    f"[{dataset}] reward mismatch qid={qid} ret={ret}: "
                    f"{ref_arm}={r_ref} vs {arm}={r_arm}. Rewards must be identical across "
                    f"arms — only the FEATURE computation should differ."
                )

    # 3. Everything OUTSIDE the retrieval-score block is bitwise identical across arms:
    #    features[0:22] (query features) and features[26:29] (cross-retriever agreement).
    #    The latter matters specifically because it is the evidence that the cross-retriever
    #    features are label-free — they are computed from retrieved passage IDs, so they
    #    must not shift when the feature MODE (which only governs 22:26) changes.
    def _outside_block(vec):
        return vec[0:22] + vec[26:]

    for arm in order[1:]:
        mismatches = [
            qid for qid in qids
            if _outside_block(envs[ref_arm][qid]["features"])
            != _outside_block(envs[arm][qid]["features"])
        ]
        assert not mismatches, (
            f"[{dataset}] features[0:22]+[26:] differ between {ref_arm!r} and {arm!r} for "
            f"{len(mismatches)} queries (e.g. qid={mismatches[0]}) — every feature outside "
            f"the retrieval-score block must be identical; something beyond features[22:26] "
            f"changed between feature modes."
        )

    # 4. features[22:26] DIFFER between arms (the whole point of the ablation).
    for i, arm_a in enumerate(order):
        for arm_b in order[i + 1:]:
            n_diff = sum(
                1 for qid in qids
                if envs[arm_a][qid]["features"][22:26] != envs[arm_b][qid]["features"][22:26]
            )
            assert n_diff > 0, (
                f"[{dataset}] features[22:26] are IDENTICAL between {arm_a!r} and {arm_b!r} "
                f"for every query — these should be different feature-mode computations; "
                f"the ablation would be comparing an arm against itself."
            )

    # 5. leaky arm genuinely reproduces the bug: f22==bm25.mrr, f23==dense.mrr, everywhere.
    leaky = envs["leaky"]

    def _leaks(entry, feat_idx, retriever):
        """Does this feature carry any label-derived effectiveness value for this arm?

        The check used to compare against retrievers.<ret>.mrr alone, which is the
        UNTRUNCATED 1/gold_rank. The leaky feature mode copies metrics_at_k[k]["mrr"] --
        MRR@k, which is zero past rank k. The two coincide only when gold_rank <= k, so on
        a k=5 build this assertion fired on 674 of 870 queries and declared the leaky arm
        broken when it was working exactly as designed. It went unnoticed because this
        script was never wired into the pipeline.

        The env data does not record which k it was built with, so the honest test is
        whether the feature matches ANY of the label-derived fields present.
        """
        r = entry["retrievers"][retriever]
        candidates = [r[key] for key in ("mrr", "mrr_at_20", "rank_reward", "hit_at_5")
                      if key in r]
        f = entry["features"][feat_idx]
        return any(f == c for c in candidates)

    bad = [qid for qid in qids
           if not _leaks(leaky[qid], 22, "bm25") or not _leaks(leaky[qid], 23, "dense")]
    assert not bad, (
        f"[{dataset}] 'leaky' arm does NOT reproduce the original bug for {len(bad)} "
        f"queries (e.g. qid={bad[0]}) — features[22]/[23] should equal bm25.mrr/dense.mrr "
        f"exactly. The leaky arm is invalid as a reference point for inflation."
    )

    # 6. Honest arms must NOT match the leak, for the large majority of queries.
    for arm in [a for a in ("honest_minimal", "honest_calibrated") if a in envs]:
        env = envs[arm]
        n_leaky_like = sum(
            1 for qid in qids
            if env[qid]["features"][22] == env[qid]["retrievers"]["bm25"]["mrr"]
            and env[qid]["features"][23] == env[qid]["retrievers"]["dense"]["mrr"]
        )
        frac = n_leaky_like / max(1, len(qids))
        assert frac < 0.5, (
            f"[{dataset}] {arm!r} matches the MRR-leak pattern on {frac:.0%} of queries — "
            f"expected well under 50% by chance. This arm may not actually be honest."
        )

    print(f"[{dataset}] integrity checks passed ({len(qids)} queries, {len(order)} arms: "
          f"{', '.join(order)}).")


# =====================================================================
# 3b. Orchestration
# =====================================================================

def ensure_env_data(dataset: str, arm: str, force: bool = False,
                    gold_type: str = "real_wikipedia"):
    """Build one arm's env data.

    gold_type MUST match whatever the honest_minimal arm's data was built with, because
    that arm reuses the main pipeline's default *_env_data.json. run_full_pipeline.py
    defaults to --gold-type real_wikipedia, so this does too — otherwise the leaky/
    calibrated arms would cover all 5984 queries while honest_minimal covers 2788, and
    check_integrity's query-ID assertion would (correctly) refuse the whole comparison.
    """
    mode, suffix = ARMS[arm]
    p = env_path(dataset, suffix)
    if os.path.exists(p) and not force:
        return
    print(f"\n[{arm}] env data missing for {dataset}, building via prepare_env_data.py "
          f"(--feature-mode {mode} --suffix {suffix!r} --gold-type {gold_type})...")
    result = subprocess.run(
        [sys.executable, "-m", "src.router.prepare_env_data",
         "--feature-mode", mode, "--suffix", suffix, "--gold-type", gold_type],
        cwd=PROJECT_ROOT, env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
    )
    if result.returncode != 0:
        raise RuntimeError(f"prepare_env_data.py failed for arm={arm}")


def ensure_canonical_results(arm: str, seeds: str = None, epochs: int = None,
                              output_override: str = None, force: bool = False):
    _mode, suffix = ARMS[arm]
    out = output_override or results_path(suffix)
    if os.path.exists(out) and not force:
        print(f"[{arm}] reusing existing {out}")
        return out

    if not suffix:
        print(f"[{arm}] NOTE: this targets results/canonical_results.json — the same file "
              f"the main pipeline (run_full_pipeline.py) produces. If that pipeline is "
              f"currently running, wait for it or pass --skip-run instead of racing it.")

    cli_args = ["--env-suffix", suffix, "--output", out]
    if seeds:
        cli_args += ["--seeds", seeds]
    if epochs:
        cli_args += ["--epochs", str(epochs)]

    script = os.path.join(RE_DIR, "canonical_evaluation.py")
    ok = run_stage(f"canonical_evaluation.py ({arm})", script, cli_args,
                    check_output=[out], force=force)
    if not ok:
        raise RuntimeError(f"canonical_evaluation.py failed for arm={arm}")
    return out


# =====================================================================
# 3c. Report
# =====================================================================

def build_table(dataset: str, arm_results: dict) -> tuple:
    """Returns (markdown_lines, latex_lines, warnings)."""
    md = [f"\n## {dataset.upper()}\n",
          "| Method | Leaky | Honest-minimal | Honest-calibrated | Inflation (leaky - honest_cal) |",
          "|---|---|---|---|---|"]
    warnings = []

    for key, label in METHOD_LABELS:
        vals = {}
        for arm in ARM_ORDER:
            v = arm_results[arm].get(key)
            vals[arm] = float(v) if v is not None else None
        if any(v is None for v in vals.values()):
            continue

        if key in FEATURE_INDEPENDENT:
            spread = max(vals.values()) - min(vals.values())
            if spread > 1e-9:
                warnings.append(
                    f"[{dataset}] {label} differs across arms by {spread:.6f} but should be "
                    f"feature-independent — check reward-table integrity."
                )

        inflation = vals["leaky"] - vals["honest_calibrated"]
        md.append(
            f"| {label} | {vals['leaky']:.4f} | {vals['honest_minimal']:.4f} | "
            f"{vals['honest_calibrated']:.4f} | {inflation:+.4f} |"
        )

    latex = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Leakage ablation (" + dataset.upper() + r"): MRR@5 under MRR-derived "
        r"(leaky), raw-score (honest-minimal), and scale-free (honest-calibrated) "
        r"retrieval-score features. Static baselines and Oracle do not depend on features "
        r"and are identical across columns by construction.}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Method & Leaky & Honest-min. & Honest-cal. & Inflation \\", r"\midrule",
    ]
    for key, label in METHOD_LABELS:
        vals = {arm: arm_results[arm].get(key) for arm in ARM_ORDER}
        if any(v is None for v in vals.values()):
            continue
        inflation = vals["leaky"] - vals["honest_calibrated"]
        latex.append(f"{label} & {vals['leaky']:.4f} & {vals['honest_minimal']:.4f} & "
                     f"{vals['honest_calibrated']:.4f} & {inflation:+.4f} \\\\")
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    return md, latex, warnings


def main():
    parser = argparse.ArgumentParser(description="Three-arm leakage ablation: leaky vs. honest_minimal vs. honest_calibrated.")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny scale (1 seed, 2 epochs), writes to results/_smoke/ "
                             "instead of results/. Validates structure/assertions only — "
                             "numbers are meaningless.")
    parser.add_argument("--skip-run", action="store_true",
                        help="Assume env data and canonical_results already exist for all "
                             "arms; only build the comparison report.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate env data and results even if they already exist.")
    parser.add_argument("--gold-type", type=str, default="real_wikipedia",
                        choices=["real_wikipedia", "all", "constructed"],
                        help="Gold-passage filter for every arm. MUST match what the main "
                             "pipeline built its default *_env_data.json with (also "
                             "real_wikipedia by default) — the honest_minimal arm reuses "
                             "those files, so a mismatch means the arms cover different "
                             "queries and the integrity check refuses to report.")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated subset of {nq,triviaqa,hotpotqa} (default: all).")
    args = parser.parse_args()

    datasets = list(DATASETS.keys())
    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(",")}
        datasets = [d for d in datasets if d in keep]

    out_dir = RESULTS_DIR
    seeds = epochs = None
    if args.smoke:
        out_dir = os.path.join(RESULTS_DIR, "_smoke")
        os.makedirs(out_dir, exist_ok=True)
        seeds, epochs = "1", 2
        print(f"SMOKE MODE: seeds={seeds}, epochs={epochs}, output -> {out_dir} "
              f"(never touches the real results/ files)")

    # --- Step A: env data for every (dataset, arm) ---
    if not args.skip_run:
        for ds in datasets:
            for arm in ARM_ORDER:
                ensure_env_data(ds, arm, force=args.force, gold_type=args.gold_type)

    # --- Step B: integrity assertions BEFORE trusting any comparison ---
    for ds in datasets:
        try:
            check_integrity(ds)
        except AssertionError as e:
            print(f"\nINTEGRITY CHECK FAILED — refusing to report numbers.\n{e}")
            sys.exit(1)
        except FileNotFoundError as e:
            print(f"\n{e}")
            sys.exit(1)

    # --- Step C: canonical_evaluation.py per arm ---
    result_files = {}
    if not args.skip_run:
        for arm in ARM_ORDER:
            _mode, suffix = ARMS[arm]
            override = None
            if args.smoke:
                name = "canonical_results.json" if not suffix else f"canonical_results{suffix}.json"
                override = os.path.join(out_dir, name)
            result_files[arm] = ensure_canonical_results(
                arm, seeds=seeds, epochs=epochs, output_override=override, force=args.force
            )
    else:
        for arm in ARM_ORDER:
            _mode, suffix = ARMS[arm]
            result_files[arm] = results_path(suffix)
            if not os.path.exists(result_files[arm]):
                print(f"--skip-run passed but {result_files[arm]} does not exist. Aborting.")
                sys.exit(1)

    # --- Step D: build the report ---
    all_data = {arm: json.load(open(p, encoding="utf-8")) for arm, p in result_files.items()}

    all_md = ["# Leakage Ablation: Leaky vs. Honest-Minimal vs. Honest-Calibrated\n",
              "Three env-data variants built from the SAME retrieval run, differing ONLY in "
              "features[22:26] (avg_bm25/avg_dense/score_gap/bm25_variance). Everything else "
              "— splits, seeds, rewards, features[0:22] and the cross-retriever features "
              "[26:29] — is identical; integrity-checked "
              "before this report was generated (see console log). `leaky` reproduces the "
              "original MRR-derived-feature bug on purpose to quantify its effect; it is not "
              "a result to report on its own.\n"]
    all_warnings = []
    all_json = {}

    for ds in datasets:
        arm_results = {arm: all_data[arm].get(ds, {}) for arm in ARM_ORDER}
        if any(not r for r in arm_results.values()):
            print(f"WARNING: missing results for dataset={ds} in one or more arms — skipping.")
            continue
        md, latex, warnings = build_table(ds, arm_results)
        all_md += md
        all_warnings += warnings
        all_json[ds] = {
            arm: {k: v for k, v in arm_results[arm].items() if k != "per_query"}
            for arm in ARM_ORDER
        }
        all_md.append("\n```latex")
        all_md += latex
        all_md.append("```")

    if all_warnings:
        all_md.append("\n## Integrity warnings\n")
        for w in all_warnings:
            all_md.append(f"- {w}")

    md_path = os.path.join(out_dir, "leakage_ablation.md")
    json_path = os.path.join(out_dir, "leakage_ablation.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_md) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_json, f, indent=2)

    print("\n".join(all_md))
    print(f"\n\nSaved: {md_path}")
    print(f"Saved: {json_path}")
    if all_warnings:
        print(f"\n{len(all_warnings)} integrity warning(s) — see report.")


if __name__ == "__main__":
    main()
