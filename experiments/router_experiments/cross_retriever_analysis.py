"""
Where does the routing headroom sit relative to retriever agreement?

Tests a falsifiable prediction about the mechanism:

    Routing headroom (oracle - best-fixed) should be LARGEST where BM25 and Dense
    disagree, and near ZERO where they agree.

The reasoning is not subtle: if two retrievers return the same passages, they earn nearly
the same reward, so which one you pick cannot matter much. Every point of achievable gain
has to come from queries where they diverge. If that trend does not appear in the data,
the routing story is wrong regardless of what the aggregate MRR says — which is precisely
what makes it worth testing rather than asserting.

Reported per agreement bin: n, best-fixed MRR, oracle MRR, headroom, and each method's
realized MRR (notably RL router vs. tuned Gradient Boosting). Plus a Spearman correlation
between per-query agreement and per-query headroom, so the trend is a number and not an
eyeball judgement.

Inputs (both produced by the normal pipeline; nothing is recomputed here):
  - results/canonical_results.json  -> the `per_query` block written by canonical_evaluation.py
  - src/router/*_env_data.json      -> features[26] = cross_ret_overlap_5

Usage:
    python experiments/router_experiments/cross_retriever_analysis.py
    python experiments/router_experiments/cross_retriever_analysis.py --results <path> --datasets nq
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from canonical_evaluation import get_dataset_paths  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEFAULT_RESULTS = os.path.join(RESULTS_DIR, "canonical_results.json")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "cross_retriever_analysis.json")
OUTPUT_MD = os.path.join(RESULTS_DIR, "cross_retriever_analysis.md")
OUTPUT_PDF = os.path.join(RESULTS_DIR, "cross_retriever_analysis.pdf")

# Index of cross_ret_overlap_5 in QueryFeatureExtractor.get_feature_order().
FEATURE_IDX_OVERLAP5 = 26
EXPECTED_FEATURE_DIM = 29

STATIC = ["bm25", "dense", "hybrid"]
# Bin edges for top-5 overlap. With |top5|=5 the value is a multiple of 0.2, so these
# buckets land on the natural discrete levels rather than splitting them arbitrarily.
BIN_EDGES = [-0.01, 0.1, 0.3, 0.5, 0.7, 1.01]
BIN_LABELS = ["0.0 (disjoint)", "0.2", "0.4", "0.6", "0.8-1.0 (agree)"]


def load_agreement(env_path):
    """query_id -> cross_ret_overlap_5, from the env data the pipeline already built."""
    with open(env_path, "r", encoding="utf-8") as f:
        env = json.load(f)
    queries = env.get("queries", env)

    agreement, short = {}, 0
    for qid, rec in queries.items():
        feats = rec.get("features", [])
        if len(feats) < EXPECTED_FEATURE_DIM:
            short += 1
            continue
        agreement[str(qid)] = float(feats[FEATURE_IDX_OVERLAP5])

    if short:
        raise ValueError(
            f"{os.path.basename(env_path)}: {short} records have fewer than "
            f"{EXPECTED_FEATURE_DIM} features — this env data predates the cross-retriever "
            f"features. Re-run: python -m src.router.prepare_env_data"
        )
    if agreement and all(v == 0.0 for v in agreement.values()):
        raise ValueError(
            f"{os.path.basename(env_path)}: cross_ret_overlap_5 is 0.0 for EVERY query. "
            f"That almost certainly means 02_run_baselines.py ran without persisting "
            f"'retrieved_ids_top10' — re-run stage 2, then stage 3."
        )
    return agreement


def analyze(per_query, agreement):
    """Bin queries by BM25/Dense agreement; report headroom and per-method MRR per bin."""
    qids = [str(q) for q in per_query["query_ids"]]
    scores = per_query["scores"]

    keep = [i for i, q in enumerate(qids) if q in agreement]
    if not keep:
        raise ValueError("No overlap between canonical_results query_ids and env data keys.")

    agree = np.array([agreement[qids[i]] for i in keep])
    arrs = {m: np.array([scores[m][i] for i in keep])
            for m in scores if len(scores[m]) == len(qids)}

    stacked = np.vstack([arrs[r] for r in STATIC])
    oracle = stacked.max(axis=0)

    # Best-fixed is a GLOBAL policy: pick the single best retriever overall, then look at
    # how it does inside each bin. Choosing the best retriever per bin would silently be a
    # coarse oracle and would understate the headroom this analysis is measuring.
    global_means = {r: float(arrs[r].mean()) for r in STATIC}
    best_fixed_name = max(global_means, key=global_means.get)
    best_fixed = arrs[best_fixed_name]

    headroom = oracle - best_fixed
    idx = np.digitize(agree, BIN_EDGES) - 1

    bins = []
    for b, label in enumerate(BIN_LABELS):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            bins.append({"label": label, "n": 0})
            continue
        row = {
            "label": label,
            "n": n,
            "best_fixed": float(best_fixed[mask].mean()),
            "oracle": float(oracle[mask].mean()),
            "headroom": float(headroom[mask].mean()),
            "methods": {m: float(v[mask].mean()) for m, v in arrs.items()},
        }
        bins.append(row)

    rho, p = spearman(agree, headroom)
    return {
        "best_fixed_name": best_fixed_name,
        "global_means": global_means,
        "n_queries": len(keep),
        "bins": bins,
        "spearman_agreement_vs_headroom": {"rho": rho, "p_value": p},
    }


def spearman(x, y):
    """Spearman rho + two-sided p. Uses scipy when available, else a normal approximation.

    Returns (nan, nan) when either input is constant — the correlation is genuinely
    undefined there, and that happens legitimately on tiny samples where every query has
    identical headroom. Caught explicitly so callers report "undefined" rather than
    surfacing a scipy warning and a bare nan.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y)
        return float(r.statistic), float(r.pvalue)
    except Exception:
        rx, ry = _rank(x), _rank(y)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        n = len(x)
        if n < 4 or abs(rho) >= 1.0:
            return rho, float("nan")
        # Fisher z; adequate here since n is in the hundreds.
        z = np.arctanh(rho) * np.sqrt((n - 3) / 1.06)
        from math import erfc, sqrt
        return rho, float(erfc(abs(z) / sqrt(2)))


def _rank(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties, so plateaus (agreement is discrete) don't bias the correlation
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def render_markdown(all_res):
    lines = [
        "# Cross-Retriever Agreement vs. Routing Headroom\n",
        "Tests the mechanism behind the routing claim: headroom (oracle - best-fixed) should",
        "concentrate where BM25 and Dense **disagree**, and vanish where they agree. Where the",
        "two retrievers return the same passages they earn nearly the same reward, so the",
        "routing decision cannot matter — every point of achievable gain must come from the",
        "queries where they diverge.\n",
        "Agreement = top-5 retrieved-passage overlap (`cross_ret_overlap_5`, feature 26),",
        "computed from passage identity only and never from the gold answer.\n",
    ]
    for ds, res in all_res.items():
        rho = res["spearman_agreement_vs_headroom"]["rho"]
        p = res["spearman_agreement_vs_headroom"]["p_value"]
        # The interpretation line used to be a static string asserting that negative rho
        # supports the hypothesis -- printed unconditionally, including on NQ where rho is
        # +0.013 at p = 0.88. It has to read the result it is interpreting, and it has to
        # weigh significance before direction: a rho of the "right" sign at p = 0.4 is not
        # weak support, it is no evidence.
        if np.isnan(rho):
            corr_line = ("- Spearman(agreement, headroom) = **undefined** (headroom or "
                         "agreement is constant across queries — too few queries to test)")
            reading = ("  - Not testable on this dataset; the bucket table below is "
                       "descriptive only.\n")
        else:
            corr_line = f"- Spearman(agreement, headroom) = **{rho:+.3f}** (p = {p:.2e})"
            if np.isnan(p) or p > 0.05:
                reading = ("  - **No significant relationship.** The hypothesis that "
                           "headroom concentrates where the retrievers disagree is not "
                           "supported on this dataset, in either direction.\n")
            elif rho < 0:
                reading = ("  - Negative rho at p < 0.05 supports the hypothesis: less "
                           "agreement -> more headroom.\n")
            else:
                reading = ("  - **Positive rho at p < 0.05 contradicts the hypothesis**: "
                           "headroom is larger where the retrievers agree, which is the "
                           "opposite of the stated mechanism.\n")
        lines += [
            f"\n## {ds.upper()}\n",
            f"- Queries analysed: **{res['n_queries']}**",
            f"- Best-fixed retriever (global): **{res['best_fixed_name']}**",
            corr_line,
            reading,
            "| agreement (top-5 overlap) | n | best-fixed | oracle | headroom | RL router | GBT |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for b in res["bins"]:
            if b["n"] == 0:
                lines.append(f"| {b['label']} | 0 | — | — | — | — | — |")
                continue
            m = b["methods"]
            lines.append(
                f"| {b['label']} | {b['n']} | {b['best_fixed']:.3f} | {b['oracle']:.3f} | "
                f"**{b['headroom']:+.3f}** | {m.get('rl_route', float('nan')):.3f} | "
                f"{m.get('gradient_boosting', float('nan')):.3f} |"
            )
    return "\n".join(lines) + "\n"


def render_pdf(all_res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping PDF.")
        return False

    n = len(all_res)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)

    for col, (ds, res) in enumerate(all_res.items()):
        bins = [b for b in res["bins"] if b["n"] > 0]
        labels = [b["label"] for b in bins]
        xs = np.arange(len(bins))

        ax = axes[0][col]
        ax.bar(xs, [b["headroom"] for b in bins], color="#4C72B0")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Headroom (oracle - best fixed)")
        rho = res["spearman_agreement_vs_headroom"]["rho"]
        rho_txt = "undefined" if np.isnan(rho) else f"{rho:+.3f}"
        ax.set_title(f"{ds.upper()} — headroom vs. agreement\nSpearman rho = {rho_txt}")
        for x, b in zip(xs, bins):
            ax.text(x, b["headroom"], f"n={b['n']}", ha="center", va="bottom", fontsize=7)

        ax = axes[1][col]
        w = 0.35
        rl = [b["methods"].get("rl_route", np.nan) for b in bins]
        gbt = [b["methods"].get("gradient_boosting", np.nan) for b in bins]
        ax.bar(xs - w / 2, rl, w, label="RL router", color="#4C72B0")
        ax.bar(xs + w / 2, gbt, w, label="Gradient Boosting", color="#DD8452")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("MRR@5")
        ax.set_title(f"{ds.upper()} — method MRR by agreement")
        ax.legend(fontsize=8)

    fig.suptitle("Routing gains concentrate where retrievers disagree", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS)
    parser.add_argument("--env-suffix", type=str, default="",
                        help="Env-data suffix to read (must match how --results was produced).")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated subset, e.g. 'nq'. Default: all in the results file.")
    parser.add_argument("--output-json", type=str, default=OUTPUT_JSON)
    parser.add_argument("--output-md", type=str, default=OUTPUT_MD)
    parser.add_argument("--output-pdf", type=str, default=OUTPUT_PDF)
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: {args.results} not found. Run canonical_evaluation.py first.")
        sys.exit(1)

    with open(args.results, "r", encoding="utf-8") as f:
        results_data = json.load(f)

    env_paths = get_dataset_paths(args.env_suffix)
    wanted = ({d.strip() for d in args.datasets.split(",")} if args.datasets
              else set(results_data.keys()))

    all_res = {}
    for ds, ds_res in results_data.items():
        if ds not in wanted or not isinstance(ds_res, dict):
            continue
        pq = ds_res.get("per_query")
        if not pq:
            print(f"  {ds}: no per_query block — skipping.")
            continue
        if ds not in env_paths or not os.path.exists(env_paths[ds]):
            print(f"  {ds}: env data not found at {env_paths.get(ds)!r} — skipping.")
            continue

        print(f"\n--- {ds} ---")
        res = analyze(pq, load_agreement(env_paths[ds]))
        all_res[ds] = res
        rho = res["spearman_agreement_vs_headroom"]["rho"]
        rho_txt = "undefined (constant input)" if np.isnan(rho) else f"{rho:+.3f}"
        print(f"  best-fixed={res['best_fixed_name']}  n={res['n_queries']}  "
              f"Spearman(agreement, headroom)={rho_txt}")
        for b in res["bins"]:
            if b["n"]:
                print(f"    overlap {b['label']:<18} n={b['n']:<5} headroom={b['headroom']:+.4f}")

    if not all_res:
        print("\nNothing analysed.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(all_res))
    wrote_pdf = render_pdf(all_res, args.output_pdf)

    print(f"\nSaved: {args.output_json}")
    print(f"Saved: {args.output_md}")
    if wrote_pdf:
        print(f"Saved: {args.output_pdf}")
    print("\nNegative Spearman rho = headroom rises as agreement falls = the mechanism holds.")
    print("If rho is ~0 or positive, the routing gain is NOT coming from retriever")
    print("disagreement, and that claim should be dropped from the paper rather than reworded.")


if __name__ == "__main__":
    main()
