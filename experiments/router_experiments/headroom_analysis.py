"""How much of the oracle's headroom does each method actually capture, and where does it live?

Two questions that the results files answer only implicitly, and that were being computed by
hand — which is how numbers end up in a paper with no artifact behind them.

1. HEADROOM CAPTURE. A raw MRR says nothing about whether routing worked, because the
   achievable range differs per dataset. The interpretable quantity is the share of the
   oracle's advantage a method actually realised:

       capture = (method - best_fixed) / (oracle - best_fixed)

   100% is perfect routing; 0% is "you may as well have picked one retriever and never
   routed"; negative is worse than that. Under leaky features every method appeared to
   capture 70-90%. Under honest features the best cell anywhere is a few percent. Same
   table, same split — the difference is entirely feature construction.

2. WHERE THE HEADROOM LIVES. The obvious defence of a negative routing result is that the
   headroom is noise in the deep ranks where nobody would notice. This decomposes the
   oracle-minus-best-fixed gap by the rank at which the winning retriever placed the gold.
   On this data ~88% of it sits in ranks 1-5 — precisely the regime a user feels — so the
   opportunity is real and meaningful, which makes failing to capture it a finding rather
   than a triviality. Reported because it is the honest check, not because it flatters the
   result.

Reads existing artifacts and trains nothing. Pass several canonical results files to get
the regime comparison in one table:

    python headroom_analysis.py \
        --results honest=results/canonical_results.json \
                  leaky=results/canonical_results_leaky.json
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

DATASETS = ["nq", "triviaqa", "hotpotqa"]
ACTIONS = ["bm25", "dense", "hybrid"]
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Methods to report capture for, in table order. Static arms are excluded on purpose: the
# best of them IS the denominator's baseline, so its capture is 0 by construction and
# reporting it invites the reader to treat 0% as a failure rather than as the definition.
METHODS = [
    ("score_gap", "Heuristic (score gap)"),
    ("logistic_regression", "Logistic Regression"),
    ("random_forest", "Random Forest"),
    ("ridge", "Ridge Regression"),
    ("gradient_boosting", "Gradient Boosting (tuned)"),
    ("rl_route_mean", "RL router"),
]

RANK_BUCKETS = [("rank 1-5", 1, 5), ("rank 6-20", 6, 20),
                ("rank 21-100", 21, 100), ("rank >100", 101, None)]


def capture(method, best_fixed, oracle):
    """Share of the oracle's advantage realised. None when there is no headroom to capture."""
    gap = oracle - best_fixed
    if gap <= 1e-12:
        return None
    return (method - best_fixed) / gap


def analyse_capture(results):
    """Per-dataset capture for every method in one canonical results file."""
    out = {}
    for ds in DATASETS:
        r = results.get(ds)
        if not isinstance(r, dict):
            continue
        arms = {a: r[a] for a in ACTIONS if a in r}
        if not arms or "oracle" not in r:
            continue
        best_name = max(arms, key=arms.get)
        best_fixed, oracle = arms[best_name], r["oracle"]
        out[ds] = {
            "best_fixed_name": best_name, "best_fixed": best_fixed, "oracle": oracle,
            "headroom": oracle - best_fixed,
            "capture": {key: capture(r[key], best_fixed, oracle)
                        for key, _ in METHODS if key in r},
        }
    return out


def analyse_decomposition(env_path, metric="mrr"):
    """Split the oracle-minus-best-fixed gap by the winning retriever's gold rank.

    Weighted by each query's actual contribution to the mean gap, not by query count: a
    bucket holding many queries that each contribute nothing is not where the headroom is.
    """
    with open(env_path, "r", encoding="utf-8") as f:
        env = json.load(f)
    env = env.get("queries", env)

    rows = []
    for entry in env.values():
        ret = entry["retrievers"]
        rows.append(({a: float(ret[a][metric]) for a in ACTIONS},
                     {a: ret[a].get("gold_rank") for a in ACTIONS}))
    n = len(rows)
    if not n:
        return None

    means = {a: float(np.mean([v[a] for v, _ in rows])) for a in ACTIONS}
    best_name = max(means, key=means.get)
    oracle = float(np.mean([max(v.values()) for v, _ in rows]))
    headroom = oracle - means[best_name]

    buckets = {label: 0.0 for label, _, _ in RANK_BUCKETS}
    for vals, ranks in rows:
        winner = max(vals, key=vals.get)
        gain = (vals[winner] - vals[best_name]) / n
        gr = ranks.get(winner)
        if gr is None:
            buckets["rank >100"] += gain
            continue
        for label, lo, hi in RANK_BUCKETS:
            if gr >= lo and (hi is None or gr <= hi):
                buckets[label] += gain
                break

    return {
        "n": n, "best_fixed_name": best_name, "best_fixed": means[best_name],
        "oracle": oracle, "headroom": headroom,
        "headroom_rel": headroom / means[best_name] if means[best_name] else None,
        "buckets": buckets,
        "bucket_share": {k: (v / headroom if headroom else None) for k, v in buckets.items()},
    }


def render(cap_by_regime, decomp):
    L = ["# Headroom: how much is captured, and where it lives", ""]

    L += ["## 1. Share of oracle headroom captured", "",
          "`capture = (method - best_fixed) / (oracle - best_fixed)`. 100% is perfect "
          "routing; 0% means you may as well have committed to one retriever; negative is "
          "worse than that.", ""]
    regimes = list(cap_by_regime)
    header = "| method | " + " | ".join(f"{ds} ({rg})" for rg in regimes for ds in DATASETS) + " |"
    L += [header, "|---" * (1 + len(regimes) * len(DATASETS)) + "|"]
    for key, label in METHODS:
        cells = []
        for rg in regimes:
            for ds in DATASETS:
                c = cap_by_regime[rg].get(ds, {}).get("capture", {}).get(key)
                cells.append("—" if c is None else f"{c*100:+.0f}%")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "Best-fixed and oracle per dataset:", "",
          "| dataset | regime | best fixed | oracle | headroom |", "|---|---|---:|---:|---:|"]
    for rg in regimes:
        for ds in DATASETS:
            e = cap_by_regime[rg].get(ds)
            if e:
                L.append(f"| {ds} | {rg} | {e['best_fixed_name']} {e['best_fixed']:.4f} | "
                         f"{e['oracle']:.4f} | +{e['headroom']:.4f} |")

    L += ["", "## 2. Where the headroom lives", "",
          "The oracle-minus-best-fixed gap, split by the rank at which the winning "
          "retriever placed the gold passage. If the headroom sat in the deep ranks it "
          "would not matter much that no method captures it.", "",
          "| dataset | " + " | ".join(l for l, _, _ in RANK_BUCKETS) + " |",
          "|---" * (1 + len(RANK_BUCKETS)) + "|"]
    for ds, d in decomp.items():
        if not d:
            continue
        L.append(f"| {ds} | " + " | ".join(
            f"{d['bucket_share'][l]*100:.1f}%" if d["bucket_share"][l] is not None else "—"
            for l, _, _ in RANK_BUCKETS) + " |")
    top5 = [d["bucket_share"]["rank 1-5"] for d in decomp.values()
            if d and d["bucket_share"]["rank 1-5"] is not None]
    if top5:
        L += ["", f"Ranks 1-5 hold {min(top5)*100:.0f}-{max(top5)*100:.0f}% of the headroom "
                  f"across datasets — the regime a user would actually notice. The "
                  f"opportunity is real and concentrated where it matters, which is what "
                  f"makes failing to capture it a result rather than a technicality.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+",
                    default=[f"honest={os.path.join(RESULTS_DIR, 'canonical_results.json')}"],
                    help="one or more label=path canonical results files")
    ap.add_argument("--env-dir", default=os.path.join(PROJECT_ROOT, "src", "router"))
    ap.add_argument("--metric", default="mrr")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    cap_by_regime = {}
    for spec in args.results:
        label, _, path = spec.partition("=")
        if not path:
            label, path = "results", spec
        if not os.path.exists(path):
            print(f"  {label}: {path} not found - skipped.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            cap_by_regime[label] = analyse_capture(json.load(f))
        print(f"  {label}: capture computed from {os.path.basename(path)}")

    if not cap_by_regime:
        raise SystemExit("No canonical results files found.")

    decomp = {}
    for ds in DATASETS:
        env_path = os.path.join(args.env_dir, f"{ds}_env_data.json")
        if os.path.exists(env_path):
            decomp[ds] = analyse_decomposition(env_path, args.metric)
            d = decomp[ds]
            if d:
                print(f"  {ds}: headroom +{d['headroom']:.4f} "
                      f"({d['headroom_rel']*100:.0f}%), "
                      f"{d['bucket_share']['rank 1-5']*100:.1f}% of it in ranks 1-5")
        else:
            print(f"  {ds}: env data not found - decomposition skipped.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "headroom_analysis.json"), "w", encoding="utf-8") as f:
        json.dump({"capture": cap_by_regime, "decomposition": decomp}, f, indent=2)
    with open(os.path.join(args.out_dir, "headroom_analysis.md"), "w", encoding="utf-8") as f:
        f.write(render(cap_by_regime, decomp))
    print(f"\nSaved: {os.path.join(args.out_dir, 'headroom_analysis.json')}")
    print(f"Saved: {os.path.join(args.out_dir, 'headroom_analysis.md')}")


if __name__ == "__main__":
    main()
