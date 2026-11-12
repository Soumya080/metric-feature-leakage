"""
cost_aware_analysis.py — Is there a routing decision to make at all?

A router can only be useful if the best action varies by query. Under a quality-only
reward that is barely true here: Dense is the best retriever on ~75% of queries, so
"always Dense" is nearly optimal, headroom is small, and no method — RL or supervised —
can separate from the best fixed choice. That is a property of the *objective*, not of
the learner, and no amount of feature engineering or policy tuning fixes it.

Cost changes that. Hybrid is not an independent third option: Reciprocal Rank Fusion runs
BM25 *and* Dense and then merges them, so it strictly dominates both in cost while winning
only sometimes. Once the reward charges for compute:

    r(q, a) = MRR(q, a)  -  lambda * cost(a)

the optimal action becomes query-dependent in a way it currently is not — cheap retrieval
where it suffices, expensive retrieval only where it pays. This script measures whether
that actually happens in our data, and at which lambda.

It is deliberately a *structural* check, not a method: it computes the ORACLE choice under
each lambda and asks how mixed that choice is. If the oracle itself stays concentrated on
one action at every lambda, then cost-aware routing is degenerate too and the idea should
be dropped — cheaply, before any training run. If the oracle becomes mixed, the problem is
real and worth learning; that still does not prove it is learnable, only that it is not
vacuous.

Two cost models, because they answer different questions:

  structural (default)  Unit costs. Hybrid costs what BM25 and Dense cost together,
                        because that is literally what it runs. Hardware- and
                        implementation-independent, so it is the one to put in the paper.

IMPORTANT — cost only creates a decision where costs actually differ. For any two actions,

    r(a) - r(b) = [MRR(a) - MRR(b)] - lambda * [cost(a) - cost(b)]

so if two retrievers cost the same, lambda cancels and the choice between them is
lambda-invariant no matter how large lambda gets. With equal BM25/Dense costs the sweep can
only prune Hybrid, whose share is absorbed by whichever of the two was already dominant —
making the problem MORE concentrated, not less. That is why the default charges Dense more
than BM25: a dense retriever runs a neural encoder forward pass over the query and then an
ANN search, against an inverted-index lookup for BM25. The BM25-vs-Dense cost gap is the
entire source of the routing decision; if you set it to zero, the analysis is vacuous by
construction rather than by evidence.
  measured              The wall-clock latency_ms recorded during retrieval. Honest about
                        this codebase but NOT a deployment cost: each query here builds a
                        fresh FAISS index over its own 500-passage corpus, so Dense's
                        measured time is dominated by index construction that a real
                        system pays once, offline. Reported for transparency; do not base
                        a claim on it.

Usage:
    python experiments/router_experiments/cost_aware_analysis.py
    python experiments/router_experiments/cost_aware_analysis.py --cost-model measured
    python experiments/router_experiments/cost_aware_analysis.py --unit-costs 1,3,4
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from canonical_evaluation import get_dataset_paths  # noqa: E402
from src.router.routing_env import create_train_val_test_split  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ACTIONS = ["bm25", "dense", "hybrid"]
SPLIT_SEED = 42


def load_all_queries(path):
    """Every query in the dataset, with its per-action quality and measured latency.

    This is a characterisation of the benchmark, not an evaluation of a method, so all
    three splits are pooled deliberately — there is no model being fit and therefore
    nothing to leak. The splits are still constructed through create_train_val_test_split
    so the query set is identical to what every experiment sees.
    """
    envs = create_train_val_test_split(path, split_seed=SPLIT_SEED)
    qids, quality, latency = [], [], []
    mrr_key = None
    for env in envs:
        for q in env.query_pool:
            rets = env.master_table[q]["retrievers"]
            if mrr_key is None:
                # Prefer the reported metric (MRR@k) over the training reward, which may be
                # the denser 1/gold_rank. The cost trade-off has to be stated in the units
                # the paper reports, not in the units training happens to optimise.
                mrr_key = next((k for k in rets["bm25"] if k.startswith("mrr_at_")), "mrr")
            qids.append(q)
            quality.append([float(rets[a][mrr_key]) for a in ACTIONS])
            latency.append([float(rets[a].get("latency_ms", 0.0)) for a in ACTIONS])
    return qids, np.array(quality), np.array(latency), mrr_key


def build_costs(latency, cost_model, unit_costs):
    """Per-query per-action cost, normalised so lambda is comparable across datasets.

    Normalising by the most expensive action's mean cost makes lambda interpretable: it is
    the amount of MRR you are willing to give up to avoid one full unit of the most
    expensive retriever. lambda=0 is the current quality-only objective; lambda=1 says a
    whole point of MRR is worth exactly one Hybrid call.
    """
    if cost_model == "structural":
        costs = np.tile(np.asarray(unit_costs, dtype=float), (latency.shape[0], 1))
    else:
        costs = latency.copy()
    ref = costs.mean(axis=0).max()
    return costs / ref if ref > 0 else costs


def sweep(quality, costs, lambdas):
    """For each lambda: how mixed is the oracle choice, and what does routing buy?"""
    rows = []
    for lam in lambdas:
        reward = quality - lam * costs
        best = reward.argmax(axis=1)
        counts = np.bincount(best, minlength=len(ACTIONS)) / len(best)

        # Majority rate: accuracy of "always predict the most common optimal action". This
        # is the baseline any router must beat, and the number that decides go/no-go.
        majority = float(counts.max())

        # Best fixed action is chosen on the cost-aware reward, then reported on both axes.
        fixed_idx = int(reward.mean(axis=0).argmax())
        oracle_reward = float(reward.max(axis=1).mean())
        fixed_reward = float(reward[:, fixed_idx].mean())

        rows.append({
            "lambda": float(lam),
            "action_share": {a: float(c) for a, c in zip(ACTIONS, counts)},
            "majority_rate": majority,
            "best_fixed_name": ACTIONS[fixed_idx],
            "oracle_reward": oracle_reward,
            "best_fixed_reward": fixed_reward,
            "headroom": oracle_reward - fixed_reward,
            "headroom_rel": ((oracle_reward - fixed_reward) / abs(fixed_reward)
                             if fixed_reward else 0.0),
            # The Pareto point: what the cost-aware oracle actually achieves on the two
            # axes a reader cares about, rather than on the scalarised reward.
            "oracle_quality": float(quality[np.arange(len(best)), best].mean()),
            "oracle_cost": float(costs[np.arange(len(best)), best].mean()),
        })
    return rows


def analyze(name, path, lambdas, cost_model, unit_costs):
    qids, quality, latency, mrr_key = load_all_queries(path)
    costs = build_costs(latency, cost_model, unit_costs)
    rows = sweep(quality, costs, lambdas)

    base = rows[0]["majority_rate"]          # lambda = 0, the objective used today
    best_row = min(rows, key=lambda r: r["majority_rate"])
    # Two different questions, deliberately reported separately. What matters is whether
    # ANY lambda leaves the decision genuinely mixed (absolute), not whether cost is what
    # made it mixed (relative). Judging on the drop alone calls a 51/49 split "degenerate"
    # whenever it was already 51/49 at lambda=0 — which is exactly backwards.
    return {
        "n_queries": len(qids),
        "metric": mrr_key,
        "cost_model": cost_model,
        "unit_costs": list(unit_costs),
        "static": {a: {"quality": float(quality[:, i].mean()),
                       "cost": float(costs[:, i].mean())}
                   for i, a in enumerate(ACTIONS)},
        "sweep": rows,
        "majority_at_zero": base,
        "min_majority": best_row["majority_rate"],
        "best_lambda": best_row["lambda"],
        "majority_drop": base - best_row["majority_rate"],
    }


def verdict(r, max_majority, min_drop):
    """NON-DEGENERATE if some lambda leaves no single action dominant.

    Primary test is absolute: at the best lambda, is the majority baseline low enough that
    a router has real work to do? The drop from lambda=0 is reported alongside it to say
    whether cost is what created the opportunity or whether it was already there, but a
    problem that is mixed at every lambda is non-degenerate regardless of the drop.

    Returns (verdict, reason). The reason is not decoration: the report prints the drop
    immediately before the verdict, so a run where the absolute test fires on a 0.0% drop
    reads as a contradiction unless the line says which criterion decided it. That
    ambiguity cost a reader an hour and a wrong bug report; naming the criterion is
    cheaper than explaining it again.
    """
    if r["min_majority"] <= max_majority:
        return "NON-DEGENERATE", (f"majority {r['min_majority']:.1%} is at or below the "
                                  f"{max_majority:.0%} bar on its own, independently of cost")
    if r["majority_drop"] >= min_drop:
        return "NON-DEGENERATE (via cost)", (f"cost lowered the majority by "
                                             f"{r['majority_drop']:.1%}, past the "
                                             f"{min_drop:.0%} bar")
    return "DEGENERATE", (f"majority never falls below {max_majority:.0%} "
                          f"(best {r['min_majority']:.1%}) and cost moves it only "
                          f"{r['majority_drop']:.1%}")


def render(all_res, max_majority, min_drop):
    L = ["# Cost-Aware Routing: is the decision non-degenerate?\n",
         "For each lambda, the ORACLE action per query under `MRR - lambda * cost`, and how",
         "concentrated that oracle choice is. `majority` is the accuracy of always picking the",
         "single most common optimal action — the baseline a router has to beat. A large drop",
         "in `majority` as lambda grows means cost genuinely creates a routing decision; a flat",
         "`majority` means it does not, and cost-aware routing is as degenerate as the",
         "quality-only version.\n"]
    for ds, r in all_res.items():
        L += [f"\n## {ds}  (n={r['n_queries']}, metric={r['metric']}, "
              f"cost={r['cost_model']})\n",
              "| lambda | BM25 | Dense | Hybrid | majority | best fixed | headroom | "
              "oracle quality | oracle cost |",
              "|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
        for row in r["sweep"]:
            s = row["action_share"]
            L.append(f"| {row['lambda']:.3f} | {s['bm25']:.0%} | {s['dense']:.0%} | "
                     f"{s['hybrid']:.0%} | **{row['majority_rate']:.1%}** | "
                     f"{row['best_fixed_name']} | {row['headroom']:+.4f} | "
                     f"{row['oracle_quality']:.4f} | {row['oracle_cost']:.3f} |")
        v, why = verdict(r, max_majority, min_drop)
        L.append(f"\nmajority at lambda=0: {r['majority_at_zero']:.1%} -> "
                 f"minimum {r['min_majority']:.1%} at lambda={r['best_lambda']:.3f} "
                 f"(drop {r['majority_drop']:.1%}) => **{v}**")
        L.append(f"\n_Decided by: {why}._\n")
    L += ["\n## How to read the verdict\n",
          f"NON-DEGENERATE means some lambda leaves the majority baseline at or below "
          f"{max_majority:.0%} — no single retriever is optimal often enough to coast on, so "
          "a router has real work to do and real headroom to capture. It does NOT mean the "
          "choice is predictable; that is the separate question the learnability analysis "
          "answers.\n",
          f"'NON-DEGENERATE (via cost)' means the majority baseline was high at lambda=0 but "
          f"fell by at least {min_drop:.0%} once compute was charged — i.e. cost is what "
          "created the routing decision.\n",
          "DEGENERATE means no lambda produces a mixed decision, and the cost-aware "
          "reframing should be dropped rather than tuned.\n",
          "\nOne algebraic caveat worth checking against the table: for any two actions, "
          "`r(a) - r(b) = [MRR(a) - MRR(b)] - lambda*[cost(a) - cost(b)]`, so two retrievers "
          "with EQUAL cost have a lambda-invariant comparison. If the BM25/Dense shares are "
          "frozen across the whole sweep, that is this identity, not a property of the "
          "data.\n"]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-suffix", type=str, default="")
    p.add_argument("--env-dir", type=str, default=None)
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--cost-model", choices=["structural", "measured"], default="structural")
    p.add_argument("--unit-costs", type=str, default="1,3,4",
                   help="Cost of bm25,dense,hybrid under the structural model. Default "
                        "1,3,4: Dense pays a neural query encoding plus ANN search where "
                        "BM25 pays an inverted-index lookup, and Hybrid runs both. Setting "
                        "BM25 and Dense equal (e.g. 1,1,2) makes the analysis vacuous — "
                        "lambda then cancels in their comparison and can only prune "
                        "Hybrid. Sweep this: it is the assumption the conclusion rests on.")
    p.add_argument("--lambdas", type=str, default=None,
                   help="Comma-separated lambdas. Default: 0 plus 12 points log-spaced "
                        "over 0.005-2.0, which brackets the full range from quality-only "
                        "to cost-dominated.")
    p.add_argument("--max-majority", type=float, default=0.65,
                   help="Absolute test: majority rate at or below this at SOME lambda means "
                        "the decision is genuinely mixed (default 0.65).")
    p.add_argument("--threshold", type=float, default=0.10,
                   help="Secondary test: drop in majority rate from lambda=0 that counts as "
                        "cost having created the decision (default 0.10).")
    p.add_argument("--output", type=str,
                   default=os.path.join(RESULTS_DIR, "cost_aware_analysis.json"))
    p.add_argument("--output-md", type=str,
                   default=os.path.join(RESULTS_DIR, "cost_aware_analysis.md"))
    args = p.parse_args()

    unit_costs = [float(x) for x in args.unit_costs.split(",")]
    if len(unit_costs) != len(ACTIONS):
        p.error(f"--unit-costs needs {len(ACTIONS)} values for {ACTIONS}")
    if args.lambdas:
        lambdas = [float(x) for x in args.lambdas.split(",")]
    else:
        lambdas = [0.0] + list(np.logspace(np.log10(0.005), np.log10(2.0), 12))

    paths = get_dataset_paths(args.env_suffix)
    if args.env_dir:
        paths = {k: os.path.join(args.env_dir, os.path.basename(v)) for k, v in paths.items()}
    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(",")}
        paths = {k: v for k, v in paths.items() if k in keep}

    all_res = {}
    for ds, path in paths.items():
        if not os.path.exists(path):
            print(f"  {ds}: env data missing ({path}) — skipped.")
            continue
        r = analyze(ds, path, lambdas, args.cost_model, unit_costs)
        all_res[ds] = r
        v, _why = verdict(r, args.max_majority, args.threshold)
        print(f"\n--- {ds} (n={r['n_queries']}, {r['cost_model']} cost) ---")
        print(f"  majority {r['majority_at_zero']:.1%} (lambda=0) -> {r['min_majority']:.1%} "
              f"at lambda={r['best_lambda']:.3f}   drop={r['majority_drop']:.1%}  => {v}")
        for row in r["sweep"]:
            s = row["action_share"]
            print(f"    lam={row['lambda']:.3f}  bm25={s['bm25']:.0%} dense={s['dense']:.0%} "
                  f"hyb={s['hybrid']:.0%}  majority={row['majority_rate']:.1%}  "
                  f"headroom={row['headroom']:+.4f}")

    if not all_res:
        print("\nNo env data found. Run prepare_env_data first (stage 3).")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(render(all_res, args.max_majority, args.threshold))
    print(f"\nSaved: {args.output}\nSaved: {args.output_md}")

    if all(verdict(r, args.max_majority, args.threshold)[0] == "DEGENERATE"
           for r in all_res.values()):
        print("\nVERDICT: no lambda produces a mixed routing decision on any dataset. "
              "Drop the cost-aware reframing rather than tuning it.")
    else:
        print("\nVERDICT: at least one dataset has a lambda where no single retriever "
              "dominates — the cost-aware decision is real. Whether it is PREDICTABLE is "
              "the separate question escalation_signal_check.py answers.")


if __name__ == "__main__":
    main()
