"""
two_arm_ablation.py — Is the negative result an artifact of a degenerate action space?

The three arms in this repo are not independent. Hybrid is Reciprocal Rank Fusion of BM25 and
Dense over the same corpus, so a third of the action space is a deterministic function of the
other two thirds. That creates two problems the paper has to answer:

  1. It weakens the routing target. If Hybrid is a compromise between the other two, it wins
     mostly on queries where neither is decisively better -- i.e. on near-ties, which are the
     least predictable queries there are. Measured here: Hybrid takes only 12.6% of
     informative queries on the 100K pool.

  2. It inflates the leak. Because Hybrid is a function of BM25 and Dense, leaking those two
     arms' metrics also carries information about Hybrid. Verified separately: a classifier
     given only the two leaked reward columns predicts Hybrid-wins at 89.9% against an 87.4%
     base rate. So the "two of three arms" leak is worth more than two of three, and the
     naive ceiling of 87.4% is not the true ceiling.

A reviewer can therefore ask: does routing fail because routing is hard, or because you gave
it three arms of which one is redundant? This script answers it by re-running the core
learnability measurement on {BM25, Dense} only -- two genuinely distinct retrievers, one
sparse and one embedding-based, with no functional dependence.

The outcome is informative either way, which is why it is worth the compute:

  * Negative again -> the finding is about routing, not about a degenerate action space, and
    the paper is strictly stronger for having checked.
  * Positive -> routing IS learnable between two distinct retrievers and the whole conclusion
    changes. That would be the most interesting possible result, and it is cheap to find.

Also reports the MINIMUM DETECTABLE EFFECT, because "we found no effect" is only meaningful
alongside "and here is the smallest effect we could have found". With a test split of ~150-230
queries and per-query rewards this variable, the MDE can exceed the headroom being chased --
in which case the experiment could not have detected success even if it were there, and
saying so first is far better than having a reviewer say it.

Methodology is imported from learnability_analysis so the two-arm numbers are directly
comparable to the three-arm ones rather than being a separate protocol.

Usage:
    python experiments/router_experiments/two_arm_ablation.py
    python experiments/router_experiments/two_arm_ablation.py --arms bm25,hybrid
"""

import argparse
import json
import math
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ALL_ACTIONS = ["bm25", "dense", "hybrid"]
CV_SEED = 0
# Test-split fraction used by create_train_val_test_split, needed for the MDE calculation.
TEST_RATIO = 0.15
# Two-sided alpha = 0.05, power = 0.80 -> the usual (z_{1-a/2} + z_{1-b})^2 = 7.849
MDE_Z2 = (1.959964 + 0.841621) ** 2


def _pin():
    from learnability_analysis import (N_REPEATS, N_SPLITS, load_queries, reward_matrix)
    return N_SPLITS, N_REPEATS, load_queries, reward_matrix


def predictability(X, R, arms, seed=CV_SEED, n_jobs=1):
    """learnability_analysis.predictability, restricted to a subset of arms.

    Kept structurally identical -- same estimator, same repeated stratified CV, same
    informative-only masking -- so the two-arm accuracy is comparable to the three-arm one.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    N_SPLITS, N_REPEATS, _, _ = _pin()
    mask = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    Xi, yi = X[mask], R[mask].argmax(axis=1)
    if len(yi) < N_SPLITS * 2:
        return None
    counts = np.bincount(yi, minlength=R.shape[1])
    if (counts[counts > 0] < N_SPLITS).any():
        return None

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    s = cross_val_score(GradientBoostingClassifier(random_state=seed), Xi, yi,
                        cv=cv, scoring="accuracy", n_jobs=n_jobs)
    majority = float(counts.max() / len(yi))
    return {
        "arms": list(arms),
        "n_informative": int(len(yi)),
        "class_counts": {a: int(c) for a, c in zip(arms, counts)},
        "majority": majority,
        "gbt": float(s.mean()),
        "gbt_std": float(s.std()),
        "lift": float(s.mean()) - majority,
    }


def corpus_stats(R, arms):
    means = R.mean(axis=0)
    best = int(means.argmax())
    oracle = float(R.max(axis=1).mean())
    strict = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    return {
        "arms": list(arms),
        "n": int(len(R)),
        "per_arm": {a: float(m) for a, m in zip(arms, means)},
        "best_fixed_arm": arms[best],
        "best_fixed": float(means[best]),
        "oracle": oracle,
        "headroom": oracle - float(means[best]),
        "tie_rate": float(1.0 - strict.mean()),
        "learnable_rate": float(strict.mean()),
    }


def mde(R, arms, test_ratio=TEST_RATIO):
    """Smallest per-query reward difference detectable on a test split of this size.

    Paired design: the comparison is method-vs-best-fixed on the SAME queries, so the
    relevant variance is that of the per-query difference, not of the rewards themselves.
    The best case for a router is that it differs from best-fixed only on queries where some
    other arm wins, so we use the per-query oracle gain as the difference series -- it is the
    largest difference any router could produce, which makes this an optimistic MDE.
    """
    n_test = max(1, int(round(len(R) * test_ratio)))
    best = int(R.mean(axis=0).argmax())
    diff = R.max(axis=1) - R[:, best]
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
    detectable = math.sqrt(MDE_Z2 * sd ** 2 / n_test) if n_test > 0 else float("nan")
    headroom = float(diff.mean())
    # The useful quantity is NOT "is a perfect oracle router detectable" -- it trivially is,
    # since the oracle gain is the largest possible effect. It is "what is the smallest
    # FRACTION of the available headroom a router must capture before this test split could
    # tell it apart from the best fixed retriever". Anything below that is invisible here,
    # real or not, and that is the number the limitations section has to quote.
    min_capture = detectable / headroom if headroom > 0 else float("nan")
    return {
        "arms": list(arms),
        "n_test": n_test,
        "diff_sd": sd,
        "mde_absolute": detectable,
        "headroom": headroom,
        "min_detectable_capture": min_capture,
        "perfect_oracle_detectable": bool(detectable <= headroom),
    }


def render_markdown(res, arms_label):
    L = ["# Two-Arm Ablation and Statistical Power\n",
         f"_Restricted action space: {arms_label}_\n",
         "\nHybrid is RRF of BM25 and Dense, so the three-arm action space contains a "
         "redundant arm. If routing fails only because one of three arms is a function of "
         "the other two, that is a setup problem rather than a finding. This re-runs the core "
         "measurement on two genuinely distinct retrievers.\n",
         "\n## Table 1 — Three arms vs. two arms\n",
         "| dataset | arms | n | best fixed | oracle | headroom | tie rate | informative n | majority | GBT | lift |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds, e in res.items():
        for tag in ("three", "two"):
            c, p = e[tag]["corpus"], e[tag]["pred"]
            arms = "+".join(c["arms"])
            if p is None:
                L.append(f"| {ds} | {arms} | {c['n']} | {c['best_fixed']:.4f} | "
                         f"{c['oracle']:.4f} | +{c['headroom']:.4f} | "
                         f"{c['tie_rate']*100:.1f}% | too few | — | — | — |")
                continue
            L.append(f"| {ds} | {arms} | {c['n']} | {c['best_fixed_arm']} {c['best_fixed']:.4f} | "
                     f"{c['oracle']:.4f} | +{c['headroom']:.4f} | {c['tie_rate']*100:.1f}% | "
                     f"{p['n_informative']} | {p['majority']*100:.1f}% | "
                     f"{p['gbt']*100:.1f}% ± {p['gbt_std']*100:.1f} | "
                     f"**{p['lift']*100:+.1f}%** |")
    L += ["\n`lift` is GBT accuracy minus the majority-class baseline. A negative lift on BOTH "
          "action spaces means the negative result is not an artifact of Hybrid's redundancy: "
          "it holds between two retrievers with no functional dependence on each other.\n",
          "\n## Table 2 — Minimum detectable effect\n",
          "| dataset | arms | n test | sd of per-query gain | MDE | headroom | must capture at least |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for ds, e in res.items():
        for tag in ("three", "two"):
            m = e[tag]["mde"]
            L.append(f"| {ds} | {'+'.join(m['arms'])} | {m['n_test']} | {m['diff_sd']:.4f} | "
                     f"{m['mde_absolute']:.4f} | {m['headroom']:.4f} | "
                     f"**{m['min_detectable_capture']*100:.0f}%** of headroom |")
    L += ["\nSmallest mean per-query improvement over the best fixed retriever that a test "
          "split of this size could detect at alpha = 0.05 with 80% power, in a paired design. "
          "The last column is the MDE as a fraction of available headroom, and it is the "
          "number the limitations section should quote: **a router capturing less than that "
          "share of the oracle headroom cannot be distinguished from the best fixed retriever "
          "on a test split this size, whether or not it is genuinely better.**\n",
          "\nThis is why the column is expressed as a share rather than as a yes/no verdict. "
          "Asking whether a *perfect* oracle router would be detectable is not informative — "
          "the oracle gain is by construction the largest effect available, so the answer is "
          "almost always yes. The binding question is how modest a real improvement could be "
          "before this design goes blind to it, and here that threshold is high: roughly half "
          "to four fifths of all available headroom.\n",
          "\nNote also that the difference series used is the per-query oracle gain, the "
          "largest difference any router could produce. A real router differs from best-fixed "
          "on fewer queries and by less, so its per-query variance is lower but its mean "
          "effect is much lower too; the share above is therefore an optimistic floor.\n"]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", type=str, default="bm25,dense",
                   help="Two arms to keep for the restricted comparison.")
    p.add_argument("--env-suffix", type=str, default="")
    p.add_argument("--env-dir", type=str, default=None)
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Fold parallelism. 1 is safe everywhere; -1 is faster where joblib's "
                        "worker pool survives (see permutation_control._cv_score).")
    p.add_argument("--outdir", type=str, default=RESULTS_DIR)
    args = p.parse_args()

    keep = [a.strip() for a in args.arms.split(",") if a.strip()]
    if len(keep) != 2 or any(a not in ALL_ACTIONS for a in keep):
        sys.exit(f"--arms must name exactly two of {ALL_ACTIONS}; got {keep}")
    idx = [ALL_ACTIONS.index(a) for a in keep]

    _, _, load_queries, reward_matrix = _pin()
    env_dir = args.env_dir or os.path.join(PROJECT_ROOT, "src", "router")

    res = {}
    for ds in ("nq", "triviaqa", "hotpotqa"):
        path = os.path.join(env_dir, f"{ds}_env_data{args.env_suffix}.json")
        if not os.path.exists(path):
            print(f"  {ds}: env data missing at {path} — skipped.")
            continue
        _, X, R = reward_matrix(load_queries(path))
        R2 = R[:, idx]
        res[ds] = {
            "three": {"corpus": corpus_stats(R, ALL_ACTIONS),
                      "pred": predictability(X, R, ALL_ACTIONS, n_jobs=args.n_jobs),
                      "mde": mde(R, ALL_ACTIONS)},
            "two": {"corpus": corpus_stats(R2, keep),
                    "pred": predictability(X, R2, keep, n_jobs=args.n_jobs),
                    "mde": mde(R2, keep)},
        }
        # predictability returns None when stratified CV cannot run -- fewer than N_SPLITS
        # members in some class, which is normal on a --smoke run and must not crash the
        # stage. The markdown renderer already handles None; the progress line has to too.
        t, w = res[ds]["three"]["pred"], res[ds]["two"]["pred"]
        if t is None or w is None:
            print(f"  {ds}: too few informative queries per class for stratified CV "
                  f"(3-arm={'ok' if t else 'skipped'}, 2-arm={'ok' if w else 'skipped'}); "
                  f"corpus stats and MDE still reported.")
        else:
            print(f"  {ds}: 3-arm lift {t['lift']*100:+.1f}% -> "
                  f"2-arm lift {w['lift']*100:+.1f}%  "
                  f"(majority {t['majority']*100:.1f}% -> {w['majority']*100:.1f}%)")

    if not res:
        print("No env data found.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    label = "+".join(keep)
    js = os.path.join(args.outdir, "two_arm_ablation.json")
    md = os.path.join(args.outdir, "two_arm_ablation.md")
    with open(js, "w", encoding="utf-8") as f:
        json.dump({"arms": keep, "cv_seed": CV_SEED, "test_ratio": TEST_RATIO,
                   "per_dataset": res}, f, indent=2)
    with open(md, "w", encoding="utf-8") as f:
        f.write(render_markdown(res, label))

    print()
    print(render_markdown(res, label))
    print(f"Saved: {js}\nSaved: {md}")


if __name__ == "__main__":
    main()
