"""Does routing predictability improve with more data, or is it bias-limited?

`learnability_analysis.py` reports that no honest feature set beats the majority-class
baseline (pooled lift -2.0%). That result is only actionable if we know *why*: a model
that is starving for data would improve as n grows, and the right response would be to
collect more queries. A model whose optimum is a constant function will not improve at
any n, and collecting more queries would be wasted compute.

This script decides between those two by subsampling the informative queries we already
have and re-running the identical classifier at each size. It reads existing env data and
trains nothing new -- there is no reason for it to touch the pipeline.

Reading the output:

  flat / plateauing   the barrier is the state representation, not sample size. More
                      queries will not help, and the curve is the evidence for saying so.
  trending upward     predictability is still improving at our current n, and scaling the
                      query count is worth its compute.

Verification: at the largest size the curve must reproduce the pooled numbers already in
results/learnability_analysis.json. `--verify-against` enforces that and exits non-zero on
mismatch, so a methodology drift between this script and learnability_analysis.py cannot
pass silently.

Usage:
    python learning_curve.py                        # uses src/router env data
    python learning_curve.py --env-dir /path/to/env # e.g. data extracted from a git tag
    python learning_curve.py --verify-against results/learnability_analysis.json
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATASETS = ["nq", "triviaqa", "hotpotqa"]
ACTIONS = ["bm25", "dense", "hybrid"]

# Kept identical to learnability_analysis.py -- if these drift, the curve stops being
# comparable to the number it is supposed to explain.
N_SPLITS = 5
N_REPEATS = 5

DEFAULT_SIZES = [250, 500, 1000, 1500, 2000]
DEFAULT_TRIALS = 3      # subsample draws per size; the spread across them is the error band


def load_pooled(env_dir, metric="mrr"):
    """Pooled feature matrix and reward matrix, stacked in DATASETS order.

    Mirrors analyze()'s pooling in learnability_analysis.py: vstack of the per-dataset
    arrays, no reweighting, so a dataset contributes in proportion to its query count.
    """
    Xs, Rs, per_ds = [], [], {}
    for ds in DATASETS:
        path = os.path.join(env_dir, f"{ds}_env_data.json")
        if not os.path.exists(path):
            print(f"  {ds}: env data missing at {path} -- skipped.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            env = json.load(f)
        queries = env.get("queries", env)
        qids = list(queries)
        X = np.array([queries[q]["features"] for q in qids], dtype=float)
        R = np.array([[float(queries[q]["retrievers"][a][metric]) for a in ACTIONS]
                      for q in qids])
        per_ds[ds] = (X, R)
        Xs.append(X)
        Rs.append(R)
    if not Xs:
        raise SystemExit(f"No env data found under {env_dir}")

    # Datasets may produce different feature counts (e.g. 46 vs 57 columns)
    # depending on corpus size.  Zero-pad narrower arrays to the widest so
    # np.vstack succeeds.  Extra zero-columns carry no signal and do not
    # affect the classifiers.
    max_cols = max(x.shape[1] for x in Xs)
    Xs_padded = []
    for x in Xs:
        if x.shape[1] < max_cols:
            pad = np.zeros((x.shape[0], max_cols - x.shape[1]), dtype=x.dtype)
            Xs_padded.append(np.hstack([x, pad]))
        else:
            Xs_padded.append(x)

    return np.vstack(Xs_padded), np.vstack(Rs), per_ds


def informative(X, R):
    """Restrict to queries with a strictly-best action.

    On a tie there is no correct label, so scoring a model on them measures the tie-break
    rather than the model. Same mask learnability_analysis.predictability() applies.
    """
    mask = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    return X[mask], R[mask].argmax(axis=1)


def evaluate(Xi, yi, seed=0, n_jobs=-1):
    """Accuracy and majority baseline on one (sub)sample, via repeated stratified CV.

    n_jobs=-1 because the curve is ~475 gradient-boosting fits and cross_val_score is
    single-threaded by default. Left at the default this takes hours on one core; the
    folds are independent, so there is no reason to run them serially.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    counts = np.bincount(yi, minlength=len(ACTIONS))
    if len(yi) < N_SPLITS * 2 or (counts[counts > 0] < N_SPLITS).any():
        return None

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    majority = float(counts.max() / len(yi))

    gbt = cross_val_score(GradientBoostingClassifier(random_state=seed), Xi, yi,
                          cv=cv, scoring="accuracy", n_jobs=n_jobs)
    lr = cross_val_score(LogisticRegression(max_iter=3000), Xi, yi,
                         cv=cv, scoring="accuracy", n_jobs=n_jobs)
    return {
        "n": int(len(yi)),
        "majority": majority,
        "gbt": float(gbt.mean()), "gbt_std": float(gbt.std()),
        "logreg": float(lr.mean()), "logreg_std": float(lr.std()),
        "gbt_lift": float(gbt.mean()) - majority,
        "logreg_lift": float(lr.mean()) - majority,
    }


def subsample(Xi, yi, n, seed):
    """Stratified subsample of size n, so the majority baseline stays comparable.

    Without stratification the class balance drifts between sizes and the lift curve
    would partly measure that drift rather than learnability.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    if n >= len(yi):
        return Xi, yi
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(sss.split(Xi, yi))
    return Xi[idx], yi[idx]


def build_curve(Xi, yi, sizes, trials, n_jobs=-1):
    """One point per size: mean and spread of lift across `trials` subsample draws."""
    points = []
    for n in sizes:
        if n > len(yi):
            continue
        runs = [r for r in (evaluate(*subsample(Xi, yi, n, seed=1000 + t), n_jobs=n_jobs)
                            for t in range(trials)) if r]
        if not runs:
            continue
        agg = {"n": n, "n_trials": len(runs)}
        for key in ("majority", "gbt", "logreg", "gbt_lift", "logreg_lift"):
            vals = [r[key] for r in runs]
            agg[key] = float(np.mean(vals))
            agg[f"{key}_spread"] = float(np.std(vals))
        points.append(agg)
        print(f"  n={n:5d}  majority={agg['majority']*100:5.1f}%  "
              f"GBT={agg['gbt']*100:5.1f}%  lift={agg['gbt_lift']*100:+5.1f}%"
              f"  (+/-{agg['gbt_lift_spread']*100:.1f} across {len(runs)} draws)")

    full = evaluate(Xi, yi, n_jobs=n_jobs)   # no subsampling -- the verification anchor
    if full:
        full["n_trials"] = 1
        for key in ("majority", "gbt", "logreg", "gbt_lift", "logreg_lift"):
            full[f"{key}_spread"] = 0.0
        points.append(full)
        print(f"  n={full['n']:5d}  majority={full['majority']*100:5.1f}%  "
              f"GBT={full['gbt']*100:5.1f}%  lift={full['gbt_lift']*100:+5.1f}%  (full set)")
    return points


def verify(points, reference_path, tol=1e-6):
    """The full-set point must match learnability_analysis.json, or the curve is not
    measuring the same quantity as the result it explains."""
    if not os.path.exists(reference_path):
        print(f"\n[verify] reference not found at {reference_path} -- skipped.")
        return None
    with open(reference_path, "r", encoding="utf-8") as f:
        ref = json.load(f)
    pooled = (ref.get("pooled") or {}).get("honest")
    if not pooled:
        print("\n[verify] reference has no pooled.honest block -- skipped.")
        return None

    full = points[-1]
    checks = {
        "n_informative": (full["n"], pooled["n_informative"]),
        "majority": (full["majority"], pooled["majority"]),
        "gbt": (full["gbt"], pooled["gbt"]),
        "gbt_lift": (full["gbt_lift"], pooled["gbt_lift"]),
    }
    print("\n[verify] full-set point vs results/learnability_analysis.json")
    ok = True
    for name, (got, want) in checks.items():
        match = abs(got - want) <= tol if isinstance(want, float) else got == want
        ok &= match
        print(f"  {'OK  ' if match else 'FAIL'}  {name}: got {got}  want {want}")
    return ok


def render_markdown(points, verified, tag):
    trend = points[-1]["gbt_lift"] - points[0]["gbt_lift"]
    L = ["# Learning Curve: is routing predictability bias-limited?", "",
         f"_Scale: {tag}. Accuracy predicting the strictly-best retriever, "
         f"{N_SPLITS}-fold stratified CV x {N_REPEATS} repeats, subsampled stratified._", "",
         "| informative n | majority | GBT | lift | spread across draws |",
         "|---:|---:|---:|---:|---:|"]
    for p in points:
        L.append(f"| {p['n']} | {p['majority']*100:.1f}% | {p['gbt']*100:.1f}% | "
                 f"**{p['gbt_lift']*100:+.1f}%** | +/-{p['gbt_lift_spread']*100:.1f} |")
    L += ["",
          f"Lift moves {trend*100:+.1f} points between the smallest and largest sample. ",
          "",
          "A flat or plateauing curve means the barrier is the state representation rather "
          "than sample size: collecting more queries would not change the conclusion. A "
          "curve still climbing at the largest n would mean the opposite, and would make "
          "scaling the query count worth its compute.", ""]
    if verified is not None:
        L += [f"Full-set point reproduces `results/learnability_analysis.json`: "
              f"**{'yes' if verified else 'NO -- methodology drift'}**.", ""]
    return "\n".join(L)


def render_pdf(points, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable -- skipping figure.")
        return
    ns = [p["n"] for p in points]
    lift = [p["gbt_lift"] * 100 for p in points]
    spread = [p["gbt_lift_spread"] * 100 for p in points]

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.axhline(0, color="#8C8C8C", lw=1, ls="--", zorder=1)
    ax.fill_between(ns, np.array(lift) - np.array(spread), np.array(lift) + np.array(spread),
                    color="#4C72B0", alpha=0.18, zorder=2)
    ax.plot(ns, lift, "o-", color="#4C72B0", lw=1.8, ms=5, zorder=3,
            label="GBT lift over majority baseline")
    ax.set_xlabel("informative queries (n)")
    ax.set_ylabel("lift over majority baseline (pp)")
    ax.set_title("Routing predictability does not improve with sample size"
                 if lift[-1] <= lift[0] + 0.5 else
                 "Routing predictability vs sample size")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-dir", default=os.path.join(PROJECT_ROOT, "src", "router"),
                    help="directory holding {dataset}_env_data.json")
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                    help="comma-separated subsample sizes")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                    help="subsample draws per size (spread across them is the error band)")
    ap.add_argument("--metric", default="mrr")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="parallel CV workers (-1 = all cores). The curve is ~475 "
                         "gradient-boosting fits; serial takes hours.")
    ap.add_argument("--verify-against",
                    default=os.path.join(RESULTS_DIR, "learnability_analysis.json"),
                    help="learnability_analysis.json to reproduce at the full-set point")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--tag", default="pooled, real_wikipedia gold")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading env data from {args.env_dir}")
    X, R, _ = load_pooled(args.env_dir, args.metric)
    Xi, yi = informative(X, R)
    print(f"  pooled: {len(X)} queries, {len(yi)} informative "
          f"({len(yi)/len(X)*100:.1f}%), {X.shape[1]} features\n")

    print("Building curve:")
    points = build_curve(Xi, yi, sizes, args.trials, n_jobs=args.n_jobs)
    verified = verify(points, args.verify_against)

    payload = {"tag": args.tag, "metric": args.metric, "n_splits": N_SPLITS,
               "n_repeats": N_REPEATS, "trials": args.trials,
               "verified_against_learnability": verified, "points": points}
    jpath = os.path.join(args.out_dir, "learning_curve.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {jpath}")

    mpath = os.path.join(args.out_dir, "learning_curve.md")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(render_markdown(points, verified, args.tag))
    print(f"  wrote {mpath}")

    render_pdf(points, os.path.join(args.out_dir, "learning_curve.pdf"))

    if verified is False:
        raise SystemExit("Verification FAILED: curve does not reproduce "
                         "learnability_analysis.json at the full-set point.")


if __name__ == "__main__":
    main()
