"""
permutation_control.py — The control that separates "no signal" from "not enough data".

The negative result in `learnability_analysis.py` is stated as a *lift*: honest GBT accuracy
minus the majority-class baseline, which is negative on every dataset. That comparison
answers "is the router useful?" (no) but it does NOT answer "do the features carry any
routing signal at all?" — because a classifier can score below the majority baseline while
still extracting real signal, simply by spending capacity on a minority class it sometimes
gets right. Negative lift and zero signal are different claims, and `learning_curve.py`
cannot tell them apart either: a rising curve is equally consistent with

  (a) real signal, still data-limited — accuracy climbs toward something above majority, or
  (b) zero signal — accuracy climbs toward the majority baseline from below, which is
      exactly what an overfitting classifier on noise features does as n grows.

Two nulls settle it, and they are the standard pair:

  LABEL PERMUTATION — shuffle the best-retriever labels, keeping the class prior intact,
      and re-run the identical CV. This destroys any feature-label relationship while
      preserving everything else (class balance, feature distribution, n, CV protocol).
      If the honest accuracy is inside this null distribution, the features are
      indistinguishable from noise and claim (b) is supported by evidence rather than by
      the absence of evidence.

  RANDOM FEATURES — replace the feature matrix with standard Gaussian noise of identical
      shape, keeping the true labels. This measures the floor the *pipeline* reports when
      handed nothing, which is not the same number as the majority baseline: a GBT given
      noise features and imbalanced labels lands somewhat below majority, and knowing
      where is what makes the honest number interpretable.

Reported as a one-sided permutation p-value with the (1 + k) / (1 + B) correction, so the
p-value can never be exactly 0 — with B permutations the smallest attainable value is
1/(1+B), and reporting "p < 1/(1+B)" rather than "p = 0" is the honest statement.

Methodology is imported from `learnability_analysis` rather than restated, for the same
reason `leak_gradient.py` does it: if the CV protocol there changes, this control must
change with it or the comparison is invalid.

Usage:
    python experiments/router_experiments/permutation_control.py
    python experiments/router_experiments/permutation_control.py --n-perm 50
    python experiments/router_experiments/permutation_control.py --env-dir /path/to/env
"""

import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CV_SEED = 0

# The protocol constants and loaders are pinned to learnability_analysis -- if the CV design
# there changes, this control must change with it or the comparison is invalid. They are
# bound lazily by _pin() rather than imported at module scope, because canonical_evaluation
# (reached transitively) does `import torch` at module level, and joblib's loky workers
# re-import this script to unpickle their payload. On a machine that cannot load torch's
# CUDA DLLs that import raises inside every worker and the whole run dies with
# BrokenProcessPool. Keeping module import cheap is what lets --n-jobs -1 work at all.
ACTIONS = N_REPEATS = N_SPLITS = load_queries = reward_matrix = None


def _pin():
    """Bind the pinned protocol into module globals. Call once, from main()."""
    global ACTIONS, N_REPEATS, N_SPLITS, load_queries, reward_matrix
    if ACTIONS is not None:
        return
    from learnability_analysis import (ACTIONS as _A, N_REPEATS as _R, N_SPLITS as _S,
                                       load_queries as _lq, reward_matrix as _rm)
    ACTIONS, N_REPEATS, N_SPLITS, load_queries, reward_matrix = _A, _R, _S, _lq, _rm


def get_dataset_paths(suffix=""):
    """Local copy of canonical_evaluation.get_dataset_paths, to keep torch out of workers.

    Asserted against the real one in main() so the two cannot silently diverge.
    """
    return {ds: os.path.join(PROJECT_ROOT, "src", "router", f"{ds}_env_data{suffix}.json")
            for ds in ("nq", "triviaqa", "hotpotqa")}


def _cv_score(X, y, seed=CV_SEED, n_jobs=-1):
    """One accuracy estimate under the exact protocol of learnability_analysis.predictability.

    n_jobs is a knob rather than a constant because joblib's default backend forks worker
    processes that re-import this module's transitive dependencies -- including torch, via
    canonical_evaluation. On a machine whose paging file cannot host torch's CUDA DLLs that
    import raises inside the worker and surfaces as BrokenProcessPool, killing the run.
    cross_val_score only parallelises across folds, so n_jobs changes wall time and not the
    result -- checked for this same GBT + RepeatedStratifiedKFold protocol in
    learnability_analysis, where both settings returned 0.6008201892744479. Pass --n-jobs 1
    if the pool breaks.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    s = cross_val_score(GradientBoostingClassifier(random_state=seed), X, y,
                        cv=cv, scoring="accuracy", n_jobs=n_jobs)
    return float(s.mean())


def informative_subset(X, R):
    """The queries learnability_analysis actually scores: those with a strictly-best action.

    Ties are excluded there because argmax on a tie is a tie-break artifact, not a label.
    This control has to use the identical subset or it is nulling a different quantity.
    """
    mask = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    return X[mask], R[mask].argmax(axis=1)


def run_controls(X, R, n_perm, label="", n_jobs=-1):
    Xi, yi = informative_subset(X, R)
    counts = np.bincount(yi, minlength=len(ACTIONS))
    if len(yi) < N_SPLITS * 2 or (counts[counts > 0] < N_SPLITS).any():
        return None

    majority = float(counts.max() / len(yi))
    t0 = time.time()
    observed = _cv_score(Xi, yi, n_jobs=n_jobs)
    per_run = time.time() - t0
    print(f"  {label}: n={len(yi)} dim={Xi.shape[1]} majority={majority*100:.2f}% "
          f"observed={observed*100:.2f}%  ({per_run:.0f}s/run, "
          f"~{per_run*2*n_perm/60:.0f} min for {n_perm}x2 controls)")

    # Each replicate gets its own permutation but the same CV seed, so the only thing
    # varying between replicates is the label assignment.
    perm_null = []
    for b in range(n_perm):
        rng = np.random.default_rng(1000 + b)
        perm_null.append(_cv_score(Xi, rng.permutation(yi), n_jobs=n_jobs))
    perm_null = np.array(perm_null)

    rand_null = []
    for b in range(n_perm):
        rng = np.random.default_rng(2000 + b)
        rand_null.append(_cv_score(rng.standard_normal(Xi.shape), yi, n_jobs=n_jobs))
    rand_null = np.array(rand_null)

    # One-sided: how often does a null run match or beat the real features?
    k_perm = int((perm_null >= observed).sum())
    k_rand = int((rand_null >= observed).sum())
    out = {
        "n_informative": int(len(yi)),
        "feature_dim": int(Xi.shape[1]),
        "majority": majority,
        "observed": observed,
        "observed_lift": observed - majority,
        "n_perm": n_perm,
        "perm_null_mean": float(perm_null.mean()),
        "perm_null_std": float(perm_null.std()),
        "perm_null_max": float(perm_null.max()),
        "perm_p": (1 + k_perm) / (1 + n_perm),
        "perm_k": k_perm,
        "signal_above_perm": observed - float(perm_null.mean()),
        "rand_null_mean": float(rand_null.mean()),
        "rand_null_std": float(rand_null.std()),
        "rand_null_max": float(rand_null.max()),
        "rand_p": (1 + k_rand) / (1 + n_perm),
        "rand_k": k_rand,
        "signal_above_rand": observed - float(rand_null.mean()),
    }
    print(f"    label-perm null {out['perm_null_mean']*100:.2f}% "
          f"± {out['perm_null_std']*100:.2f}  p={out['perm_p']:.3f}  "
          f"signal={out['signal_above_perm']*100:+.2f} pts")
    print(f"    random-feat null {out['rand_null_mean']*100:.2f}% "
          f"± {out['rand_null_std']*100:.2f}  p={out['rand_p']:.3f}  "
          f"signal={out['signal_above_rand']*100:+.2f} pts")
    return out


def verdict(e):
    """The sentence the paper is entitled to write, given this dataset's numbers.

    Deliberately keyed off the permutation p-value and not off the sign of the lift: the
    whole point of this script is that those two can disagree, and when they do it is the
    p-value that speaks to whether signal exists.
    """
    if e is None:
        return "too few informative queries to null"
    if e["perm_p"] > 0.10:
        return ("indistinguishable from permuted labels — no detectable routing signal; "
                "the negative result is a signal problem, not a sample-size problem")
    if e["observed_lift"] < 0:
        return ("detectable signal (beats permuted labels) but still below the majority "
                "baseline — signal exists and is too weak to exploit; scale may help")
    return "detectable signal and above majority — routing is learnable at this scale"


def render_markdown(per_ds, pooled, tag, n_perm):
    L = [f"# Permutation and Random-Feature Controls\n", f"_Scale: {tag} · B={n_perm}_\n",
         "\nDoes the honest feature set carry routing signal at all? The lift in "
         "`learnability_analysis.md` cannot answer this — it compares against the majority "
         "baseline, not against chance. These two nulls compare against chance.\n",
         "\n## Table 1 — Honest features vs. two nulls\n",
         "| dataset | n inf | majority | observed | label-perm null | p | random-feat null | p |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    rows = list(per_ds.items()) + ([("**pooled**", pooled)] if pooled else [])
    for ds, e in rows:
        if e is None:
            L.append(f"| {ds} | — | — | — | — | — | — | — |")
            continue
        L.append(f"| {ds} | {e['n_informative']} | {e['majority']*100:.1f}% | "
                 f"**{e['observed']*100:.1f}%** | "
                 f"{e['perm_null_mean']*100:.1f}% ± {e['perm_null_std']*100:.1f} | "
                 f"{e['perm_p']:.3f} | "
                 f"{e['rand_null_mean']*100:.1f}% ± {e['rand_null_std']*100:.1f} | "
                 f"{e['rand_p']:.3f} |")
    L += [f"\n`observed` is GBT accuracy on the honest features ({N_SPLITS}-fold stratified "
          f"CV × {N_REPEATS} repeats), identical to the GBT column of "
          "`learnability_analysis.md`. `label-perm null` shuffles the best-retriever labels "
          f"across {n_perm} replicates, preserving the class prior; `random-feat null` "
          "replaces the features with Gaussian noise and keeps the true labels. "
          f"p-values are one-sided with the (1+k)/(1+B) correction, so the floor is "
          f"{1/(1+n_perm):.3f}.\n",
          "\n## Table 2 — What each dataset licenses the paper to say\n",
          "| dataset | signal above chance | verdict |", "|---|---:|---|"]
    for ds, e in rows:
        sig = f"{e['signal_above_perm']*100:+.1f} pts" if e else "—"
        L.append(f"| {ds} | {sig} | {verdict(e)} |")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-suffix", type=str, default="")
    p.add_argument("--env-dir", type=str, default=None)
    p.add_argument("--n-perm", type=int, default=20,
                   help="Replicates per null. 20 gives a p-value floor of 0.048.")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Fold-level parallelism. Use 1 if joblib's worker pool breaks on "
                        "importing torch (see _cv_score); results are unaffected.")
    p.add_argument("--tag", type=str, default="full run")
    p.add_argument("--outdir", type=str, default=RESULTS_DIR)
    p.add_argument("--datasets", type=str, default=None,
                   help="Comma-separated subset (e.g. nq). Each dataset's nulls are "
                        "independent, so running them in separate processes is equivalent to "
                        "one combined run and keeps peak memory low enough for the worker "
                        "pool to survive. Pooled is computed only when all three are present.")
    args = p.parse_args()

    _pin()
    # Guard the local get_dataset_paths against drift from the canonical one. Safe to import
    # torch here: main() does not execute in spawned workers, so this stays in the parent.
    from canonical_evaluation import get_dataset_paths as _canonical_paths
    assert get_dataset_paths(args.env_suffix) == _canonical_paths(args.env_suffix), (
        "local get_dataset_paths has diverged from canonical_evaluation's")

    paths = get_dataset_paths(args.env_suffix)
    if args.env_dir:
        paths = {k: os.path.join(args.env_dir, os.path.basename(v)) for k, v in paths.items()}
    n_all = len(paths)
    if args.datasets:
        want = [d.strip() for d in args.datasets.split(",") if d.strip()]
        unknown = [d for d in want if d not in paths]
        if unknown:
            sys.exit(f"unknown dataset(s): {', '.join(unknown)}; choose from {list(paths)}")
        paths = {d: paths[d] for d in want}

    print(f"Permutation controls ({args.tag}), B={args.n_perm} per null...")
    per_ds, pooled_X, pooled_R = {}, [], []
    for ds, path in paths.items():
        if not os.path.exists(path):
            print(f"  {ds}: env data missing at {path} — skipped.")
            continue
        _, X, R = reward_matrix(load_queries(path))
        per_ds[ds] = run_controls(X, R, args.n_perm, label=ds, n_jobs=args.n_jobs)
        pooled_X.append(X)
        pooled_R.append(R)

    if not per_ds:
        print("No env data found.")
        sys.exit(1)

    pooled = None
    if len(pooled_X) < n_all:
        print(f"  Pooled skipped: {len(pooled_X)} of {n_all} datasets in this run. Pooling a "
              f"subset would report a different quantity than the full pooled null.")
    elif pooled_X and len({X.shape[1] for X in pooled_X}) == 1:
        pooled = run_controls(np.vstack(pooled_X), np.vstack(pooled_R), args.n_perm,
                              label="pooled", n_jobs=args.n_jobs)
    elif pooled_X:
        detail = ", ".join(f"{ds}={X.shape[1]}" for ds, X in zip(per_ds, pooled_X))
        print(f"  Pooled skipped: feature widths differ ({detail}).")

    # B is the parameter that changes the answer -- it fixes the attainable p-floor at
    # (1+0)/(1+B) -- so it belongs in the path, not only inside the file. A B=20 run previously
    # landed at results/permutation_control.json reporting p=0.048 for NQ, which IS that floor;
    # the B=1000 rerun put the same cell at 0.066. Two runs that disagree must not be able to
    # occupy the same filename, or the figures and the text drift apart silently.
    outdir = os.path.join(args.outdir, f"perm_B{args.n_perm}")
    os.makedirs(outdir, exist_ok=True)
    # A subset run is a partial result; give it its own filename so it cannot be mistaken
    # for -- or overwrite -- the full three-dataset artifact.
    part = "" if len(paths) == n_all else "_" + "-".join(paths)
    js = os.path.join(outdir, f"permutation_control{part}.json")
    md = os.path.join(outdir, f"permutation_control{part}.md")
    with open(js, "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "n_perm": args.n_perm, "cv_seed": CV_SEED,
                   "n_jobs": args.n_jobs, "per_dataset": per_ds, "pooled": pooled},
                  f, indent=2)
    with open(md, "w", encoding="utf-8") as f:
        f.write(render_markdown(per_ds, pooled, args.tag, args.n_perm))

    print()
    print(render_markdown(per_ds, pooled, args.tag, args.n_perm))
    print(f"Saved: {js}\nSaved: {md}")


if __name__ == "__main__":
    main()
