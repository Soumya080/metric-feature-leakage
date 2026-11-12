"""How much contamination does it take? A dose-response curve for feature leakage.

learnability_analysis.py answers whether contamination matters: leaky features reach 73.2%
best-retriever prediction accuracy where honest ones reach 60.1%, against a 61.4% majority
baseline. That establishes the effect. It does not establish the DOSE, which is the question
anyone auditing their own router actually has -- "I did not copy the MRR column, so am I
fine?"

This runs the identical classifier across every feature-mode arm, so the arms differ ONLY in
features 22-25 and the accuracy difference is attributable to nothing else:

  honest_minimal   raw top-5 score mean/std -- label-free, the shipped fix
  leak_noisy       MRR plus deterministic jitter. Evades exact-equality leak detection
                    entirely, so if it still buys accuracy then "no feature equals the
                    reward" is not a sufficient defence
  leak_hit         Hit@k instead of MRR -- does the leak need the reward's value, or only
                    its sign?
  leak_one_arm     only BM25's MRR leaks; Dense stays honest. The half-finished fix
  leaky            both arms' MRR -- the original bug, full strength

Deliberately skips the 7-group feature breakdown that learnability_analysis runs, because
that is a per-arm cost multiplier and the question here is one number per arm. Uses
n_jobs=-1: learnability_analysis is single-threaded and took 41 minutes for two arms.

No RL training. leakage_ablation.py answers the same question on MRR-after-training, but it
invokes canonical_evaluation per arm at ~86 minutes each. This is the cheap path to the
number the paper actually quotes.

Integrity: the arms must share identical reward tables. If they do not, they were built from
different retrieval runs and no comparison between them means anything -- checked, not
assumed.

Usage:
    python leak_gradient.py
    python leak_gradient.py --verify-against results/learnability_analysis.json
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from learnability_analysis import ACTIONS, N_REPEATS, N_SPLITS, load_queries  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ROUTER_DIR = os.path.join(PROJECT_ROOT, "src", "router")
DATASETS = ["nq", "triviaqa", "hotpotqa"]

# Arm -> env-data suffix, ordered from no leak to full leak so the table reads as a curve.
ARMS = [
    ("honest_minimal", "", "raw top-5 score mean/std (label-free)"),
    ("leak_noisy", "_leaknoisy", "MRR + deterministic jitter (evades exact-match detection)"),
    ("leak_hit", "_leakhit", "Hit@k instead of MRR (sign, not value)"),
    ("leak_one_arm", "_leak1", "BM25's MRR only; Dense honest"),
    ("leaky", "_leaky", "both arms' MRR -- the original bug"),
]


def env_path(dataset, suffix):
    return os.path.join(ROUTER_DIR, f"{dataset}_env_data{suffix}.json")


def load_arm(suffix, metric="mrr"):
    """Pooled (X, R) across datasets for one arm, stacked in DATASETS order.

    Returns None when any dataset is missing, rather than silently pooling a subset -- an
    arm computed over two datasets is not comparable to one computed over three.
    """
    Xs, Rs = [], []
    for ds in DATASETS:
        p = env_path(ds, suffix)
        if not os.path.exists(p):
            return None
        q = load_queries(p)
        qids = list(q)
        Xs.append(np.array([q[i]["features"] for i in qids], dtype=float))
        Rs.append(np.array([[float(q[i]["retrievers"][a][metric]) for a in ACTIONS]
                            for i in qids]))
    widths = {x.shape[1] for x in Xs}
    if len(widths) > 1:
        print(f"  suffix {suffix or '(none)'!r}: feature widths differ {sorted(widths)} - "
              f"cannot pool, skipped.")
        return None
    return np.vstack(Xs), np.vstack(Rs)


def evaluate(X, R, seed=0, n_jobs=-1):
    """Accuracy predicting the strictly-best retriever. Same protocol as
    learnability_analysis.predictability(), minus the per-group breakdown."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    mask = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    Xi, yi = X[mask], R[mask].argmax(axis=1)
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
        "n_informative": int(len(yi)), "majority": majority,
        "gbt": float(gbt.mean()), "gbt_std": float(gbt.std()),
        "logreg": float(lr.mean()), "logreg_std": float(lr.std()),
        "gbt_lift": float(gbt.mean()) - majority,
        "logreg_lift": float(lr.mean()) - majority,
    }


def check_rewards_identical(arms):
    """Arms must share reward tables -- they are meant to differ ONLY in features 22-25.

    If they differ, the arms came from different retrieval runs and the whole comparison is
    void. Reported as a hard failure rather than a warning: a dose-response curve over
    inconsistent rewards would look perfectly plausible and mean nothing.
    """
    ref_name, ref_R = arms[0][0], arms[0][2]
    problems = []
    for name, _, R in arms[1:]:
        if R.shape != ref_R.shape:
            problems.append(f"{name}: reward shape {R.shape} != {ref_name} {ref_R.shape}")
        elif not np.array_equal(R, ref_R):
            n = int((R != ref_R).any(axis=1).sum())
            problems.append(f"{name}: reward table differs from {ref_name} on {n} queries")
    return problems


def verify(rows, reference_path, tol=1e-9):
    """The honest and leaky arms must reproduce learnability_analysis.json.

    Without this the curve could drift from the number the paper quotes and nobody would
    notice -- the two scripts would simply disagree.
    """
    if not os.path.exists(reference_path):
        print(f"\n[verify] reference not found at {reference_path} - skipped.")
        return None
    with open(reference_path, "r", encoding="utf-8") as f:
        pooled = (json.load(f).get("pooled") or {})
    checks, ok = [], True
    for arm, ref_key in (("honest_minimal", "honest"), ("leaky", "leaky")):
        ref, got = pooled.get(ref_key), rows.get(arm)
        if not ref or not got:
            continue
        for field in ("n_informative", "majority", "gbt"):
            a, b = got[field], ref[field]
            match = (a == b) if isinstance(b, int) else abs(a - b) <= tol
            ok &= match
            checks.append((f"{arm}.{field}", a, b, match))
    if not checks:
        print("\n[verify] no comparable arms in the reference - skipped.")
        return None
    print("\n[verify] vs results/learnability_analysis.json")
    for name, a, b, match in checks:
        print(f"  {'OK  ' if match else 'FAIL'}  {name}: got {a}  want {b}")
    return ok


def render(rows, verified, tag):
    order = [a for a, _, _ in ARMS if a in rows]
    honest = rows.get("honest_minimal")
    L = ["# Leakage Dose-Response: how much contamination does it take?", "",
         f"_Scale: {tag}. Accuracy predicting the strictly-best retriever, "
         f"{N_SPLITS}-fold stratified CV x {N_REPEATS} repeats, pooled over "
         f"{', '.join(DATASETS)}._", "",
         "Every arm differs from the others **only in features 22-25**. Reward tables are "
         "identical across arms (asserted, not assumed), so the accuracy difference is "
         "attributable to feature construction and nothing else.", "",
         "| arm | what leaks | accuracy | lift vs majority | vs honest |",
         "|---|---|---:|---:|---:|"]
    desc = {a: d for a, _, d in ARMS}
    for arm in order:
        r = rows[arm]
        delta = "" if not honest else f"{(r['gbt'] - honest['gbt'])*100:+.1f} pts"
        if arm == "honest_minimal":
            delta = "_baseline_"
        L.append(f"| `{arm}` | {desc[arm]} | {r['gbt']*100:.1f}% ± {r['gbt_std']*100:.1f} | "
                 f"**{r['gbt_lift']*100:+.1f}%** | {delta} |")
    if honest:
        L += ["", f"Majority-class baseline: **{honest['majority']*100:.1f}%** "
                  f"(n = {honest['n_informative']} informative queries).", ""]

    noisy = rows.get("leak_noisy")
    if noisy and honest:
        gain = (noisy["gbt"] - honest["gbt"]) * 100
        L += ["## What the noisy arm means", ""]
        if gain > 1.0:
            L += [f"`leak_noisy` buys **{gain:+.1f} points** over honest features while "
                  f"matching no reward value exactly -- jitter breaks equality, so a check "
                  f"of the form \"no feature equals any reward\" does not see it. Checking "
                  f"for exact equality is therefore **not a sufficient defence**: a leak "
                  f"can be an imprecise proxy and still inflate results, which is what an "
                  f"accidental leak usually is.", ""]
        else:
            L += [f"`leak_noisy` buys only **{gain:+.1f} points** over honest features, so "
                  f"at this jitter scale the leak does not survive being an imprecise "
                  f"proxy. Exact-equality detection would catch most of what matters here.",
                  ""]
    if verified is not None:
        L += [f"Honest and leaky arms reproduce `results/learnability_analysis.json`: "
              f"**{'yes' if verified else 'NO -- methodology drift'}**.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="mrr")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--tag", default="full run, real_wikipedia gold")
    ap.add_argument("--verify-against",
                    default=os.path.join(RESULTS_DIR, "learnability_analysis.json"))
    args = ap.parse_args()

    loaded = []
    for arm, suffix, _ in ARMS:
        data = load_arm(suffix, args.metric)
        if data is None:
            print(f"  {arm}: env data missing or unpoolable - skipped.")
            continue
        X, R = data
        loaded.append((arm, X, R))
        print(f"  {arm}: {X.shape[0]} queries, {X.shape[1]} features")

    if len(loaded) < 2:
        raise SystemExit("Need at least two arms to compare.")

    problems = check_rewards_identical(loaded)
    if problems:
        print("\nINTEGRITY FAILURE - arms are not a clean ablation:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(1)
    print(f"\n  reward tables identical across all {len(loaded)} arms - clean ablation\n")

    rows = {}
    for arm, X, R in loaded:
        r = evaluate(X, R, n_jobs=args.n_jobs)
        if r is None:
            print(f"  {arm}: too few informative queries to cross-validate - skipped.")
            continue
        rows[arm] = r
        print(f"  {arm:16s} acc={r['gbt']*100:5.1f}%  lift={r['gbt_lift']*100:+5.1f}%  "
              f"(majority {r['majority']*100:.1f}%)")

    verified = verify(rows, args.verify_against)

    os.makedirs(args.out_dir, exist_ok=True)
    payload = {"tag": args.tag, "metric": args.metric, "n_splits": N_SPLITS,
               "n_repeats": N_REPEATS, "verified_against_learnability": verified,
               "arms": rows}
    with open(os.path.join(args.out_dir, "leak_gradient.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(args.out_dir, "leak_gradient.md"), "w", encoding="utf-8") as f:
        f.write(render(rows, verified, args.tag))
    print(f"\nSaved: {os.path.join(args.out_dir, 'leak_gradient.json')}")
    print(f"Saved: {os.path.join(args.out_dir, 'leak_gradient.md')}")

    if verified is False:
        raise SystemExit("Verification FAILED: arms do not reproduce "
                         "learnability_analysis.json.")


if __name__ == "__main__":
    main()
