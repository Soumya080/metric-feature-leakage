"""Reproduce the leak dose-response on RouterBench: is the mechanism domain-general?

WHY THIS EXISTS
    Every number in the dose-response comes from one codebase -- ours. The paper argues the
    hazard is general, but four of its five generality arguments are arguments rather than
    measurements: the honest replacement is inert, the two kinds of number sit side by side,
    Kapoor & Narayanan decline to subdivide [L2], and no benchmark ships an engineered feature
    matrix. Only the jitter finding is measured, and it is measured on our data.

    Tests whether the effect is specific to our pipeline: take a routing dataset built by
    others, in a different domain (LLM routing, not retriever routing), inject the same five
    rungs of contamination, and check whether the curve reproduces, in particular whether the
    jittered rung still tracks the exact one.

WHAT WOULD FALSIFY THE PAPER'S CLAIM
    If exact leakage lifts accuracy but jittered leakage does not, the jitter finding is specific
    to our reward distribution. The script reports either outcome; a flat curve would bound the
    claim's generality, not overturn the result on our data.

WHY THE ARM COUNT DIFFERS, AND WHY WE REPORT BOTH
    Our leak covered 2 of 3 arms. RouterBench has 11. Leaking 2 of 11 is a much smaller dose in
    proportional terms, so a weaker curve there would be expected and would prove nothing. We
    therefore sweep k, the number of arms whose reward is exposed, and report the whole sweep.
    The claim under test is not "the lift is the same size" -- the settings are not comparable
    that way -- but "jitter costs little relative to the exact leak", which is a ratio and
    survives the difference in dose.

DATA
    RouterBench ships pickles, and unpickling executes arbitrary code, so this script downloads
    nothing. Fetch and vet routerbench_0shot.pkl yourself, then point --data at it:
        https://huggingface.co/datasets/withmartian/routerbench

USAGE
    python experiments/router_experiments/leak_injection_external.py --data /path/to/rb.pkl
    python experiments/router_experiments/leak_injection_external.py --data ... --smoke
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.router_experiments.audit_external import MODELS, prompt_features  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

N_SPLITS = 5
N_REPEATS = 5
JITTER_FRAC = 0.15   # same 15% magnitude as the RL router leak_noisy arm -- NOT the same construction


def jitter(values, frac=JITTER_FRAC, seed=0):
    """Deterministic multiplicative Gaussian noise, seeded.

    CORRECTED 2026-10-31: this is NOT the RL router leak_noisy construction, as this docstring
    used to say. RL router adds a value-keyed crc32 offset (prepare_env_data._jitter), so equal
    values get equal offsets. This multiplies by (1 + 0.15 N(0, 1)) drawn independently per
    cell, so every zero reward stays exactly 0. On RouterBench's mostly-binary correctness
    rewards that keeps 0 and 1 almost perfectly separable, which is why retention here is
    92-98% largely by construction. The paper says so (Section 4, "The mechanism transplants").

    Deterministic because the point of the arm is that it is reproducible: the claim is that a
    leak surviving THIS perturbation evades an equality check, and a reader must be able to
    regenerate the exact perturbed values to confirm no column equals any reward.
    """
    rng = np.random.default_rng(seed)
    return values * (1.0 + frac * rng.standard_normal(values.shape))


def build_arms(X_honest, R, k, seed=0):
    """The five rungs, differing ONLY in what is appended to an identical honest block.

    Returns {arm_name: X}. R itself is never modified, so the label is identical across arms by
    construction rather than by assertion -- the external analogue of the byte-identical reward
    table check that guards the RL router dose-response.
    """
    # Expose the k arms with the highest mean reward. Picking the strongest arms is the
    # pessimistic choice for us: those are the ones a router would most want to identify, so if
    # the leak is going to look impressive anywhere it is here, and we are not flattering the
    # result by choosing arms that happen to separate well.
    order = np.argsort(-R.mean(axis=0))[:k]
    exact = R[:, order]

    # Hit analogue: the sign of the reward, not its value. RouterBench performance is already
    # roughly in [0,1], so a median split is the closest thing to Hit@k that does not assume a
    # particular scoring scale.
    med = np.median(R[:, order], axis=0, keepdims=True)
    hit = (R[:, order] >= med).astype(float)

    return {
        "honest":     X_honest,
        "leak_one":   np.hstack([X_honest, R[:, order[:1]]]),
        "leak_hit":   np.hstack([X_honest, hit]),
        "leak_noisy": np.hstack([X_honest, jitter(exact, seed=seed)]),
        "leaky":      np.hstack([X_honest, exact]),
    }


def evaluate(X, y, seed=0, n_jobs=-1):
    """Same protocol as leak_gradient.evaluate: GBT, repeated stratified CV, accuracy."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    counts = np.bincount(y)
    if len(y) < N_SPLITS * 2 or (counts[counts > 0] < N_SPLITS).any():
        return None
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    s = cross_val_score(GradientBoostingClassifier(random_state=seed), X, y,
                        cv=cv, scoring="accuracy", n_jobs=n_jobs)
    return {"acc": float(s.mean()), "std": float(s.std()),
            "majority": float(counts.max() / len(y))}


def assert_no_exact_leak(arms, R, tol=0.0):
    """The jittered arm must contain no column exactly equal to any reward column.

    This is the whole point of that arm. If jitter happened to leave a column untouched the
    result would be an exact leak wearing a noisy label, and the headline comparison would be
    vacuous. Checked rather than assumed.
    """
    Xn = arms["leak_noisy"]
    for j in range(Xn.shape[1]):
        for a in range(R.shape[1]):
            if np.allclose(Xn[:, j], R[:, a], atol=tol, rtol=0):
                raise SystemExit(
                    f"jittered arm column {j} is exactly reward column {a}: the noisy arm is "
                    f"not noisy, so no conclusion can be drawn from it")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True,
                   help="routerbench_0shot.pkl, fetched and vetted by you. Nothing is downloaded.")
    p.add_argument("--k-arms", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="how many arms' rewards to expose; swept because dose is not comparable "
                        "across settings with different arm counts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="3,000 rows, k=2 only, for a fast check")
    p.add_argument("--outdir", default=RESULTS_DIR)
    args = p.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required to read RouterBench's pickle format.")
    if not os.path.exists(args.data):
        sys.exit(f"not found: {args.data}")

    df = pd.read_pickle(args.data)
    if args.smoke:
        df = df.sample(n=min(3000, len(df)), random_state=0)
        args.k_arms = [2]

    missing = [m for m in MODELS if m not in df.columns]
    if missing:
        sys.exit(f"expected model columns absent: {missing}")

    R = df[MODELS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    keep = ~np.isnan(R).any(axis=1)
    R, df = R[keep], df[keep]

    # Only strictly-best queries carry a label. RouterBench ties on ~93% of queries, so this
    # discards most of the data -- which is itself the structural finding from the audit, and
    # the reason the informative count is reported rather than glossed.
    strict = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    if strict.sum() < N_SPLITS * 4:
        sys.exit(f"only {int(strict.sum())} strictly-best queries: too few to fit anything. "
                 f"The reward table is too tied for this experiment to say anything.")

    prompts = df["prompt"] if "prompt" in df.columns else df.iloc[:, 0]
    X_all, _ = prompt_features(prompts)
    X_all, R_all = X_all[strict], R[strict]
    y = R_all.argmax(axis=1)
    print(f"{len(R):,} queries loaded; {len(y):,} strictly-best ({strict.mean():.1%}) usable")

    out = {"n_total": int(len(R)), "n_informative": int(len(y)),
           "tie_rate": float(1 - strict.mean()), "n_arms": int(R.shape[1]),
           "jitter_frac": JITTER_FRAC, "seed": args.seed, "by_k": {}}

    for k in args.k_arms:
        if k > R.shape[1]:
            continue
        arms = build_arms(X_all, R_all, k, seed=args.seed)
        assert_no_exact_leak(arms, R_all)
        res = {}
        for name, X in arms.items():
            r = evaluate(X, y, seed=args.seed)
            if r is None:
                sys.exit("too few samples per class for the CV protocol")
            res[name] = r
            print(f"  k={k:<2} {name:<11} {r['acc']*100:6.2f}%  "
                  f"(majority {r['majority']*100:.2f}%)")
        base = res["honest"]["acc"]
        res["_premium_exact"] = (res["leaky"]["acc"] - base) * 100
        res["_premium_noisy"] = (res["leak_noisy"]["acc"] - base) * 100
        res["_jitter_cost"] = res["_premium_exact"] - res["_premium_noisy"]
        res["_jitter_retention"] = (res["_premium_noisy"] / res["_premium_exact"]
                                    if res["_premium_exact"] > 1e-9 else None)
        out["by_k"][str(k)] = res
        ret = res["_jitter_retention"]
        print(f"  k={k:<2} exact +{res['_premium_exact']:.2f}pts, "
              f"jittered +{res['_premium_noisy']:.2f}pts, "
              f"retention {'n/a' if ret is None else f'{ret:.1%}'}\n")

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "leak_injection_routerbench.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")

    rets = [v["_jitter_retention"] for v in out["by_k"].values()
            if v.get("_jitter_retention") is not None]
    if not rets:
        print("\nNo exact-leak premium to speak of: the mechanism does not reproduce here, and "
              "the paper should say the finding is specific to dense reward tables.")
    else:
        print(f"\nJitter retention across k: {min(rets):.1%}-{max(rets):.1%} "
              f"(RL router: 97%). Retention near 1 means the mechanism is domain-general; "
              f"retention near 0 means it is not, and that bounds the paper's claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
