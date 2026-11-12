"""Does the dose-response survive a change of random seed?

WHY THIS EXISTS
    The paper carries "one split seed" in its Limitations, which is accurate and also unsatisfying:
    it tells a reader the risk exists without telling them how large it is. Every reported spread
    is training-seed variance at a single cross-validation seed, so nothing published so far
    distinguishes "the leak is worth 12.2 points" from "the leak is worth 12.2 points at seed 0".

    This resamples the whole dose-response across several CV seeds. It changes no claim if the
    ordering holds; if the ordering does NOT hold -- if honest ever beats leak_one, or jittered
    ever collapses toward honest -- that is a finding the paper must report, and better found here
    than by a reviewer.

WHAT IT DOES NOT FIX
    This varies the cross-validation partition, not the upstream train/test split used to build
    the candidate sets, which is fixed at 42 in the data-generation stage. So it bounds one source
    of variance, not all of them, and Limitations should continue to say so.

USAGE
    python experiments/router_experiments/seed_stability.py                 # seeds 0-4
    python experiments/router_experiments/seed_stability.py --seeds 0 1 2   # faster
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.router_experiments.leak_gradient import (  # noqa: E402
    ARMS, load_arm, evaluate)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--outdir", default=RESULTS_DIR)
    args = p.parse_args()

    loaded = {}
    for arm, suffix, _ in ARMS:
        try:
            loaded[arm] = load_arm(suffix)
        except (FileNotFoundError, OSError) as e:
            print(f"skipping {arm}: {e}")
    if len(loaded) < 2:
        sys.exit("need at least two arms present to say anything about stability")

    out = {"seeds": args.seeds, "per_arm": {}}
    print(f"{'arm':<12}" + "".join(f"{s:>9}" for s in args.seeds) + f"{'mean':>9}{'spread':>9}")
    print("-" * (12 + 9 * (len(args.seeds) + 2)))

    for arm in loaded:
        X, R = loaded[arm]
        accs = []
        for s in args.seeds:
            r = evaluate(X, R, seed=s)
            accs.append(None if r is None else r["gbt"] * 100)
        good = [a for a in accs if a is not None]
        out["per_arm"][arm] = {"acc_by_seed": accs,
                               "mean": float(np.mean(good)) if good else None,
                               "spread": float(max(good) - min(good)) if good else None}
        cells = "".join(f"{a:>9.2f}" if a is not None else f"{'--':>9}" for a in accs)
        print(f"{arm:<12}{cells}{np.mean(good):>9.2f}{max(good)-min(good):>9.2f}")

    # The claim under test is the ORDERING, not the point estimates. A dose-response whose rungs
    # swap under reseeding is not a dose-response.
    #
    # This must be DOSE order, which is not the order ARMS happens to be declared in --
    # leak_gradient lists honest, noisy, hit, one_arm, leaky, whereas the dose runs
    # honest < one_arm < hit < noisy < leaky. An earlier version of this script iterated ARMS
    # directly and reported ten "inversions" that were the true ordering seen through the wrong
    # comparator. Stated explicitly here so it cannot drift again.
    DOSE_ORDER = ["honest_minimal", "leak_one_arm", "leak_hit", "leak_noisy", "leaky"]
    unknown = set(out["per_arm"]) - set(DOSE_ORDER)
    if unknown:
        sys.exit(f"arm(s) {sorted(unknown)} have no place in DOSE_ORDER; refusing to guess "
                 f"where they belong on the dose axis")
    order = [a for a in DOSE_ORDER if a in out["per_arm"]]
    inversions = []
    for i in range(len(order) - 1):
        lo, hi = order[i], order[i + 1]
        for j, s in enumerate(args.seeds):
            a, b = out["per_arm"][lo]["acc_by_seed"][j], out["per_arm"][hi]["acc_by_seed"][j]
            if a is not None and b is not None and a > b:
                inversions.append({"seed": s, "expected_lower": lo, "expected_higher": hi,
                                   "values": [a, b]})
    out["ordering_inversions"] = inversions

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "seed_stability.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print()
    if inversions:
        print(f"{len(inversions)} ordering inversion(s) across seeds -- the dose-response is NOT "
              f"stable and the paper must report this:")
        for inv in inversions[:6]:
            print(f"   seed {inv['seed']}: {inv['expected_lower']} "
                  f"({inv['values'][0]:.2f}) > {inv['expected_higher']} ({inv['values'][1]:.2f})")
    else:
        print("No ordering inversions: every rung stays above the one below it at every seed, "
              "so the dose-response is not an artifact of the seed.")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
