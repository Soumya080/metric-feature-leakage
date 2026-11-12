#!/usr/bin/env python3
"""R3: what an equality check and leakcheck report on every R1 arm, at the 61-D scale.

For each arm (Table 2 anchors + crc_v2 noise variants), per dataset and pooled:
  X = the full 61-D feature matrix as the router sees it
  R = per-arm full-ranking reward (retrievers[arm].rank_reward)

Records
  - leakcheck.audit(X, R): verdicts vs suspicions (verdicts only from exact agreement),
    and any finding outside slots 22-25 (a false alarm on honest features);
  - per (slot, arm) raw statistics from leakcheck.detectors for slots 22-24: exact-match
    rate on non-zero-reward rows, |Spearman| on non-zero rows, and single-feature argmax
    recovery lift. The argmax lift is recorded so it can later be compared with R1's
    accuracy inflation; no claim about that relationship is made here.

Model-free and fast (minutes). No training, no CV.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import common as C
import leakcheck
from leakcheck import detectors as D
from r1_classifier_sweep import arm_paths

SLOT_ARM = {22: (0,), 23: (1,), 24: (0, 1)}


def audit_one(X: np.ndarray, R: np.ndarray) -> dict:
    rep = leakcheck.audit(X, R, arm_names=list(C.ACTIONS))
    fnd = [{"feature": f.feature, "arm": C.ACTIONS[f.arm], "rung": f.rung, "statistic": float(f.statistic),
            "verdict": bool(f.is_verdict), "selectivity": None if np.isnan(f.selectivity) else float(f.selectivity)}
           for f in rep.findings]
    slots = []
    for s, arm_ids in SLOT_ARM.items():
        for a in arm_ids:
            rate, n_agree, n_elig, n_dist = D.exact_match(X[:, s], R[:, a])
            rho, n_rho = D.rank_agreement(X[:, s], R[:, a])
            slots.append({"slot": s, "arm": C.ACTIONS[a], "exact_match_rate": float(rate), "n_agree": n_agree,
                          "n_eligible_nonzero": n_elig, "n_distinct_agreeing": n_dist, "abs_spearman": float(rho)})
    argmax = []
    for s in (22, 23, 24):
        acc, lift, n = D.argmax_recovery(X[:, s], R)
        argmax.append({"slot": s, "argmax_recovery_acc": float(acc), "lift_vs_majority": float(lift), "n_strict": n})
    return {"leaked_verdict": bool(rep.leaked), "n_verdicts": len(rep.verdicts), "n_suspicions": len(rep.suspicions),
            "findings": fnd, "findings_outside_slots_22_25": [f for f in fnd if f["feature"] not in C.LEAK_SLOTS],
            "slot_stats": slots, "argmax_recovery": argmax}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-dir", default=C.CRC_ENV_DIR)
    ap.add_argument("--out-dir", default=os.path.join(C.CRC_RESULTS, "r3_leakcheck_sweep"))
    args = ap.parse_args(argv)

    arms = arm_paths(args.env_dir)
    missing = sorted({a for a, ps in arms.items() for p in ps.values() if not os.path.exists(p)})
    if missing:
        raise SystemExit(f"env files missing for {missing}; run r1_build_noise_variants.py first")
    out_dir = C.prepare_out_dir(args.out_dir)

    result = {"header": C.run_header(args), "leakcheck_version": getattr(leakcheck, "__version__", "unknown"),
              "settings": {"EXACT_MATCH_THRESHOLD": D.EXACT_MATCH_THRESHOLD, "MIN_DISTINCT": D.MIN_DISTINCT,
                           "EQ_TOL": D.EQ_TOL, "RANK_CORRELATION_SUSPECT": D.RANK_CORRELATION_SUSPECT},
              "reward": "retrievers[arm].rank_reward", "arms": {}}
    for arm, paths in arms.items():
        per = {}
        for ds in C.DATASETS:
            q = C.load_env(paths[ds])
            qids = list(q)
            per[ds] = audit_one(C.feature_matrix(q, qids), C.reward_matrix(q, qids))
        X, R = C.pooled(paths, metric="rank_reward")
        per["pooled"] = audit_one(X, R)
        result["arms"][arm] = per
        p = per["pooled"]
        print(f"  {arm:24s} verdict={p['leaked_verdict']!s:5s} verdicts={p['n_verdicts']} "
              f"suspicions={p['n_suspicions']} outside_22_25={len(p['findings_outside_slots_22_25'])}", flush=True)
    C.write_json(os.path.join(out_dir, "leakcheck_sweep.json"), result)
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
