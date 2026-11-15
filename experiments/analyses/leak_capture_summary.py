#!/usr/bin/env python3
"""Table 2's leak rungs expressed as apparent capture of the oracle gap.

Table 2 reports accuracy at predicting the strictly-best arm. Routing papers report reward, here
capture of the oracle gap. oracle_gap_capture.py already trained the canonical tuned gradient-boosting router
on every rung with Table 3's protocol (70/15/15 split, split seeds 42, 7, 13, 99, 2024) and
scored it on the test split as capture = (method - best fixed) / (oracle - best fixed), best
fixed chosen in hindsight on the test split exactly as in Table 3, ties included.
This script only aggregates those committed rows; it fits nothing.

Reads results/analyses/oracle_gap_capture{,_o1}/capture_rows.csv. Writes --out-dir (refuses non-empty).
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st

import common as C

RUNGS = ("honest_minimal", "leak_one_arm", "leak_hit", "leak_noisy", "leaky")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = C.ANALYSIS_RESULTS
    ap.add_argument("--inputs", nargs="+", default=[os.path.join(base, "oracle_gap_capture", "capture_rows.csv"),
                                                     os.path.join(base, "oracle_gap_capture_o1", "capture_rows.csv")])
    ap.add_argument("--method", default="gradient_boosting")
    ap.add_argument("--out-dir", default=os.path.join(C.ANALYSIS_RESULTS, "leak_capture_summary"))
    args = ap.parse_args(argv)
    out_dir = C.prepare_out_dir(args.out_dir)
    rows = [r for p in args.inputs for r in csv.DictReader(open(p, encoding="utf-8"))]
    out = {"header": {**C.run_header(args), "input_sha256": {p: C.sha256_file(p) for p in args.inputs}},
           "method": args.method, "rungs": {}}
    for arm in RUNGS:
        rs = [r for r in rows if r["arm"] == arm and r["method"] == args.method]
        per_ds = {}
        for ds in C.DATASETS:
            v = [float(r["capture_vs_test_best"]) for r in rs if r["dataset"] == ds]
            per_ds[ds] = {"n_splits": len(v), "mean": st.mean(v), "min": min(v), "max": max(v)}
        allv = [float(r["capture_vs_test_best"]) for r in rs]
        seeds = sorted({r["split_seed"] for r in rs})
        out["rungs"][arm] = {"split_seeds": seeds, "n": len(allv), "mean": st.mean(allv),
                             "min": min(allv), "max": max(allv), "per_dataset": per_ds}
        print(f"{arm:15s} mean capture {100*st.mean(allv):+6.1f}%  range [{100*min(allv):+6.1f}, {100*max(allv):+6.1f}]  "
              + "  ".join(f"{ds[:3]} {100*d['mean']:+5.1f}" for ds, d in per_ds.items()) + f"  (n={len(allv)})")
    C.write_json(os.path.join(out_dir, "summary.json"), out)
    print("Done:", out_dir)


if __name__ == "__main__":
    main()
