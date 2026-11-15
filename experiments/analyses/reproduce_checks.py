#!/usr/bin/env python3
"""Check that the paper's leak-gradient numbers reproduce from the committed artifacts.

Three checks, all read-only
with respect to existing results (everything is written under results/analyses/reproduce_checks/):

  1. Re-run leak_gradient.py (Table 2's script) into a NEW out-dir and compare every
     arm's accuracy, majority and n against the committed results/leak_gradient.json.
     leak_gradient also self-verifies honest/leaky against learnability_analysis.json.
  2. Hash every env file Table 2 / Table 3 read (src/router/*_env_data*.json) and record
     the hashes, so later experiments can prove they read the same inputs.
  3. Run paper/audit_numbers.py --unchecked and save its list of numbers in main.tex
     that have not been traced to an artifact (a to-do list, not a pass/fail).

Exit code is non-zero if check 1 finds any mismatch.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import common as C

TOL = 1e-9


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(C.ANALYSIS_RESULTS, "reproduce_checks"))
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args(argv)
    out_dir = C.prepare_out_dir(args.out_dir)
    report = {"header": C.run_header(args)}

    # 1. Table 2 recompute
    lg_out = os.path.join(out_dir, "leak_gradient")
    cmd = [sys.executable, os.path.join(C.ROUTER_EXP_DIR, "leak_gradient.py"),
           "--out-dir", lg_out, "--n-jobs", str(args.n_jobs)]
    with open(os.path.join(out_dir, "leak_gradient.log"), "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, cwd=C.PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    report["leak_gradient_exit_code"] = rc
    mismatches = []
    new_p, old_p = os.path.join(lg_out, "leak_gradient.json"), os.path.join(C.RESULTS_DIR, "leak_gradient.json")
    if rc == 0 and os.path.exists(new_p):
        new, old = json.load(open(new_p, encoding="utf-8"))["arms"], json.load(open(old_p, encoding="utf-8"))["arms"]
        for arm in sorted(set(new) | set(old)):
            for k in ("n_informative", "majority", "gbt", "logreg"):
                a, b = (new.get(arm) or {}).get(k), (old.get(arm) or {}).get(k)
                if a is None or b is None or abs(float(a) - float(b)) > TOL:
                    mismatches.append({"arm": arm, "field": k, "recomputed": a, "committed": b})
    else:
        mismatches.append({"error": f"leak_gradient.py exited {rc}; see leak_gradient.log"})
    report["table2_mismatches"] = mismatches

    # 2. Input hashes
    report["env_sha256"] = {os.path.basename(p): C.sha256_file(p)
                            for p in sorted(glob.glob(os.path.join(C.ROUTER_DIR, "*_env_data*.json")))
                            if not p.endswith("_layout.json")}

    # 3. Unchecked numbers in the manuscript
    with open(os.path.join(out_dir, "audit_numbers_unchecked.txt"), "w", encoding="utf-8") as f:
        report["audit_numbers_exit_code"] = subprocess.call(
            [sys.executable, os.path.join(C.PROJECT_ROOT, "paper", "audit_numbers.py"), "--unchecked"],
            cwd=C.PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT)

    C.write_json(os.path.join(out_dir, "r0_report.json"), report)
    print(f"Table 2 recompute: {'OK' if not mismatches else f'{len(mismatches)} MISMATCH(ES)'}; report in {out_dir}")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
