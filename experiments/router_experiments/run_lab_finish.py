"""
run_lab_finish.py — The finishing run. Everything that is stale, and nothing that is not.

This is NOT run_lab_full.py. That script rebuilds the world: corpora, retrieval, env data,
then every analysis. It is the right tool when the data changes. Nothing about the data has
changed here, so re-running it would spend a day recomputing artifacts that are already
correct.

What actually needs the lab machine is two things:

  1. canonical_results.json is STALE, because compute_heuristic_baseline in
     canonical_evaluation.py was fixed. The Score-Gap heuristic used to sweep thresholds over
     np.linspace(-5, 5, 100) while the feature it thresholds lives in [-42.8, -2.96], so
     0-0.7% of values fell inside the swept interval and the "heuristic" silently degenerated
     into a fixed-arm baseline -- bit-identical to Always-BM25 on HotpotQA. It now sweeps the
     training split's own quantiles. Verified in isolation, the fix moves the heuristic from
     0.1079 to 0.1786 on NQ, 0.1477 to 0.1729 on TriviaQA, 0.0477 to 0.0745 on HotpotQA. The
     committed table still carries the old numbers until this stage re-runs, and everything
     downstream of it -- the LaTeX tables, headroom capture, the figures -- inherits them.

  2. The permutation control has only ever run on NQ, because joblib's worker pool cannot hold
     more than one dataset in memory on the dev laptop. NQ gave observed 61.28% against a
     shuffled-label null of 60.75% (p = 0.238) and a random-feature null of 60.92% (p = 0.286).
     That is the paper's strongest single sentence and it currently rests on one of three
     datasets.

Everything else here is cheap and runs only because it reads one of those two.

ORDER MATTERS. Stage 2 reads canonical_results.json, so it cannot precede stage 1. This has
been got wrong before in this repo.

Usage:
    python experiments/router_experiments/run_lab_finish.py
    python experiments/router_experiments/run_lab_finish.py --n-perm 20 --n-jobs -1
    python experiments/router_experiments/run_lab_finish.py --skip-canonical   # nulls only
    python experiments/router_experiments/run_lab_finish.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PROJECT_ROOT, "results")
DATASETS = ("nq", "triviaqa", "hotpotqa")

# Every child runs under sys.executable -- the interpreter running THIS file -- so there is no
# python/python3 ambiguity and no way to execute a stage in a different environment than the
# one the dependencies were installed into. Same reasoning as run_lab_full.py.
PY = sys.executable


def banner(txt):
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78, flush=True)


def run(label, args, required, dry_run, log):
    """Run one stage. `required` stages abort the run on failure; additive ones only warn."""
    cmd = [PY, "-u"] + args
    print(f"\n--- {label} ---")
    print("    " + " ".join(cmd[1:]), flush=True)
    if dry_run:
        log.append((label, "dry-run", 0.0))
        return True
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=PROJECT_ROOT)
    el = time.time() - t0
    if rc == 0:
        print(f"    OK  ({el/60:.1f} min)", flush=True)
        log.append((label, "ok", el))
        return True
    status = "FAILED (required)" if required else "failed (additive, continuing)"
    print(f"    {status}  rc={rc}  after {el/60:.1f} min", flush=True)
    log.append((label, "FAILED" if required else "failed", el))
    if required:
        print("\nAborting: later stages read this stage's output and would silently reuse "
              "the stale artifact.", flush=True)
        sys.exit(rc)
    return False


def verify_score_gap_fix():
    """Confirm the regenerated table actually reflects the fix, before anything reads it.

    The failure this catches is specific and has a precedent in this repo: a stage appears to
    succeed, but the resumable pipeline skipped it because its output already existed, so the
    "new" results are the old ones. Checking the number rather than the exit code is the only
    way to know.
    """
    path = os.path.join(RESULTS, "canonical_results.json")
    if not os.path.exists(path):
        print(f"  cannot verify: {path} absent")
        return False
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    # Recompute the heuristic live and compare, rather than checking against hard-coded
    # expected values. An earlier version of this function pinned the post-fix numbers
    # (0.178592 / 0.172932 / 0.074464) measured against one snapshot of the env data. That is
    # brittle for the obvious reason: a lab run that regenerates the env data shifts every
    # number slightly, and the check would then abort a perfectly good run for disagreeing
    # with a stale constant. Recomputing tests the property that matters -- does the committed
    # table agree with what the current code produces on the current data -- and stays correct
    # across data regenerations.
    #
    # compute_heuristic_baseline is deterministic given the split (split_seed=42, no
    # sampling), so agreement should be exact, not approximate.
    TOL = 1e-9
    try:
        sys.path.insert(0, HERE)
        sys.path.insert(0, PROJECT_ROOT)
        from canonical_evaluation import compute_heuristic_baseline
        from src.router.routing_env import create_train_val_test_split
    except ImportError as e:
        print(f"  cannot verify (import failed): {e}")
        return False

    ok = True
    for ds in DATASETS:
        if ds not in d:
            continue
        row = d[ds]
        sg = row.get("score_gap")
        if sg is None:
            print(f"  {ds:9s} score_gap absent from the results file")
            ok = False
            continue
        env_path = os.path.join(PROJECT_ROOT, "src", "router", f"{ds}_env_data.json")
        if not os.path.exists(env_path):
            print(f"  {ds:9s} score_gap={sg:.6f}  (env data absent, cannot recompute)")
            continue
        env_train, _, env_test = create_train_val_test_split(env_path, split_seed=42)
        live, _ = compute_heuristic_baseline(env_train, env_test, "mrr")

        # Equality with a fixed arm is not automatically wrong. If thresholding genuinely
        # cannot beat committing to one retriever, the sweep is entitled to say so -- the
        # endpoints are in the candidate set precisely so it can express that. What is wrong
        # is the table disagreeing with the code that supposedly produced it.
        dup = [k for k in ("bm25", "dense", "hybrid") if row.get(k) == sg]
        note = f"  (equals {dup[0]})" if dup else ""
        if abs(sg - live) < TOL:
            print(f"  {ds:9s} score_gap={sg:.6f}{note}  OK, matches a live recomputation")
        else:
            print(f"  {ds:9s} score_gap={sg:.6f}{note}  <-- STALE: the current code produces "
                  f"{live:.6f} on this env data, a difference of {abs(sg - live):.6f}")
            ok = False

    if ok:
        print("  verified: the table agrees with the current quantile sweep")
    else:
        print("  PROBLEM: the table disagrees with what the code now produces. "
              "canonical_evaluation.py overwrites its output unconditionally, so re-run "
              "stage 1 (drop --skip-canonical).")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-perm", type=int, default=20,
                   help="Permutation replicates per null. 20 gives a p-value floor of 0.048, "
                        "which is ample -- NQ came in at p=0.238, nowhere near the floor. "
                        "Each extra replicate costs two more CV fits per dataset.")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Fold parallelism. -1 uses all cores and is right on the lab machine. "
                        "Drop to 4 if joblib's worker pool dies (see permutation_control).")
    p.add_argument("--skip-canonical", action="store_true",
                   help="Skip the 90-minute retrain. Only valid if canonical_results.json was "
                        "already regenerated since the score_gap fix.")
    p.add_argument("--skip-nulls", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    # Defaults copied from the existing 100K artifact's provenance block so the regenerated
    # file is comparable to it rather than being a differently-labelled experiment.
    p.add_argument("--gold-type", default="real_wikipedia")
    p.add_argument("--pool-size", type=int, default=100000)
    p.add_argument("--run-tag", default="pool100k")
    args = p.parse_args()

    log = []
    t_start = time.time()

    banner("RL-Route finishing run\n"
           f"  python      {PY}\n"
           f"  repo        {PROJECT_ROOT}\n"
           f"  n_perm      {args.n_perm}   n_jobs {args.n_jobs}\n"
           "  estimate    ~90 min (stage 1) + ~2 h (stage 3) + a few minutes for the rest")

    # ---- STAGE 1: the only expensive, genuinely required stage ---------------------------
    if not args.skip_canonical:
        banner("STAGE 1/4  Regenerate canonical_results.json with the fixed Score-Gap sweep\n"
               "           15 seeds x 3 datasets x 150 epochs. ~90 minutes. Required.")
        # canonical_evaluation.py overwrites its --output unconditionally; the resume logic
        # lives in run_full_pipeline.py, which is why this calls the stage directly rather
        # than going through the pipeline (which would skip it, the artifact existing).
        #
        # The three provenance flags are NOT optional. provenance.py records gold_type,
        # pool_size and run_tag from the command line, and the existing 100K artifact carries
        # real_wikipedia / 100000 / pool100k. Omitting them would produce a results file whose
        # provenance block reads null, which is exactly the ambiguity provenance.py exists to
        # prevent -- and it is how canonical_results_all.json became indistinguishable from
        # the headline run at a glance.
        run("canonical_evaluation",
            [os.path.join(HERE, "canonical_evaluation.py"),
             "--gold-type", args.gold_type,
             "--pool-size", str(args.pool_size),
             "--run-tag", args.run_tag],
            required=True, dry_run=args.dry_run, log=log)
        if not args.dry_run:
            print("\n  Verifying the fix actually landed:")
            if not verify_score_gap_fix():
                sys.exit("Stage 1 produced stale numbers. Stopping before anything reads them.")
    else:
        banner("STAGE 1/4  SKIPPED by request")
        if not args.dry_run:
            print("  Checking the existing file is post-fix anyway:")
            if not verify_score_gap_fix():
                sys.exit("canonical_results.json is pre-fix and --skip-canonical was passed. "
                         "Re-run without that flag.")

    # ---- STAGE 2: cheap, and every one of these reads canonical_results.json -------------
    banner("STAGE 2/4  Regenerate everything downstream of the results table\n"
           "           Seconds each. All read canonical_results.json, so they had to wait.")
    run("generate_latex_tables", [os.path.join(HERE, "generate_latex_tables.py")],
        required=False, dry_run=args.dry_run, log=log)
    run("headroom_analysis", [os.path.join(HERE, "headroom_analysis.py")],
        required=False, dry_run=args.dry_run, log=log)
    run("gold_type_analysis", [os.path.join(HERE, "gold_type_analysis.py")],
        required=False, dry_run=args.dry_run, log=log)
    run("cross_retriever_analysis", [os.path.join(HERE, "cross_retriever_analysis.py")],
        required=False, dry_run=args.dry_run, log=log)

    # ---- STAGE 3: the nulls, and the two analyses whose scripts changed ------------------
    if not args.skip_nulls:
        banner("STAGE 3/4  The controls\n"
               "           permutation_control is the expensive one: ~2 h at n_perm=20.\n"
               "           This is the stage that produces the paper's strongest claim.")

        # Try all three datasets in one process so the POOLED null is computed -- it is only
        # produced when every dataset is present, since pooling a subset would report a
        # different quantity. If the worker pool dies on memory (which is what happens on the
        # dev laptop), fall back to one dataset per process and accept losing pooled.
        combined_ok = run(
            "permutation_control (all 3 + pooled)",
            [os.path.join(HERE, "permutation_control.py"),
             "--n-perm", str(args.n_perm), "--n-jobs", str(args.n_jobs),
             "--tag", "100K Wikipedia pool"],
            required=False, dry_run=args.dry_run, log=log)

        if not combined_ok and not args.dry_run:
            print("\n  Combined run failed -- almost certainly peak memory in the worker pool.")
            print("  Falling back to one dataset per process. Pooled will not be produced,")
            print("  because a pooled null over a subset is a different quantity.\n")
            for ds in DATASETS:
                run(f"permutation_control [{ds}]",
                    [os.path.join(HERE, "permutation_control.py"),
                     "--datasets", ds, "--n-perm", str(args.n_perm),
                     "--n-jobs", "4", "--tag", f"100K Wikipedia pool ({ds})"],
                    required=False, dry_run=args.dry_run, log=log)

        # Both scripts changed since their last run, so their committed artifacts are stale.
        run("two_arm_ablation", [os.path.join(HERE, "two_arm_ablation.py"),
                                 "--n-jobs", str(args.n_jobs)],
            required=False, dry_run=args.dry_run, log=log)
        run("oracle_inflation", [os.path.join(HERE, "oracle_inflation.py")],
            required=False, dry_run=args.dry_run, log=log)
    else:
        banner("STAGE 3/4  SKIPPED by request")

    # ---- STAGE 4: figures last, because they read stage 2's output -----------------------
    banner("STAGE 4/4  Paper figures\n"
           "           Reads headroom_analysis.json from stage 2, so it goes last.")
    run("make_paper_figures", [os.path.join(HERE, "make_paper_figures.py")],
        required=False, dry_run=args.dry_run, log=log)

    # ---- report -------------------------------------------------------------------------
    banner("SUMMARY")
    width = max(len(l) for l, _, _ in log) if log else 10
    for label, status, el in log:
        print(f"  {label:<{width}}  {status:<8}  {el/60:6.1f} min")
    failed = [l for l, s, _ in log if s in ("FAILED", "failed")]
    print(f"\n  total {(time.time() - t_start)/60:.1f} min")
    if failed:
        print(f"  {len(failed)} stage(s) failed: {', '.join(failed)}")
        print("  These are additive analyses; the results table and controls above them are "
              "unaffected.")
    else:
        print("  all stages ok")

    print("\n  Commit and push these:")
    for f in ("canonical_results.json", "main_results_table.tex", "headroom_analysis.json",
              "headroom_analysis.md",
              # B-tagged: permutation_control.py writes into results/perm_B<n>/, because B fixes
              # the attainable p-floor and two runs at different B must not share a filename.
              os.path.join("perm_B1000", "permutation_control.json"),
              os.path.join("perm_B1000", "permutation_control.md"),
              "two_arm_ablation.json", "two_arm_ablation.md", "oracle_inflation.json",
              "oracle_inflation.md", "gold_type_stratified.json",
              "cross_retriever_analysis.json"):
        mark = "+" if os.path.exists(os.path.join(RESULTS, f)) else "-"
        print(f"    {mark} results/{f}")
    print("    + results/figures/")
    print("\n    git add results/ && git commit -m 'Lab finishing run: post-score-gap-fix "
          "results and the full permutation control' && git push")


if __name__ == "__main__":
    main()
