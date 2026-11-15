#!/usr/bin/env python3
"""Regenerate every results file from raw data, in dependency order.

Stages: data preparation and retrieval (experiments/baseline_v2), feature and reward tables
(src/router/prepare_env_data.py), then the routing analyses in this directory. Each stage is
skipped if its output already exists; pass --force to rerun everything, or delete one output
to rerun that stage. Nothing here needs a GPU. For a wall-clock estimate on your machine, run
estimate_pipeline_time.py first (read-only, 1-2 minutes).

Usage:
    python experiments/router_experiments/run_full_pipeline.py
    python experiments/router_experiments/run_full_pipeline.py --force
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["PYTHONPATH"] = PROJECT_ROOT
sys.path.insert(0, PROJECT_ROOT)


# Written whenever the pipeline runs at smoke scale. Its existence means every artifact on
# disk is a 20-query toy, and the resume logic ("skip stages whose outputs already exist")
# would otherwise carry those toys straight into the real run's tables. A one-shot run is
# exactly the situation where that must not be possible.
SMOKE_MARKER = os.path.join(PROJECT_ROOT, "results", ".smoke_run_marker")


def wiki_pool_size():
    """How many passages the corpora were actually mined from.

    Counted from the cache rather than read from 01_prepare_data.WIKI_POOL_SIZE, because
    that constant is the *intended* size and the two have disagreed before -- 01 itself
    warns when they do. Stamping the intended value would make two runs at genuinely
    different pool sizes look identical in provenance, which is the one thing the
    20K-vs-100K comparison cannot afford.
    """
    path = os.path.join(PROJECT_ROOT, "experiments", "baseline_v2", "cache",
                        "wiki_passage_pool.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return len(json.load(f))
    except (OSError, ValueError):
        return None


def run_tag():
    """Short label for this run, derived from the real pool size: pool20k, pool100k, ..."""
    n = wiki_pool_size()
    return f"pool{n // 1000}k" if n else "pool-unknown"


def run_stage(name, script_path, args=None, check_output=None, force=False):
    """Run one pipeline stage as a subprocess, skipping it if its expected output
    already exists (unless force=True). Returns True on success/skip, False on failure."""
    args = args or []

    if check_output and not force:
        missing = [p for p in check_output if not os.path.exists(p)]
        if not missing:
            print(f"\n[SKIP] {name} — all expected outputs already exist:")
            for p in check_output:
                print(f"       {os.path.relpath(p, PROJECT_ROOT)}")
            print("       (pass --force to regenerate anyway)")
            return True

    print(f"\n{'='*70}")
    print(f"RUNNING: {name}")
    print(f"Script:  {os.path.relpath(script_path, PROJECT_ROOT)} {' '.join(args)}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path] + args,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
    )
    elapsed = time.time() - start

    status = "SUCCESS" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    print(f"\n{'='*70}")
    print(f"FINISHED: {name} — {status} in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}\n")

    if result.returncode != 0:
        return False

    if check_output:
        missing = [p for p in check_output if not os.path.exists(p)]
        if missing:
            print(f"WARNING: {name} exited 0 but expected output(s) missing: {missing}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Regenerate the full RL router pipeline, from raw data to paper tables/figures.")
    parser.add_argument("--force", action="store_true", help="Regenerate every stage even if its output already exists.")
    parser.add_argument("--smoke", type=int, default=None, metavar="N",
                        help="Plumbing check, not an experiment: run every stage on N "
                             "questions per dataset with 1 seed and 5 epochs. Implies "
                             "--force. Verifies each stage executes and writes its outputs "
                             "before committing a compute window to a one-shot run. The "
                             "numbers produced are meaningless and the run is recorded, so "
                             "the next real run cannot silently resume from them.")
    parser.add_argument(
        "--gold-type", type=str, default="real_wikipedia",
        choices=["real_wikipedia", "all", "constructed"],
        help="Gold-passage provenance for the HEADLINE env data (stage 3). Default "
             "'real_wikipedia' drops queries whose gold passage is a generated template "
             "(01_prepare_data.py's construct_gold fallback), which BM25 cannot match by "
             "construction — an artifact that suppresses real routing headroom. The "
             "unfiltered 'all' variant is always built alongside it as *_env_data_all.json "
             "for the appendix table, so nothing is lost.",
    )
    parser.add_argument("--skip-appendix", action="store_true", help="Skip the unfiltered all-queries appendix evaluation (stage 4b, ~2.5hrs).")
    parser.add_argument("--reward", type=str, default="rank", choices=["rank", "mrr"],
                        help="Training reward. 'rank' (default) = 1/rank of the first "
                             "answer-bearing passage over the whole corpus; 'mrr' = MRR@k. "
                             "MRR@k is 0 beyond rank k, leaving 70-83%% of queries with every "
                             "retriever tied at 0; the rank reward raises the share of "
                             "queries with a strictly-best action from ~33%% to ~92%%. "
                             "MRR@k is still what gets reported either way.")
    parser.add_argument("--embed-dims", type=int, default=32,
                        help="PCA components of the query embedding appended as router "
                             "features (default 32, 0 disables). Query semantics are not "
                             "expressible by the handcrafted surface features.")
    parser.add_argument("--reward-k", type=str, default="20", choices=["1", "3", "5", "10", "20"],
                        help="Cutoff k for the reward (MRR@k), default 20. Costs nothing — "
                             "02_run_baselines.py computes every k. At k=5 the reward is so "
                             "sparse that 70-83%% of queries have all retrievers scoring 0, "
                             "leaving no routing signal; k=20 roughly doubles the share of "
                             "queries with a strictly-best action at equal oracle headroom.")
    args = parser.parse_args()

    # A smoke run must not be resumable-from. Its outputs look exactly like real ones to
    # the "skip if the file exists" check, so without this a plumbing check silently
    # supplies 20-query tables to the next real run -- the precise failure a one-shot
    # compute window cannot absorb.
    if args.smoke:
        args.force = True
        os.makedirs(os.path.dirname(SMOKE_MARKER), exist_ok=True)
        with open(SMOKE_MARKER, "w", encoding="utf-8") as f:
            f.write(f"smoke run: {args.smoke} questions/dataset, 1 seed, 5 epochs\n")
    elif os.path.exists(SMOKE_MARKER):
        print("\n" + "=" * 70)
        print("Artifacts on disk were produced by a SMOKE run (20-query toys).")
        print("Forcing full regeneration so they cannot reach the real results.")
        print("=" * 70 + "\n")
        args.force = True

    print("RL router Full Pipeline (post-fix)")
    print(f"Project Root: {PROJECT_ROOT}")
    if args.smoke:
        print(f"*** SMOKE MODE: {args.smoke} questions/dataset, 1 seed, 5 epochs. "
              f"The numbers are meaningless; this checks that every stage runs. ***")
        print(f"*** This OVERWRITES the real prepared corpora and results with toys. A "
              f"marker is written to {os.path.relpath(SMOKE_MARKER, PROJECT_ROOT)}, and "
              f"the next non-smoke run regenerates everything, so the damage is "
              f"self-healing -- but do not stop between the two. ***")
    print(f"Headline gold-type filter: {args.gold_type}")

    try:
        import torch
        print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("WARNING: PyTorch not found in current environment — activate your venv first.")

    RE = os.path.join(PROJECT_ROOT, "experiments", "router_experiments")
    BV2 = os.path.join(PROJECT_ROOT, "experiments", "baseline_v2")
    RESULTS = os.path.join(PROJECT_ROOT, "results")

    # Per-stage smoke arguments. Each script names the flag differently, so they cannot be
    # passed uniformly; an unrecognised flag would abort the very run meant to prove the
    # plumbing works.
    smoke_prepare = ["--num_questions", str(args.smoke)] if args.smoke else []
    smoke_baselines = ["--max_questions", str(args.smoke)] if args.smoke else []
    smoke_train = ["--seeds", "1", "--epochs", "5"] if args.smoke else []
    # The permutation control costs one full CV fit per replicate per dataset, so its runtime
    # is set by --n-perm and not by anything else in the pipeline. 20 keeps the attainable
    # p-value floor at 1/21 = 0.048, which is what "significant at 0.05" requires; 2 in smoke
    # mode because smoke checks that the stage executes, not what it concludes.
    smoke_perm = ["--n-perm", "2"] if args.smoke else ["--n-perm", "20"]

    stages = [
        (
            "1. Prepare raw QA + 500-passage corpora (NQ/TriviaQA/HotpotQA)",
            os.path.join(BV2, "01_prepare_data.py"),
            smoke_prepare,
            [
                os.path.join(BV2, "data", "nq_prepared.json"),
                os.path.join(BV2, "data", "triviaqa_prepared.json"),
                os.path.join(BV2, "data", "hotpotqa_prepared.json"),
            ],
        ),
        (
            "2. Run BM25 / Dense (FAISS) / Hybrid (RRF) retrieval + metrics (fixed: raw scores + word-boundary matching)",
            os.path.join(BV2, "02_run_baselines.py"),
            ["--dataset", "all"] + smoke_baselines,
            [
                os.path.join(BV2, "results", "nq_per_query_results.json"),
                os.path.join(BV2, "results", "triviaqa_per_query_results.json"),
                os.path.join(BV2, "results", "hotpotqa_per_query_results.json"),
            ],
        ),
        (
            "3. Extract router features + build RL environment lookup tables (fixed: real retrieval scores, not MRR)",
            None,  # module, run separately below
            [],
            [
                os.path.join(PROJECT_ROOT, "src", "router", "nq_env_data.json"),
                os.path.join(PROJECT_ROOT, "src", "router", "triviaqa_env_data.json"),
                os.path.join(PROJECT_ROOT, "src", "router", "hotpotqa_env_data.json"),
            ],
        ),
        (
            # Needs only env data (stage 3), no training, and takes ~1 min. It is the
            # evidence for the paper's central claim, so it runs BEFORE the multi-hour
            # training stages — an interrupted run must not lose it.
            "3b. Routing learnability: headroom, reward sparsity, honest-vs-leaky predictability",
            os.path.join(RE, "learnability_analysis.py"),
            ["--leaky-suffix", "_leaky", "--tag", "full run"],
            [os.path.join(RESULTS, "learnability_analysis.md")],
        ),
        (
            # Also cheap (no RL training) and also central evidence, so it sits beside 3b
            # ahead of the multi-hour stages.
            "3c. Per-retriever routing: one regressor per retriever vs. the 3-way classifier",
            os.path.join(RE, "per_retriever_router.py"),
            [],
            [os.path.join(RESULTS, "per_retriever_router.md")],
        ),
        (
            # Needs stage 2's per-query fusion ranks and stage 3's env data; no training.
            # Sits with 3b/3c ahead of the multi-hour stages because it answers the
            # paper's most damaging question -- "why route instead of fusing the lists you
            # already have?" -- and an interrupted run must not lose that answer.
            "3d. Fusion baselines: RRF (Hybrid) and tuned convex combination vs. the arms",
            os.path.join(RE, "fusion_baseline.py"),
            [],
            [os.path.join(RESULTS, "fusion_baseline.md")],
        ),
        (
            # Answers "would more queries have helped?" from the data already on disk, so
            # the negative result can be reported as bias-limited rather than merely
            # asserted. Reads env data only; no training.
            "3e. Learning curve: is routing predictability bias-limited or data-limited?",
            os.path.join(RE, "learning_curve.py"),
            [],
            [os.path.join(RESULTS, "learning_curve.json")],
        ),
        (
            # The leak gradient. honest-vs-leaky establishes that contamination matters;
            # this establishes the DOSE -- partial, coarser, and jittered leaks -- which is
            # the question a reader auditing their own router actually has. Reads the env
            # arms stage 3 already built; no training, unlike leakage_ablation.py which
            # invokes canonical_evaluation per arm at ~86 min each.
            "3f. Leakage dose-response across all feature-mode arms",
            os.path.join(RE, "leak_gradient.py"),
            [],
            [os.path.join(RESULTS, "leak_gradient.md")],
        ),
        (
            # The control that turns the negative result from "we could not find signal" into
            # "the features are indistinguishable from noise". 3b's lift is measured against
            # the majority baseline, which cannot separate "no signal" from "signal too weak
            # to beat a constant policy", and 3e's rising curve is equally consistent with
            # both. These two nulls separate them. Env data only; no training. Placed here
            # because it is now the strongest single piece of evidence in the paper.
            "3g. Permutation and random-feature controls: is there any signal at all?",
            os.path.join(RE, "permutation_control.py"),
            smoke_perm + ["--tag", "full run"],
            [os.path.join(RESULTS, "permutation_control.md")],
        ),
        (
            # Answers "is the negative result an artifact of a redundant action space?".
            # Hybrid is RRF of BM25 and Dense, so one arm is a function of the other two;
            # this re-runs the core measurement on two independent retrievers, and reports
            # the minimum detectable effect so "no effect" is stated alongside "and here is
            # the smallest effect this design could have seen". Env data only.
            "3h. Two-arm ablation (drop Hybrid) and minimum detectable effect",
            os.path.join(RE, "two_arm_ablation.py"),
            [],
            [os.path.join(RESULTS, "two_arm_ablation.md")],
        ),
        (
            # Oracle headroom is the motivation for the whole paper and it is a max over
            # three noisy arms, so it is upward-biased by Jensen. This measures the arm
            # selection bias directly, the sensitivity of headroom to injected noise, and
            # how much of it sits on near-ties. Env data only; no training.
            "3i. Oracle inflation: how much of the reported headroom is real?",
            os.path.join(RE, "oracle_inflation.py"),
            [],
            [os.path.join(RESULTS, "oracle_inflation.md")],
        ),
        (
            "4. Canonical evaluation: static/heuristic/supervised/RL baselines, 15 seeds, bootstrap significance",
            os.path.join(RE, "canonical_evaluation.py"),
            ["--gold-type", args.gold_type, "--pool-size", str(wiki_pool_size()),
             "--run-tag", run_tag()] + smoke_train,
            [os.path.join(RESULTS, "canonical_results.json")],
        ),
        (
            # Reads canonical_results.json, so it must follow stage 4. Capture per method,
            # and the rank bucket the headroom actually sits in -- both were being computed
            # by hand, with no producing artifact behind either number.
            "4a. Headroom: share captured per method, and where the headroom lives",
            os.path.join(RE, "headroom_analysis.py"),
            [],
            [os.path.join(RESULTS, "headroom_analysis.md")],
        ),
        (
            "5. Generate main results LaTeX table",
            os.path.join(RE, "generate_latex_tables.py"),
            [],
            [os.path.join(RESULTS, "main_results_table.tex")],
        ),
        # 4b and 12 are deliberately LAST despite their numbering: 4b is a ~2.5h appendix
        # robustness check, and 12 depends on its output. Everything the paper actually
        # needs (Table 1, Figure 1, the ablation) finishes first, so a late failure or a
        # truncated run still leaves a submittable result set.
        (
            # Depends only on the HEADLINE canonical_results.json, so it runs before the
            # expensive appendix — an interrupted run should still leave the mechanism figure.
            "13. Cross-retriever agreement vs. routing headroom (mechanism check)",
            os.path.join(RE, "cross_retriever_analysis.py"),
            [],
            [os.path.join(RESULTS, "cross_retriever_analysis.md")],
        ),
        (
            "4b. Appendix evaluation on the UNFILTERED all-queries env data (robustness check)",
            os.path.join(RE, "canonical_evaluation.py"),
            ["--env-suffix", "_all", "--output", os.path.join(RESULTS, "canonical_results_all.json")]
            + smoke_train,
            [os.path.join(RESULTS, "canonical_results_all.json")],
        ),
        (
            "12. Stratify results by gold-passage provenance (real Wikipedia vs. constructed)",
            os.path.join(RE, "gold_type_analysis.py"),
            # Must read the UNFILTERED results — the headline run has already dropped every
            # constructed-gold query, so there would be nothing left to stratify.
            ["--results", os.path.join(RESULTS, "canonical_results_all.json")],
            [os.path.join(RESULTS, "gold_type_stratified.md")],
        ),
    ]

    if args.skip_appendix or args.gold_type == "all":
        # Stage 12 stratifies the appendix results, so it goes when the appendix goes.
        stages = [s for s in stages if not (s[0].startswith("4b.") or s[0].startswith("12."))]
        if args.skip_appendix:
            print("Skipping the unfiltered appendix evaluation and its stratification (--skip-appendix).")
        else:
            print("Headline run is already unfiltered (--gold-type all) — no separate appendix needed.")

    overall_start = time.time()
    for name, script_path, extra_args, check_output in stages:
        if script_path is None:
            # Stage 3 is a module invocation, not a standalone script. It runs TWICE: once
            # filtered to --gold-type under the default filenames (so every downstream
            # script picks up the corrected data with no changes), and once unfiltered
            # under *_env_data_all.json for the appendix table.
            print(f"\n{'='*70}\nRUNNING: {name}\nModule: python -m src.router.prepare_env_data\n{'='*70}\n")
            if not args.force and all(os.path.exists(p) for p in check_output):
                print("[SKIP] outputs already exist (pass --force to regenerate)")
                continue
            t0 = time.time()
            # (gold_type, suffix, feature_mode)
            variants = [(args.gold_type, "", "honest_minimal")]
            if args.gold_type != "all":
                variants.append(("all", "_all", "honest_minimal"))
            # A leaky-feature variant of the SAME queries, so learnability_analysis.py can
            # quantify how much of the routing signal was reward leakage rather than skill.
            # Never used for a headline number — only as the comparison arm.
            variants.append((args.gold_type, "_leaky", "leaky"))
            # Graded leak arms, over the SAME queries. honest-vs-leaky answers whether
            # contamination matters; these answer how much it takes — a partial leak (one
            # arm), a coarser one (Hit@k instead of MRR), and a degraded one (MRR plus
            # jitter, which exact-equality leak detection cannot see at all). That turns
            # the ablation from a binary into a dose-response curve, which is the more
            # useful result for anyone auditing their own router.
            variants += [
                (args.gold_type, "_leak1", "leak_one_arm"),
                (args.gold_type, "_leakhit", "leak_hit"),
                (args.gold_type, "_leaknoisy", "leak_noisy"),
            ]
            ok = True
            for gt, sfx, fmode in variants:
                print(f"\n>>> env data: --gold-type {gt} --suffix {sfx!r} --feature-mode {fmode}"
                      f" --k {args.reward_k} --reward {args.reward} --embed-dims {args.embed_dims}")
                result = subprocess.run(
                    [sys.executable, "-m", "src.router.prepare_env_data",
                     "--gold-type", gt, "--suffix", sfx,
                     "--feature-mode", fmode, "--k", args.reward_k,
                     "--reward", args.reward, "--embed-dims", str(args.embed_dims)],
                    cwd=PROJECT_ROOT,
                    env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
                )
                if result.returncode != 0:
                    ok = False
                    break
            elapsed = time.time() - t0
            ok = ok and all(os.path.exists(p) for p in check_output)
            print(f"FINISHED: {name} — {'SUCCESS' if ok else 'FAILED'} in {elapsed:.1f}s")
            if not ok:
                print(f"\nPipeline stopped at: {name}")
                sys.exit(1)
            continue

        ok = run_stage(name, script_path, extra_args, check_output, force=args.force)
        if not ok:
            print(f"\nPipeline stopped at: {name}")
            print("Fix the error above and re-run this script — already-completed stages will be skipped.")
            sys.exit(1)

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print("\nGenerated/refreshed:")
    print(f"  - {os.path.join(RESULTS, 'canonical_results.json')}  (gold_type={args.gold_type})")
    print(f"  - {os.path.join(RESULTS, 'canonical_results_all.json')}  (appendix: unfiltered all-queries robustness check)")
    print(f"  - {os.path.join(RESULTS, 'cross_retriever_analysis.md')} + .pdf  (mechanism: headroom concentrates where retrievers disagree)")
    print(f"  - {os.path.join(RESULTS, 'main_results_table.tex')}")
    print(f"  - {os.path.join(RESULTS, 'gold_type_stratified.md')}  (real_wikipedia vs. constructed breakdown — headline the real subset)")
    print("\nNext: diff these against the committed versions to see whether the run moved")
    print("the numbers, then update paper/main.tex and REPRODUCE.md accordingly.")


if __name__ == "__main__":
    main()
