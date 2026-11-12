"""
run_lab_full.py — Single entry point for the lab run. Windows and Linux.

Same phases as run_lab_full.sh, but driven from Python so it needs no bash, no tmux and no
Git Bash. On Windows this is the one to use; the .sh is kept for Linux clusters where the
existing run_on_cluster.sh wrapper is already the validated path.

Two things this gets right that a shell script cannot:

  * Every child process is launched with sys.executable, i.e. THE INTERPRETER RUNNING THIS
    FILE. There is no python/python3 ambiguity and no way to accidentally run a stage under
    a different environment than the one you installed into.
  * Paths are built with os.path.join, so drive letters and backslashes work unchanged.

Phase A is the main pipeline (run_full_pipeline.py) plus the supplementary experiments.
Phase B is the new analysis set: cost-aware degeneracy, the cross-scoring env arm, the
escalation-signal go/no-go, per-retriever routing, and the learned cascade — each with a
control run that omits cross-scores so any difference is attributable.

Phase B stages never abort the run. They are additive analyses, and by the time they run
Phase A's results are already complete, so a failure is recorded and reported at the end
rather than losing the day's work.

Usage:
    python experiments/router_experiments/run_lab_full.py
    python experiments/router_experiments/run_lab_full.py --skip-ablation --skip-appendix
    python experiments/router_experiments/run_lab_full.py --phase-b-only
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RE = os.path.join(PROJECT_ROOT, "experiments", "router_experiments")
BV2 = os.path.join(PROJECT_ROOT, "experiments", "baseline_v2")
RESULTS = os.path.join(PROJECT_ROOT, "results")
LOGS = os.path.join(PROJECT_ROOT, "logs")

PY = sys.executable          # never "python" / "python3" — see module docstring
FAILED = []
LOG_PATH = None

# Windows consoles default to a legacy codepage (cp1252), which raises UnicodeEncodeError
# on any non-ASCII byte in a child stage's output — an em-dash in a progress line is enough
# to kill the driver mid-run. Force UTF-8 with replacement so output is never fatal.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Children inherit this, so their own prints are UTF-8 too rather than dying on the way out.
CHILD_ENV = os.environ.copy()
CHILD_ENV["PYTHONIOENCODING"] = "utf-8"
CHILD_ENV["PYTHONUTF8"] = "1"


def log(msg=""):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        # Last-resort path for a console that refuses UTF-8 even after reconfigure.
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8", errors="replace") as f:
            f.write(msg + "\n")


def hdr(msg):
    log("")
    log("=" * 70)
    log(msg)
    log("=" * 70)


def run(cmd, label, soft=True):
    """Run a child process, streaming output to console and log.

    `soft` failures are recorded and execution continues. Output is streamed rather than
    captured so a multi-hour stage shows progress live instead of going silent.
    """
    log("")
    log(f">>> {label}")
    log(f"    $ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1,
                                env=CHILD_ENV)
        for line in proc.stdout:
            log(line.rstrip())
        rc = proc.wait()
    except Exception as exc:
        rc = -1
        log(f"    launch failed: {type(exc).__name__}: {exc}")

    mins = (time.time() - t0) / 60.0
    if rc == 0:
        log(f"    OK: {label}  ({mins:.1f} min)")
        return True
    if soft:
        log(f"    FAILED: {label} (rc={rc}, {mins:.1f} min) — continuing.")
        FAILED.append(label)
        return False
    log(f"    FATAL: {label} (rc={rc}) — stopping.")
    sys.exit(1)


def preflight():
    hdr("Preflight")
    log(f"  Python:     {sys.version.split()[0]}")
    log(f"  Executable: {PY}")
    ok = True
    for m in ("torch", "numpy", "sklearn", "faiss", "sentence_transformers"):
        try:
            __import__(m)
            log(f"  {m}: ok")
        except ImportError as e:
            log(f"  {m}: MISSING ({e})")
            ok = False
    try:
        import torch
        cuda = torch.cuda.is_available()
        log(f"  torch build: {torch.__version__}")
        log(f"  CUDA available: {cuda}")
        if cuda:
            log(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            # Worth shouting about: a CPU-only torch imports fine and produces correct
            # results, it just makes an already-long run far longer. That failure mode is
            # invisible unless it is checked for explicitly.
            log("  WARNING: no GPU visible. If this machine HAS a GPU, you likely have a "
                "CPU-only torch build — reinstall with the matching CUDA index-url.")
    except Exception as exc:
        log(f"  torch check failed: {exc}")
    try:
        import spacy
        spacy.load("en_core_web_sm")
        log("  spaCy en_core_web_sm: ok")
    except Exception as exc:
        log(f"  WARNING: spaCy model unavailable ({exc}); entity features fall back to zeros.")

    for name, path in (("data", os.path.join(BV2, "data")),
                       ("cache", os.path.join(BV2, "cache"))):
        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if f.endswith(".json")]
            size = sum(os.path.getsize(os.path.join(path, f)) for f in files) / 1e9
            log(f"  {name}: {len(files)} json files, {size:.2f} GB")
        else:
            log(f"  WARNING: {path} missing.")
    try:
        free = shutil.disk_usage(PROJECT_ROOT).free / 1e9
        log(f"  free disk: {free:.1f} GB" + ("" if free > 10 else "   <-- LOW, want ~10 GB"))
    except Exception:
        pass

    if not ok:
        log("\nFATAL: install dependencies first:  pip install -r requirements.txt")
        sys.exit(1)


def check_cross_scores():
    """Confirm the retrieval stage persisted cross_scores before Phase B consumes them."""
    import json
    hdr("Checking cross-scoring data landed")
    p = os.path.join(BV2, "results", "nq_per_query_results.json")
    if not os.path.exists(p):
        log(f"  WARNING: {p} not found — cannot verify.")
        return
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        recs = d.get("per_query_results", d) if isinstance(d, dict) else d
        if isinstance(recs, dict):
            recs = next((v for v in recs.values() if isinstance(v, list)), [])
        r = recs[0]
        has_cs = bool(r.get("retrievers", {}).get("bm25", {}).get("cross_scores"))
        log(f"  cross_scores present:     {has_cs}")
        log(f"  cross_score_norm present: {bool(r.get('cross_score_norm'))}")
        if not has_cs:
            log("  WARNING: retrieval results predate the cross-scoring change. Re-run "
                "stage 2 or the _xscore arm will be degraded.")
    except Exception as exc:
        log(f"  WARNING: could not verify ({exc}).")


def main():
    global LOG_PATH
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-ablation", action="store_true", help="Skip the ~4-5 hr ablation.")
    p.add_argument("--skip-appendix", action="store_true", help="Skip the ~2.5 hr appendix.")
    p.add_argument("--skip-failure-mode", action="store_true")
    p.add_argument("--skip-supplementary", action="store_true")
    p.add_argument("--phase-b-only", action="store_true",
                   help="Reuse an existing Phase A run; analyses only (minutes, no GPU).")
    p.add_argument("--force", action="store_true", help="Regenerate stages that already ran.")
    p.add_argument("--smoke", type=int, default=None, metavar="N",
                   help="Plumbing check across BOTH phases: run everything on N questions "
                        "per dataset with 1 seed and 5 epochs. Phase A gets --smoke N; "
                        "Phase B runs on the resulting tiny env data. Use 60 rather than "
                        "20 -- the gold-type filter drops roughly half, and Phase B's "
                        "cross-validation needs enough members per class to split. The "
                        "numbers are meaningless; this only proves every stage executes.")
    p.add_argument("--gold-type", default="real_wikipedia",
                   choices=["real_wikipedia", "all", "constructed"])
    p.add_argument("--reward", default="rank", choices=["rank", "mrr"])
    p.add_argument("--embed-dims", default="32")
    p.add_argument("--reward-k", default="20")
    p.add_argument("--unit-costs", default="1,3,4")
    p.add_argument("--fail-k", default="20")
    p.add_argument("--cascade-lambda", default="0.02")
    args = p.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = os.path.join(LOGS, f"lab_{stamp}.log")

    hdr(f"RL-Route lab run — {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Repo: {PROJECT_ROOT}")
    log(f"Log:  {LOG_PATH}")
    preflight()

    if not args.phase_b_only:
        # Untracked results are the only copy of the previous numbers, so back them up
        # before anything overwrites them.
        if os.path.isdir(RESULTS) and os.listdir(RESULTS):
            backup = os.path.join(PROJECT_ROOT, f"results_backup_{stamp}")
            try:
                shutil.copytree(RESULTS, backup)
                log(f"\nBacked up results/ -> {backup}")
            except Exception as exc:
                log(f"\nWARNING: backup failed ({exc}); continuing.")

        hdr("PHASE A — main pipeline")
        log("This is the long one. Everything after it is minutes.")
        cmd = [PY, os.path.join(RE, "run_full_pipeline.py"),
               "--gold-type", args.gold_type, "--reward", args.reward,
               "--embed-dims", str(args.embed_dims), "--reward-k", str(args.reward_k)]
        if args.smoke:
            # Phase A shrinks explicitly; Phase B inherits the small env data it produces,
            # so it needs no flag of its own. Phase B failures are soft and land in the
            # end-of-run summary rather than aborting -- at smoke scale some of them are
            # genuine sample-size limits (cross-validation needs members per class), not
            # code faults, and the error text distinguishes the two.
            cmd += ["--smoke", str(args.smoke)]
        for flag, on in (("--skip-ablation", args.skip_ablation),
                         ("--skip-appendix", args.skip_appendix),
                         ("--skip-failure-mode", args.skip_failure_mode),
                         ("--force", args.force)):
            if on:
                cmd.append(flag)
        # Not soft: everything downstream reads what this produces, so continuing past a
        # failure here would only generate confusing secondary errors.
        run(cmd, "Phase A: run_full_pipeline.py", soft=False)

        if not args.skip_supplementary:
            hdr("Supplementary experiments (paper sections 4.2 / 4.4 / 4.5)")
            for script, desc in (("15_grpo_comparison.py", "algorithm comparison"),
                                 ("07_cross_dataset.py", "cross-dataset transfer"),
                                 ("17_latency_analysis.py", "measured latency")):
                run([PY, os.path.join(RE, script)], f"{script} — {desc}")
    else:
        log("\nPhase A skipped (--phase-b-only); reusing existing results/ and env data.")

    check_cross_scores()

    hdr("PHASE B — cost-aware, cross-scoring, cascade (CPU, ~25 min)")

    run([PY, os.path.join(RE, "cost_aware_analysis.py"), "--unit-costs", args.unit_costs],
        "B1 cost-aware degeneracy check")

    run([PY, os.path.join(RE, "cost_aware_analysis.py"), "--unit-costs", "1,1,2",
         "--output", os.path.join(RESULTS, "cost_aware_analysis_equalcost.json"),
         "--output-md", os.path.join(RESULTS, "cost_aware_analysis_equalcost.md")],
        "B1b cost-aware sensitivity (equal BM25/Dense cost — degenerate by algebra)")

    run([PY, "-m", "src.router.prepare_env_data", "--cross-scores", "--suffix", "_xscore",
         "--gold-type", args.gold_type, "--reward", args.reward,
         "--embed-dims", str(args.embed_dims), "--k", str(args.reward_k)],
        "B2 build _xscore env arm")

    run([PY, os.path.join(RE, "escalation_signal_check.py"),
         "--env-suffix", "_xscore", "--k", str(args.fail_k)],
        "B3 escalation signal check (cascade go/no-go)")

    run([PY, os.path.join(RE, "per_retriever_router.py"), "--env-suffix", "_xscore",
         "--output", os.path.join(RESULTS, "per_retriever_router_xscore.json"),
         "--output-md", os.path.join(RESULTS, "per_retriever_router_xscore.md")],
        "B4 per-retriever routing WITH cross-scores")

    run([PY, os.path.join(RE, "per_retriever_router.py"),
         "--output", os.path.join(RESULTS, "per_retriever_router_base.json"),
         "--output-md", os.path.join(RESULTS, "per_retriever_router_base.md")],
        "B4b per-retriever routing WITHOUT cross-scores (control)")

    run([PY, os.path.join(RE, "cascade_router.py"), "--env-suffix", "_xscore",
         "--k", str(args.fail_k), "--lam", str(args.cascade_lambda)],
        "B5 learned cross-scoring cascade")

    run([PY, os.path.join(RE, "cascade_router.py"), "--env-suffix", "",
         "--k", str(args.fail_k), "--lam", str(args.cascade_lambda), "--no-reinforce",
         "--output", os.path.join(RESULTS, "cascade_router_base.json"),
         "--output-md", os.path.join(RESULTS, "cascade_router_base.md")],
        "B5b cascade WITHOUT cross-scores (control)")

    hdr(f"DONE — {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Full log: {LOG_PATH}")
    log("")
    log("Read in this order — each decides whether the next matters:")
    log("  1. results/cost_aware_analysis.md          Is there a decision once cost counts?")
    log("  2. results/escalation_signal.md            Is failure detectable post-hoc?")
    log("  3. results/cascade_router.md               THE METHOD. 'vs matched random'.")
    log("  4. results/cascade_router_base.md          Control, no cross-scores")
    log("  5. results/per_retriever_router_xscore.md  vs _base.md")
    log("  6. results/canonical_results.json          Main table")
    log("")
    if FAILED:
        log("Stages that FAILED (Phase A results are unaffected):")
        for s in FAILED:
            log(f"  - {s}")
    else:
        log("All stages succeeded.")
    log("=" * 70)


if __name__ == "__main__":
    main()
