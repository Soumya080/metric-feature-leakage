#!/bin/bash
# run_lab_full.sh — Single entry point for the GPU lab run.
#
# Phase A is the existing, already-validated pipeline (run_on_cluster.sh): data prep,
# retrieval, env construction, canonical evaluation, figures. Unchanged — this script
# delegates to it rather than reimplementing it, so there is one source of truth for the
# main results.
#
# Phase B is the new analysis set. It is CPU-only and takes minutes, but it runs here so
# everything lands in one log and one results/ directory:
#
#   B1  cost_aware_analysis.py     Is there a routing decision at all once compute is
#                                  charged? Seconds. Run first because it is the cheapest
#                                  go/no-go in the project.
#   B2  prepare_env_data.py        Build the _xscore env arm (adds 6 cross-scoring
#         --cross-scores           features). Separate suffix, so the main results from
#                                  Phase A are untouched and the two arms stay comparable.
#   B3  escalation_signal_check    Can a retriever's failure be detected from its own
#                                  output? Decides whether the cascade is worth building.
#   B4  per_retriever_router.py    Does routing beat best-fixed with cross-scores added?
#   B5  cascade_router.py          THE METHOD. Run the cheap retriever, inspect what it
#                                  returned, escalate to Hybrid only when the evidence says
#                                  it failed. Judged against escalating at random at the
#                                  SAME rate — that is what separates targeting from simply
#                                  spending more compute.
#
# Each B stage has a matching control run without cross-scoring features, so any difference
# is attributable to the features rather than to anything else that changed.
#
# Phase B stages never abort the run. They are additive analyses; if one fails, Phase A's
# results are still complete and valid, so a failure is reported and the script continues.
#
# Usage:
#   cd efficient-rag-pipeline
#   bash experiments/router_experiments/run_lab_full.sh
#
#   # Survives SSH disconnect:
#   nohup bash experiments/router_experiments/run_lab_full.sh > /dev/null 2>&1 &
#   tail -f logs/lab_*.log
#
#   # Analyses only, reusing an existing Phase A run (minutes, no GPU):
#   SKIP_PIPELINE=1 bash experiments/router_experiments/run_lab_full.sh
#
# Passes through every run_on_cluster.sh override: SKIP_ABLATION, SKIP_FAILURE_MODE,
# SKIP_CALIBRATION, SKIP_APPENDIX, SKIP_SUPPLEMENTARY, SKIP_BACKUP, FORCE, GOLD_TYPE.
# Phase B knobs: REWARD, EMBED_DIMS, REWARD_K, UNIT_COSTS, FAIL_K.

# pipefail is load-bearing, not decoration. Every stage below is piped into `tee`, and a
# pipeline's exit status is its LAST command's — so without it, `cmd | tee` reports tee's
# success and every failure is silently swallowed: Phase B stages would all report OK, and
# a failed Phase A would fall through into Phase B instead of stopping the run.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/lab_${TIMESTAMP}.log"

GOLD_TYPE="${GOLD_TYPE:-real_wikipedia}"
REWARD="${REWARD:-rank}"
EMBED_DIMS="${EMBED_DIMS:-32}"
REWARD_K="${REWARD_K:-20}"
UNIT_COSTS="${UNIT_COSTS:-1,3,4}"
FAIL_K="${FAIL_K:-20}"
# Quality is ~0-1 and cascade cost is 1-4, so 0.02 trades roughly one MRR point per cost
# unit. Only selects the reported operating point on the Pareto curve — the full curve is
# written out regardless, so this is a presentation choice, not a result.
CASCADE_LAMBDA="${CASCADE_LAMBDA:-0.02}"

FAILED_STAGES=()

# Use the interpreter that actually has the dependencies, rather than assuming a name.
# Inside an activated venv `python` and `python3` are the same binary, but outside one they
# routinely are not: on Windows `python3` resolves to a Microsoft Store stub with an empty
# site-packages, and on some minimal Linux images only one of the two names exists. Probing
# for a real import turns a confusing mid-run ImportError into a correct choice up front.
# Override explicitly with PYTHON=/path/to/python.
detect_python() {
    local cand
    for cand in "${PYTHON:-}" python3 python; do
        [ -z "$cand" ] && continue
        command -v "$cand" >/dev/null 2>&1 || continue
        if "$cand" -c "import torch, sklearn, numpy" >/dev/null 2>&1; then
            echo "$cand"; return 0
        fi
    done
    # Nothing imports cleanly — fall back to any interpreter that exists so preflight can
    # report exactly which dependency is missing instead of dying on "command not found".
    for cand in python3 python; do
        command -v "$cand" >/dev/null 2>&1 && { echo "$cand"; return 0; }
    done
    return 1
}
PY="$(detect_python)" || { echo "FATAL: no python interpreter found on PATH."; exit 1; }

log()  { echo "$@" | tee -a "$LOG"; }
hdr()  { log ""; log "=================================================================="; log "$@"; log "=================================================================="; }

# Phase B stages are additive: a failure loses one analysis, not the run. Recorded and
# surfaced in the final summary so a failure is never silently lost in a long log.
soft() {
    local label="$1"; shift
    log ""
    log ">>> $label"
    if "$@" 2>&1 | tee -a "$LOG"; then
        log "    OK: $label"
    else
        log "    FAILED: $label (continuing — Phase A results are unaffected)"
        FAILED_STAGES+=("$label")
    fi
}

hdr "RL-Route lab run — $(date)"
log "Repo:  $REPO_ROOT"
log "Log:   $LOG"
log "Config: gold_type=$GOLD_TYPE reward=$REWARD embed_dims=$EMBED_DIMS reward_k=$REWARD_K"
log "        unit_costs=$UNIT_COSTS fail_k=$FAIL_K"

# --- Preflight -------------------------------------------------------------------
# Fails fast and loudly. Everything below is hours long; a missing dependency should
# surface in the first ten seconds, not after stage 2.
hdr "Preflight"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import importlib, sys
ok = True
for m in ("torch", "numpy", "sklearn", "faiss", "sentence_transformers"):
    try:
        importlib.import_module(m)
        print(f"  {m}: ok")
    except ImportError as e:
        print(f"  {m}: MISSING ({e})"); ok = False
try:
    import torch
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: no GPU visible — retrieval will run on CPU and be far slower.")
except Exception:
    pass
try:
    import spacy; spacy.load("en_core_web_sm")
    print("  spaCy en_core_web_sm: ok")
except Exception as e:
    print(f"  WARNING: spaCy model unavailable ({e}); entity features fall back to zeros.")
sys.exit(0 if ok else 1)
PY

# --- Phase A: the validated pipeline ---------------------------------------------
if [ -z "$SKIP_PIPELINE" ]; then
    hdr "PHASE A — main pipeline (run_on_cluster.sh)"
    log "This is the long one. Everything below it is minutes."
    bash experiments/router_experiments/run_on_cluster.sh 2>&1 | tee -a "$LOG"
else
    log ""
    log "Skipping Phase A (SKIP_PIPELINE=1) — reusing existing results/ and env data."
fi

# --- Did stage 2 actually persist the cross-scores? ------------------------------
# Checked between phases rather than at the end: if the retrieval stage predates the
# cross-scoring change, B2-B4 will silently produce a degraded arm, and it is worth
# knowing that here instead of debugging empty columns afterwards.
hdr "Checking cross-scoring data landed"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import json, os, sys
p = os.path.join("experiments", "baseline_v2", "results", "nq_per_query_results.json")
if not os.path.exists(p):
    print(f"  WARNING: {p} not found — cannot verify."); sys.exit(0)
with open(p, encoding="utf-8") as f:
    d = json.load(f)
recs = d.get("per_query_results", d) if isinstance(d, dict) else d
if isinstance(recs, dict):
    recs = next((v for v in recs.values() if isinstance(v, list)), [])
if not recs:
    print("  WARNING: no per-query records found."); sys.exit(0)
r = recs[0]
has_cs = bool(r.get("retrievers", {}).get("bm25", {}).get("cross_scores"))
has_norm = bool(r.get("cross_score_norm"))
print(f"  cross_scores present:     {has_cs}")
print(f"  cross_score_norm present: {has_norm}")
if not (has_cs and has_norm):
    print("  WARNING: retrieval results predate the cross-scoring change. Re-run stage 2")
    print("           (02_run_baselines.py) or the _xscore arm will be degraded.")
PY

# --- Phase B: new analyses --------------------------------------------------------
hdr "PHASE B — cost-aware, cross-scoring, cascade checks (CPU, minutes)"

# B1 — cheapest go/no-go in the project. Structural unit costs; the BM25-vs-Dense cost gap
# is the load-bearing assumption, so a second sweep records sensitivity to it.
soft "B1 cost-aware degeneracy check" \
    "$PY" experiments/router_experiments/cost_aware_analysis.py \
        --unit-costs "$UNIT_COSTS"

soft "B1b cost-aware sensitivity (equal BM25/Dense cost — expected degenerate by algebra)" \
    "$PY" experiments/router_experiments/cost_aware_analysis.py \
        --unit-costs 1,1,2 \
        --output results/cost_aware_analysis_equalcost.json \
        --output-md results/cost_aware_analysis_equalcost.md

# B2 — separate suffix so Phase A's env data and every number derived from it stay intact.
soft "B2 build _xscore env arm (cross-scoring features)" \
    "$PY" src/router/prepare_env_data.py \
        --cross-scores --suffix _xscore \
        --gold-type "$GOLD_TYPE" \
        --reward "$REWARD" \
        --embed-dims "$EMBED_DIMS" \
        --k "$REWARD_K"

# B3 — decides whether the cascade is worth building at all.
soft "B3 escalation signal check (cascade go/no-go)" \
    "$PY" experiments/router_experiments/escalation_signal_check.py \
        --env-suffix _xscore --k "$FAIL_K"

# B4 — routing with cross-scores, against the same best-fixed/oracle baselines.
soft "B4 per-retriever routing WITH cross-scores" \
    "$PY" experiments/router_experiments/per_retriever_router.py \
        --env-suffix _xscore \
        --output results/per_retriever_router_xscore.json \
        --output-md results/per_retriever_router_xscore.md

# B4b — the same router without cross-scores, so B4's delta is attributable to the new
# features rather than to anything else that changed.
soft "B4b per-retriever routing WITHOUT cross-scores (control)" \
    "$PY" experiments/router_experiments/per_retriever_router.py \
        --output results/per_retriever_router_base.json \
        --output-md results/per_retriever_router_base.md

# B5 — the method itself: run the cheap retriever, inspect what it returned, escalate to
# Hybrid only when the evidence says it failed. Judged against escalating at random at the
# SAME rate, which is what separates targeting from simply spending more.
soft "B5 learned cross-scoring cascade" \
    "$PY" experiments/router_experiments/cascade_router.py \
        --env-suffix _xscore --k "$FAIL_K" --lam "$CASCADE_LAMBDA"

# B5b — cascade without cross-scores. Isolates how much of B5 comes from post-retrieval
# evidence rather than from the cascade structure alone.
soft "B5b cascade WITHOUT cross-scores (control)" \
    "$PY" experiments/router_experiments/cascade_router.py \
        --env-suffix "" --k "$FAIL_K" --lam "$CASCADE_LAMBDA" --no-reinforce \
        --output results/cascade_router_base.json \
        --output-md results/cascade_router_base.md

# --- Summary ----------------------------------------------------------------------
hdr "DONE — $(date)"
log "Full log: $LOG"
log ""
log "Read in this order — each decides whether the next matters:"
log "  1. results/cost_aware_analysis.md          Is there a decision to make once cost counts?"
log "  2. results/escalation_signal.md            Is retriever failure detectable post-hoc?"
log "  3. results/cascade_router.md               THE METHOD. Look at 'vs matched random'."
log "  4. results/cascade_router_base.md          Same without cross-scores (control)"
log "  5. results/per_retriever_router_xscore.md  vs _base.md — did cross-scores help routing?"
log "  6. results/canonical_results.json          Main table (Phase A)"
log ""
if [ ${#FAILED_STAGES[@]} -eq 0 ]; then
    log "All Phase B stages succeeded."
else
    log "Phase B stages that FAILED (Phase A results are unaffected):"
    for s in "${FAILED_STAGES[@]}"; do log "  - $s"; done
fi
log "=================================================================="
