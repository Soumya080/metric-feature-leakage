#!/bin/bash
# run_on_cluster.sh - End-to-end entry point for the GPU cluster / lab machine.
#
# Does three things, in order:
#   1. Sanity-checks the environment (venv active, deps installed).
#   2. Runs estimate_pipeline_time.py (read-only, ~1-2 min) so you see a real,
#      machine-measured time estimate BEFORE the expensive run starts.
#   3. Runs the full pipeline (run_full_pipeline.py), tee'd to a timestamped log file,
#      so you can safely disconnect (via nohup/tmux/screen) and check progress later.
#
# Usage:
#   cd efficient-rag-pipeline
#   bash experiments/router_experiments/run_on_cluster.sh
#
#   # Run in the background, survives SSH disconnect:
#   nohup bash experiments/router_experiments/run_on_cluster.sh > /dev/null 2>&1 &
#   tail -f logs/pipeline_*.log        # check progress later
#
#   # Skip the slowest stage (~4-5hrs feature ablation) if you just need Table 1 fast:
#   SKIP_ABLATION=1 bash experiments/router_experiments/run_on_cluster.sh
#
#   # Skip the calibration step and go straight to the full run (if you already know the timing):
#   SKIP_CALIBRATION=1 bash experiments/router_experiments/run_on_cluster.sh
#
#   # Headline results use only queries with a REAL Wikipedia gold passage (default).
#   # 01_prepare_data.py falls back to a generated template when the pool has no passage
#   # containing the answer; that template repeats the answer but shares no terms with the
#   # question, so BM25 fails on it by construction. Measured on the pre-fix data, dropping
#   # those queries RAISES oracle routing headroom (NQ 13%->32%, TriviaQA 8%->35%,
#   # HotpotQA 10%->67%), because a fake guaranteed-Dense-win leaves nothing to route.
#   # The unfiltered variant is still evaluated as an appendix robustness check.
#   GOLD_TYPE=all bash experiments/router_experiments/run_on_cluster.sh   # unfiltered headline
#
#   # Other overrides: SKIP_FAILURE_MODE=1  SKIP_SUPPLEMENTARY=1  SKIP_BACKUP=1  FORCE=1
#                      SKIP_APPENDIX=1  GOLD_TYPE=real_wikipedia|all|constructed

set -e  # exit on first error — a failed stage should stop the script, not silently continue

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/pipeline_${TIMESTAMP}.log"

echo "=================================================================="
echo "RL-Route: GPU cluster end-to-end run"
echo "Repo root: $REPO_ROOT"
echo "Log file:  $LOG_FILE"
echo "=================================================================="

# --- 1. Environment sanity check ---
echo ""
echo "--- Environment check ---"
python3 -c "
import sys
print('Python:', sys.version.split()[0])
try:
    import torch
    print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
except ImportError:
    print('ERROR: PyTorch not installed. Run: pip install -r requirements.txt')
    sys.exit(1)
"
if [ $? -ne 0 ]; then
    echo "Environment check failed — install dependencies first: pip install -r requirements.txt"
    exit 1
fi

# --- Back up the current (pre-fix) results before overwriting them ---
# NOTE: results/*.json is untracked in git (never committed), so there is no
# git-committed "old" version to diff against later — this backup is the only way
# to compare pre-fix vs. post-fix numbers afterward.
if [ -d "results" ] && [ -z "$SKIP_BACKUP" ]; then
    BACKUP_DIR="results_backup_${TIMESTAMP}"
    echo ""
    echo "--- Backing up current results/ to $BACKUP_DIR (pre-fix numbers, for later comparison) ---"
    cp -r results "$BACKUP_DIR"
    echo "Backed up. Compare later with e.g.:"
    echo "  diff <(python3 -m json.tool $BACKUP_DIR/canonical_results.json) <(python3 -m json.tool results/canonical_results.json)"
fi

# --- 2. Timing calibration (read-only, safe, ~1-2 min) ---
if [ -z "$SKIP_CALIBRATION" ]; then
    echo ""
    echo "--- Timing calibration (read-only, does not touch any real data) ---"
    python3 experiments/router_experiments/estimate_pipeline_time.py 2>&1 | tee -a "$LOG_FILE"

    echo ""
    echo "=================================================================="
    echo "Calibration done. Review the estimate above."
    echo "Starting the full pipeline in 15 seconds — Ctrl+C now to abort."
    echo "(Set SKIP_CALIBRATION=1 to skip this pause next time.)"
    echo "=================================================================="
    sleep 15
else
    echo "Skipping calibration (SKIP_CALIBRATION=1)."
fi

# --- 3. Full pipeline ---
PIPELINE_ARGS=""
if [ -n "$SKIP_ABLATION" ]; then
    PIPELINE_ARGS="$PIPELINE_ARGS --skip-ablation"
fi
if [ -n "$SKIP_FAILURE_MODE" ]; then
    PIPELINE_ARGS="$PIPELINE_ARGS --skip-failure-mode"
fi
if [ -n "$FORCE" ]; then
    PIPELINE_ARGS="$PIPELINE_ARGS --force"
fi
if [ -n "$SKIP_APPENDIX" ]; then
    PIPELINE_ARGS="$PIPELINE_ARGS --skip-appendix"
fi
PIPELINE_ARGS="$PIPELINE_ARGS --gold-type ${GOLD_TYPE:-real_wikipedia}"

echo ""
echo "--- Running full pipeline (args: ${PIPELINE_ARGS:-none}) ---"
python3 experiments/router_experiments/run_full_pipeline.py $PIPELINE_ARGS 2>&1 | tee -a "$LOG_FILE"

# --- 4. Supplementary experiments backing paper sections 4.2 / 4.4 / 4.5 ---
# These are NOT part of run_full_pipeline.py (that one produces Table 1 + Figure 1), but
# they back specific paper sections and were all affected by the test-set-leak and
# per-seed-split bugs fixed in this audit (see BUGS.md P1-1/P1-2/P1-3/P1-6). They must be
# re-run for those sections' numbers to be valid.
if [ -z "$SKIP_SUPPLEMENTARY" ]; then
    echo ""
    echo "--- Supplementary re-runs (paper sections 4.2 / 4.4 / 4.5) ---"
    for script in \
        "15_grpo_comparison.py:section 4.2 algorithm comparison (fixed split + val)" \
        "07_cross_dataset.py:section 4.4 cross-dataset transfer (source-set val)" \
        "17_latency_analysis.py:section 4.5 latency (measured, not hardcoded)"
    do
        name="${script%%:*}"; desc="${script##*:}"
        echo ""
        echo ">>> $name — $desc"
        python3 "experiments/router_experiments/$name" 2>&1 | tee -a "$LOG_FILE" || {
            echo "WARNING: $name failed; continuing (it does not block Table 1/Figure 1)." | tee -a "$LOG_FILE"
        }
    done
else
    echo "Skipping supplementary experiments (SKIP_SUPPLEMENTARY=1)."
fi

echo ""
echo "=================================================================="
echo "DONE. Full log saved to: $LOG_FILE"
echo "Next steps:"
if [ -n "$BACKUP_DIR" ]; then
echo "  1. Compare new vs. pre-fix numbers:"
echo "       diff <(python3 -m json.tool $BACKUP_DIR/canonical_results.json) <(python3 -m json.tool results/canonical_results.json)"
fi
echo "  2. Review results/online_adaptation_summary.md for the online-adaptation prose/table."
echo "  3. Review experiments/router_experiments/results/failure_mode_analysis_*.json for qualitative examples."
echo "=================================================================="
