"""
Scale the retrieval corpus from 50 to 500 documents per query.

This re-runs the baseline evaluation pipeline with a larger corpus,
making retriever disagreement more pronounced and results more credible.

Usage:
    cd <repo>
    python experiments/baseline_v2/scale_corpus.py
"""
import os, sys, json, time, random
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# IMPORTANT: You need to find the actual data loading mechanism in baseline_v2.
# Look for scripts that:
#   1. Load NQ/TriviaQA raw data
#   2. Create a corpus of gold + distractor documents per query
#   3. Run BM25/Dense/Hybrid retrieval on that corpus
#   4. Save per-query results to nq_per_query_results.json
#
# The key parameter to change: NUM_DISTRACTORS = 499 (from ~49)
# This gives 500 total docs per query (1 gold + 499 distractors)
#
# After changing NUM_DISTRACTORS and re-running the pipeline:
#   python -m src.router.prepare_env_data
# This regenerates nq_env_data.json and triviaqa_env_data.json
#
# Then re-run ALL experiments (01-12) with the new data.

# STEP 1: Find the baseline script
baseline_dir = os.path.join(PROJECT_ROOT, "experiments", "baseline_v2")
print(f"Looking for baseline scripts in: {baseline_dir}")
if os.path.exists(baseline_dir):
    for f in os.listdir(baseline_dir):
        print(f"  {f}")
else:
    print("baseline_v2 directory not found! Check project structure.")
    print("Look for NUM_DISTRACTORS, num_distractors, or similar in:")
    print("  - run_experiment.py")
    print("  - run_pipeline.py")
    print("  - Any file in experiments/")
