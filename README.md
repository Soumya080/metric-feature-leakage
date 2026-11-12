# Metric-as-Feature Leakage in Retriever Routing

Code, data and the `leakcheck` audit tool for **"Metric-as-Feature Leakage: A Feature-Side
Failure Mode in Retriever Routing"** — Soumy Sawant and Chirag Joshi (Manipal University
Jaipur). NewInML Workshop at NeurIPS 2026, Paris.

**Start here: [`REVIEWERS.md`](REVIEWERS.md)** — reproduces the headline numbers in three
commands, under two minutes, no GPU and no downloads.

**Check your own router:** `leakcheck.audit(X, R)` on your feature matrix and per-arm reward
table — see [`leakcheck/README.md`](leakcheck/README.md).

## What this work reports

We built a router that selects between BM25, Dense and Hybrid retrieval per query, trained
with REINFORCE, and it beat every fixed retriever. Four of its sixty-one features turned out
to have been computed from the evaluation metric, which for two of the three arms *is* the
reward.

Rather than repair that silently, the paper measures it. Five feature sets differ only in how
much of the evaluation metric reaches the features, with the reward table held byte-identical.
Accuracy at predicting the best retriever rises from **59.9%** with label-free features to
**73.1%** with the exact leak, against a **60.9%** majority baseline, and a noise-perturbed
leak keeps part of that advantage while matching no reward value (about half under additive
noise of ±0.15) — so an equality-based audit does not catch it.

With the leak removed, no selection method beats always using the dense retriever (best point
estimate of capture **+0.4%**, 95% CI 0.0–1.5%). That negative is bounded by the test splits'
minimum detectable capture, **68% / 43% / 72%** of headroom (NQ / TriviaQA / HotpotQA,
one-sided, 80% power): the claim is that no method we tested reaches that, not that no useful
router can exist. Concurrent work reports retriever routing succeeding at larger scale with
post-retrieval supervision, and the paper says so.

**The earlier framing of this repository — a working contextual-bandit router that adapts
online — is what the leak produced.** The paper retracts it, `REPRODUCE.md` marks every
superseded number row by row, and `results/KNOWN_ISSUES.md` records the artifact defects an
audit found rather than silently correcting them. The committed `results/` were regenerated
after the fix; `paper/main_v2.tex` is the camera-ready paper.

## What we release

- **`leakcheck/`** — a standalone, numpy-only detector over `(feature matrix, per-arm reward
  table)`. It issues automatic verdicts only on exact agreement and reports rank findings as
  suspicions carrying their evidence. `leakcheck/CALIBRATION.md` records the external audit
  that falsified its first design. It is triage, not a certificate of cleanliness.
- **`examples/leak_demo.py`** — reproduces the dose–response mechanism from synthetic data in
  one command, with no corpus and no downloads.
- **`results/`** — every committed artifact behind every number in the paper, with a
  `_provenance` block in each file.
- **`REPRODUCE.md`** — maps each claim to the command that regenerates it.
- **`results/KNOWN_ISSUES.md`** — defects found by audit in our own artifacts, recorded rather
  than quietly corrected.

## Scope and caveats

- Experiments use a **500-passage candidate set per query** (1 gold + 499 hard negatives), not
  a live full-Wikipedia index. Section 4's scale study reaches a 100K shared index. These are
  two different settings and the paper states which claims rest on which.
- Reward is the reciprocal rank of the first **answer-bearing** passage. That is a proxy for
  answer quality, and 27–46% of queries contain answer-bearing distractors.
- Roughly half of NQ's gold passages are template-constructed (`gold_type: constructed`) where
  no real Wikipedia snippet containing the answer was found. The paper's headline numbers use
  `real_wikipedia` gold only; `canonical_results_all.json` covers all gold types and is a
  different experiment with different means.
- The Wikipedia passage pool is **not redistributed** — it is CC BY-SA text and regenerates
  deterministically at a fixed seed. See `DATA_LICENSES.md`.

## Quick Start (Reproduce the Main Results Table)

All intermediate data required to reproduce the paper's tables and figures is **already
included in this repository** (see [Data Availability](#data-availability) below) — you do
**not** need a GPU, an internet connection, or to re-run retrieval to reproduce Table 1.

```bash
git clone <anonymized-repo-url>
cd efficient-rag-pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# One command reproduces everything downstream of the committed per-query data:
python experiments/router_experiments/run_all_gpu_lab.py
```

This runs, in order:

1. `canonical_evaluation.py` — static/heuristic/supervised/RL baselines + bootstrap significance tests → `results/canonical_results.json`
2. `online_adaptation.py` — the NQ→TriviaQA→HotpotQA distribution-shift experiment → `results/online_adaptation_results.json`
3. `generate_latex_tables.py` — renders `results/canonical_results.json` → `results/main_results_table.tex` (Table 1 in the paper)
4. `plot_online_adaptation.py` — renders `results/online_adaptation_results.json` → `results/online_adaptation_plot.pdf` (Figure 1 in the paper)

Expect ~15–30 minutes on CPU (the RL training loop operates over pre-computed per-query
lookup tables, not live retrieval, so no GPU is required for this part).

To run each stage individually instead, see [Step-by-Step Reproduction](#step-by-step-reproduction).

## Data Availability

Because full retrieval (BM25 + FAISS dense + RRF hybrid, ~2,000 queries × 3 datasets ×
500-passage pools) is expensive to re-run, we commit the intermediate artifacts directly:

| File                                                                          | Contents                                                                          |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `experiments/baseline_v2/results/{nq,triviaqa,hotpotqa}_per_query_results.json` | Raw per-query, per-retriever Hit/MRR/F1/EM at k ∈ {1,3,5,10,20}                     |
| `src/router/{nq,triviaqa,hotpotqa}_env_data.json`                             | Feature vectors + per-retriever rewards, ready to load into `RouterEnvironment`   |
| `experiments/router_experiments/results/*.json`                              | All baseline, ablation, and significance-test outputs backing the paper's tables |

If you want to regenerate these from scratch (e.g., to test with a different embedding
model or corpus size), see [Full Pipeline (From Raw Data)](#full-pipeline-from-raw-data).

## Step-by-Step Reproduction

```bash
# Main results table (Table 1): static, heuristic, supervised, and RL baselines,
# 15 seeds, paired bootstrap significance vs. Gradient Boosting.
python experiments/router_experiments/canonical_evaluation.py
python experiments/router_experiments/generate_latex_tables.py

# Online adaptation figure (Figure 1): NQ -> TriviaQA -> HotpotQA distribution shift.
python experiments/router_experiments/online_adaptation.py
python experiments/router_experiments/plot_online_adaptation.py
```

Supplementary analyses referenced in the paper:

```bash
# Feature ablation (Section 4.3, "Lean Router"): all-features vs. retrieval-only vs. no-retrieval.
python experiments/router_experiments/23_multi_seed_ablation.py

# Algorithm comparison (REINFORCE vs. PPO vs. GRPO), Section 4.2.
python experiments/router_experiments/14_ppo_vs_reinforce.py
python experiments/router_experiments/15_grpo_comparison.py

# Cross-dataset transfer (Section 4.4).
python experiments/router_experiments/07_cross_dataset.py

# Latency benchmark (Section 4.5, 0.092ms figure).
python experiments/router_experiments/17_latency_analysis.py

# Qualitative examples: specific queries where RL-Route routes correctly and tuned GBT
# doesn't (and vice versa) — strengthens the Results discussion beyond aggregate MRR.
python experiments/router_experiments/failure_mode_analysis.py --dataset nq

# Turns online_adaptation_results.json's segment averages into paper-ready prose + a
# LaTeX table (right now that effect only appears in Figure 1, which requires the reader
# to eyeball a plot).
python experiments/router_experiments/summarize_online_adaptation.py

# Additional single-split baselines used during development (not in the main table):
python experiments/router_experiments/21_fixed_split_multi_seed.py
python experiments/router_experiments/22_regression_baseline.py
```

## Full Pipeline (From Raw Data)

**Recommended for a GPU cluster / lab machine: one script, calibrated first.** The
previously-committed `experiments/baseline_v2/data/*_prepared.json` files were found to be
stale (wrong question count/corpus size, and `hotpotqa_prepared.json` was missing entirely)
and did not match the committed evaluation results — so a from-scratch regeneration is the
reliable path right now, not an optional extra.

```bash
bash experiments/router_experiments/run_on_cluster.sh
```

This does three things in order: (1) checks the environment (PyTorch/CUDA), (2) runs
`estimate_pipeline_time.py` — a **read-only** calibration that times retrieval and RL
training on synthetic/real-but-unmodified data and extrapolates a real wall-clock estimate
for *this machine* (on the machine this pipeline was developed on, that came out to
**~7-12 hours total**, dominated by retrieval (~1h40m) and the feature ablation stage
(~4-5h, the single largest cost) — your hardware will differ, that's the point of running
it), then (3) backs up whatever's currently in `results/` (nothing here is in git history,
so this backup is the only way to compare pre-fix vs. post-fix numbers afterward) and runs
`run_full_pipeline.py`, with output tee'd to a timestamped log so you can safely disconnect:

```bash
nohup bash experiments/router_experiments/run_on_cluster.sh > /dev/null 2>&1 &
tail -f logs/pipeline_*.log
```

Useful environment variable overrides: `SKIP_CALIBRATION=1` (skip straight to the full
run), `SKIP_ABLATION=1` (skip the slowest single stage, ~4-5hrs, not required for Table 1),
`SKIP_FAILURE_MODE=1`, `FORCE=1` (regenerate every stage regardless of what already exists).

To run just the calibration on its own (safe, ~1-2 min, touches no real files):

```bash
python experiments/router_experiments/estimate_pipeline_time.py
```

Or run the pipeline directly without the shell wrapper — `run_full_pipeline.py` is
resumable on its own (each stage is skipped if its output file already exists):

```bash
python experiments/router_experiments/run_full_pipeline.py
python experiments/router_experiments/run_full_pipeline.py --skip-ablation --skip-failure-mode  # faster
python experiments/router_experiments/run_full_pipeline.py --force                              # ignore existing outputs
```

**Hardware note:** despite the name of the sibling `run_all_gpu_lab.py` script, nothing in
this pipeline actually requires a GPU. `all-MiniLM-L6-v2` is an ~80MB embedding model that
runs fast on CPU; FAISS here is `faiss-cpu`; BM25 is pure CPU; RL training is a tiny 3-layer
MLP over pre-computed lookup tables. A GPU helps embedding throughput somewhat but a modest
laptop GPU (or none at all) is enough — a 2-query smoke test of the retrieval stage
completed in under a second on CPU during development of this pipeline.

Equivalent manual steps, if you'd rather run stage-by-stage:

```bash
# 1. Prepare NQ/TriviaQA/HotpotQA questions + 500-passage pools (1 gold + 499 hard negatives).
python experiments/baseline_v2/01_prepare_data.py

# 2. Run BM25 / Dense (FAISS) / Hybrid (RRF) retrieval + compute Hit/MRR/F1/EM per query.
#    (Also persists raw, label-free retrieval scores and uses word-boundary answer matching.)
python experiments/baseline_v2/02_run_baselines.py --dataset all

# 3. Extract routing features (from raw retrieval scores, not MRR) and assemble the
#    RL environment lookup tables.
python -m src.router.prepare_env_data

# 4. Proceed to Quick Start / Step-by-Step Reproduction above.
```

Step 2 requires downloading `all-MiniLM-L6-v2` (sentence-transformers, cached after first
use) and is the slowest stage; steps 3-4 are fast (CPU, no downloads, lookup-table-based).

## Directory Structure

```text
src/
  router/              Core RL environment, feature extractor, policy network, trainers
    routing_env.py        RouterEnvironment + train/val/test split utilities
    feature_extractor.py  26-dim query feature extraction (lexical/entity/structural/retrieval)
    policy_network.py     3-layer MLP policy + REINFORCE action selection
    train_reinforce.py    REINFORCE training loop with EMA baseline + entropy bonus
    train_ppo.py / train_grpo.py   Alternative policy-gradient optimizers (Section 4.2)
  retrieval/           BM25, dense (FAISS), and hybrid (RRF) retrievers
  evaluation/           Hit/MRR/F1/EM metrics, oracle analysis
experiments/
  baseline_v2/         Data prep + BM25/Dense/Hybrid evaluation on real Wikipedia passages
  router_experiments/  All routing experiments; canonical_evaluation.py is the master script
paper/                 LaTeX source (main_v2.tex, references.bib) and figures
results/               Generated tables/figures consumed by paper/main_v2.tex
```

## Requirements

See `requirements.txt`. Install via:

```bash
pip install -r requirements.txt
```

`vllm` and `transformers`/`accelerate` are only required for the optional end-to-end
generation benchmark (`18_end_to_end_eval.py`) and are not needed to reproduce Table 1
or Figure 1.

## Citation

```bibtex
@inproceedings{metricleak2026,
  title={Metric-as-Feature Leakage: A Feature-Side Failure Mode in Retriever Routing},
  author={Anonymous},
  booktitle={NewInML Workshop at NeurIPS},
  year={2026}
}
```

This repository is anonymized for double-blind review. Author information will be added
upon acceptance.
