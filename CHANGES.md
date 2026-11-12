# Repository Structure & Change Log

This document has two parts: (1) the full tracked file structure of the repository as of
this pass, and (2) a before/after log of every code change made during the bug-fix pass
(feature leakage, substring-matching, val/test-split leak, seed count, GBT tuning depth,
citation metadata, dependency pinning, stray-file cleanup).

## 1. Repository Structure

```text
efficient-rag-pipeline/
├── CLAUDE.md                          Guidance for future Claude Code sessions in this repo
├── CHANGES.md                         This file
├── README.md                          Reproduction instructions, data availability, citation
├── requirements.txt                   Pinned dependencies (torch/sklearn/faiss-cpu/... )
├── run_all_experiments.sh / .ps1      Legacy "run everything" scripts (baseline_v1 era)
├── run_gpu_lab.ps1, run_remaining*.ps1  Legacy Windows launch scripts
├── verify_codebase.py, verify_corpus.py, check_sizes.py, bench_gpu.py   Ad hoc dev utilities
│
├── data/knowledge_base/*.txt          Small static text corpus used by the original
│                                       (non-router) RAG pipeline demo (src/pipeline.py)
│
├── dump/  (gitignored, not deleted)   Stray files pulled out of src/router/ this session:
│                                       Untitled-1.txt (broken VS Code scratch file, unrelated
│                                       CartPole actor-critic experiment), router_dataset.csv
│                                       (leftover demo artifact from feature_extractor.py's
│                                       __main__ block)
│
├── src/
│   ├── pipeline.py                    Original (non-router) end-to-end RAG pipeline
│   ├── chunking/                      naive / semantic / sentence chunking strategies
│   ├── embeddings/encoder.py          Sentence-transformers wrapper
│   ├── retrieval/                     sparse.py (BM25), dense.py, dense_faiss.py, hybrid.py (RRF)
│   ├── generation/generator.py        LLM answer generation (used by 18_end_to_end_eval.py)
│   ├── evaluation/
│   │   ├── metrics.py                 Hit/MRR/Recall/F1/EM with word-boundary answer matching
│   │   ├── benchmark_datasets.py      Dataset loading utilities (used by tests/)
│   │   └── oracle_analysis.py         Oracle/winner analysis for the original pipeline
│   └── router/                        *** The core RL routing system ***
│       ├── routing_env.py               RouterEnvironment + train/val/test split utilities
│       ├── feature_extractor.py         Canonical 26-dim feature order (spaCy-based, full)
│       ├── prepare_env_data.py          Fast regex-based feature extraction (used in practice)
│       ├── policy_network.py            RouterPolicy (3-layer MLP) + PolicyAgent
│       ├── train_reinforce.py           ReinforceTrainer — the real training loop
│       ├── train_ppo.py, train_grpo.py  Alternative optimizers (Section 4.2 comparison)
│       ├── compute_reward.py            Reward/cost-penalty helpers
│       └── {nq,triviaqa,hotpotqa}_env_data.json   Committed RL environment lookup tables
│
├── experiments/
│   ├── baseline_v2/                   Current data generation + retrieval pipeline
│   │   ├── 01_prepare_data.py           Build QA + 500-passage corpora (real Wikipedia)
│   │   ├── 02_run_baselines.py          Run BM25/Dense/Hybrid, compute + persist metrics
│   │   ├── 03_analyze_oracle.py         Oracle-vs-best-fixed gap analysis
│   │   ├── scale_corpus.py
│   │   ├── cache/wiki_passage_pool.json Cached Wikipedia passage pool (20K passages)
│   │   ├── data/*_prepared.json         Prepared QA + corpora (⚠ stale, see Change Log §2.7)
│   │   └── results/*_per_query_results.json, oracle_analysis.json, figures/
│   │
│   └── router_experiments/
│       ├── 01_*.py – 20_*.py           Exploratory/superseded scripts from earlier iterations
│       ├── canonical_evaluation.py     *** Master evaluation script — Table 1 source of truth ***
│       ├── generate_latex_tables.py    canonical_results.json -> main_results_table.tex
│       ├── online_adaptation.py        NQ->TriviaQA->HotpotQA distribution-shift experiment
│       ├── plot_online_adaptation.py   -> online_adaptation_plot.pdf (Figure 1)
│       ├── 21_fixed_split_multi_seed.py, 22_regression_baseline.py, 23_multi_seed_ablation.py
│       │                              Supplementary multi-seed baselines / feature ablation
│       ├── run_all_gpu_lab.py          Runs canonical_evaluation + online_adaptation + table/plot gen
│       ├── run_full_pipeline.py        *** NEW: full pipeline from raw data, resumable ***
│       └── results/*.json, figures/*.png
│
├── paper/
│   ├── main.tex                       LaTeX manuscript
│   ├── references.bib                 Bibliography (8 fabricated-metadata entries fixed, see §2.6)
│   ├── main_results_table.tex          Generated Table 1 (⚠ reflects pre-fix numbers, see §2.7)
│   └── online_adaptation_plot.pdf      Generated Figure 1
│
├── results/                            Top-level generated outputs (mirrors paper/ artifacts)
├── showcase/                           Standalone demo web page (index.html, app.js, styles.css)
├── tests/test_prepared_datasets.py     Single pytest smoke test
└── archive/  (gitignored)              Old scripts/exploratory work from before baseline_v2
```

## 2. Change Log (this session)

### 2.1 Feature leakage fix — `experiments/baseline_v2/02_run_baselines.py`, `src/router/prepare_env_data.py`

**Before:** the router's `avg_bm25`/`avg_dense` features were built from each retriever's
MRR (`bm25_k_metrics.get("mrr", 0.0)`) — but MRR requires knowing the gold answer's rank,
and the reward for choosing action 0 (BM25) / action 1 (Dense) *is* that retriever's MRR
(`src/router/routing_env.py::step()`). Verified bit-for-bit against `nq_env_data.json`:
`feature[22] == retrievers.bm25.mrr` and `feature[23] == retrievers.dense.mrr`, exactly, on
every query checked.

**After:** `02_run_baselines.py` now computes and persists raw, label-free retrieval scores
per query per retriever — `top1_score`, `mean_score_top5`, `std_score_top5` — computed
directly from BM25/FAISS similarity scores, before any gold-answer matching happens.
`prepare_env_data.py` now builds `avg_bm25`/`avg_dense`/`score_gap`/`bm25_variance` from
those raw scores. Verified via smoke test: `feature[22] != retrievers.bm25.mrr` post-fix.

**Action required:** requires re-running `02_run_baselines.py` (and everything downstream)
to regenerate real numbers — see §2.7.

### 2.2 Substring-matching bug — `experiments/baseline_v2/02_run_baselines.py`

**Before:** `check_hit`/`compute_mrr`/`substring_match_score` used naive
`if normalize_answer(ans) in content_norm`, a bare substring check with no word boundaries
— e.g. gold answer "may" would match inside "maybe", "china" inside "chinatown".

**After:** added `_word_boundary_contains()` (mirrors the word-boundary regex already used
correctly in `src/evaluation/metrics.py::_answer_matches_context`) and routed all three
functions through it.

### 2.3 Train/val/test leak — `experiments/router_experiments/21_fixed_split_multi_seed.py`, `23_multi_seed_ablation.py`

**Before:** both scripts used `create_train_test_split` (train/test only) and passed no
`env_val` to `ReinforceTrainer`, so best-checkpoint selection silently fell back to
test-set performance (`train_reinforce.py`'s documented fallback behavior).
`canonical_evaluation.py` already did this correctly.

**After:** both scripts now use `create_train_val_test_split` (or an equivalent
train/val/test key split for the masked-feature-ablation case in #23) and pass
`env_val=env_val` to `ReinforceTrainer`, matching `canonical_evaluation.py`'s methodology.

### 2.4 Seeds 5 → 15, GBT tuning depth — `experiments/router_experiments/canonical_evaluation.py`

**Before:** `SEEDS` had 5 entries, but the auto-generated table caption
(`generate_latex_tables.py`) already hardcoded "15 seeds" (a stale mismatch). Gradient
Boosting tuning used `RandomizedSearchCV(..., n_iter=5, cv=3)` with a narrower param grid
than `04_classifier_baseline.py`'s `n_iter=20` + `min_samples_leaf`.

**After:** `SEEDS` now has 15 entries (caption is now accurate). GBT tuning aligned to
`n_iter=20` with the same `min_samples_leaf` param, matching `04_classifier_baseline.py`.

### 2.5 Dead/misleading code — `src/router/policy_network.py`

**Before:** `PolicyAgent.update_policy()` implements a bare single-sample REINFORCE update
(no baseline, no entropy, no clipping) with no indication that it's not what any real
experiment uses — a reader could reasonably mistake it for "the" update rule.

**After:** kept (still used by the file's own `__main__` demo) but documented with an
explicit docstring pointing to `train_reinforce.py::ReinforceTrainer._train_epoch` as the
actual training logic.

### 2.6 Fabricated citation metadata — `paper/references.bib`, `paper/main.tex`

Verified against arXiv/AAAI/ACL Anthology directly (WebFetch on each paper's abstract
page). **8 of ~15 entries had wrong author lists** (several also wrong venue/year):

| Key | Was cited as | Actually |
|---|---|---|
| `zhao2026r3ag` | Zhao, Xin and others | Zhao, Tong; Zhu, Yutao; Tian, Yucheng; Dou, Zhicheng (ACL 2026) |
| `campos2024routerretriever` | Campos, Daniel and others (arXiv 2024) | Lee, Hyunji; Soldaini, Luca; Cohan, Arman; Seo, Minjoon; Lo, Kyle (AAAI 2025) |
| `qin2024ltrr` | Qin, Yujia and others (arXiv 2024) | Kim, To Eun; Diaz, Fernando (SIGIR 2026) |
| `qrag2026` | Chen, Yuxuan; Wang, Haoyu; Li, Ming | Sorokin, Artyom et al. (10 authors) (ICLR 2026 Oral) |
| `r32025` | Kim, Jihoon; Park, Seojin; Lee, Donghyun (NewInML workshop) | Zhou, Jiawei; Chen, Lei (NeurIPS 2025 main track) |
| `r3rag2025` | Gupta, Rohan; Singh, Ananya; Kumar, Rahul (ACL) | Li, Yuan et al. (10 authors) (EMNLP 2025 Findings) |
| `sun2025dynamicrag` | Sun, Yifan and others | Sun, Jiashuo; Zhong, Xianrui; Zhou, Sizhe; Han, Jiawei (title also corrected) |

All fixed with verified author lists, venues, and years, plus an arXiv ID `note={}` for
traceability. Citation *keys* were deliberately left unchanged (e.g.
`campos2024routerretriever` still refers to the Lee et al. paper) to avoid missing a
`\cite{}` update somewhere in `main.tex` — this is cosmetic only (key name vs. content are
independent in BibTeX) but worth a rename pass later if time allows.

Also corrected two related-work sentences in `main.tex` whose *technical* claims didn't
match the real papers' abstracts (R³AG's "eight retrievers / disentangled dual encoders /
80B LLM judge" claim, and Q-RAG's description) — both are marked with
`%% TODO(verify)` comments since the corrections are based on abstracts only, not a full
read of either paper. **Read both papers before final submission and remove the TODOs.**

### 2.7 Dependency pinning — `requirements.txt`

Pinned every dependency to the version actually verified installed and working in this
environment (`torch==2.11.0`, `scikit-learn==1.7.2`, `numpy==2.1.3`, etc.) instead of
unpinned. Moved `vllm`/`accelerate`/`transformers` to a clearly-labeled optional block (only
needed for `18_end_to_end_eval.py`).

### 2.8 ⚠ Known gap, not yet resolved: stale prepared data

`experiments/baseline_v2/data/nq_prepared.json` / `triviaqa_prepared.json` on disk have
only 496-500 entries with 50-passage corpora (not 500), and `hotpotqa_prepared.json` is
missing entirely — none of which matches the committed
`experiments/baseline_v2/results/*_per_query_results.json` (1989-2000 entries, 500-passage
corpora) that the paper's current numbers were actually computed from. This means:

- The fixes in §2.1-2.4 are code-correct and smoke-tested (verified end-to-end on a
  2-query sample), but **have not yet been applied to a full re-run** — the committed
  `results/canonical_results.json` and `paper/main_results_table.tex` still reflect
  **pre-fix** numbers.
- `experiments/router_experiments/run_full_pipeline.py` (new, this session) regenerates
  everything from scratch with the fixes applied — this is what needs to be run next, on
  hardware with more headroom than was used for this fix-and-verify pass. See `README.md`
  for usage; it is resumable and does not require a GPU.
- **Also found in passing:** ~56% of NQ's "gold passages" have `gold_type: "constructed"`
  (template-generated filler text, not real Wikipedia content — see `construct_gold()` in
  `01_prepare_data.py`). Retrievers behave measurably differently on `real_wikipedia` vs.
  `constructed` gold (BM25 does worse on constructed, Dense does better). Worth a line in
  the paper's Limitations section regardless of the leakage fix's outcome.

### 2.9 Incident during this session (self-reported)

A smoke test of the fixed `02_run_baselines.py` (`--dataset nq --max_questions 2`) wrote
through the script's normal output path and overwrote the committed 1989-query
`nq_per_query_results.json` with the 2-query test output. This was caught immediately via
`git status` and reverted with `git checkout -- <file>`; the restored file was verified to
have all 1989 entries again. No data was permanently lost, but flagging this for
transparency — future smoke tests of this script should redirect output to a scratch path
rather than running against the real results directory.

## 3. Second Pass: Cluster Tooling and Additional Fixes

Follow-up pass adding CLI-driven quick-test support, a read-only timing calibration tool, a
GPU-cluster entry point, and three paper-strengthening add-ons requested after the first
pass. Also found and fixed one more real bug (§3.7) while validating the new tooling.

### 3.1 CLI overrides for quick-scale runs

Added `argparse`-based overrides so any script can be run at a reduced scale for timing
calibration or smoke-testing, without touching hardcoded constants:

- `01_prepare_data.py`: `--num_questions`, `--num_distractors`.
- `canonical_evaluation.py`: `--seeds` (comma list or a count N to take the first N of the
  default 15), `--epochs`, `--datasets` (subset of nq/triviaqa/hotpotqa), `--output` (so
  test runs never overwrite `results/canonical_results.json`).
- `23_multi_seed_ablation.py`: same `--seeds`/`--epochs`/`--output` pattern. Also bumped its
  default seed count from 5 to 15 to match `canonical_evaluation.py` (avoids a "why does
  the ablation table use fewer seeds than Table 1?" question).

### 3.2 RL-Route vs. Best-Fixed significance test — `canonical_evaluation.py`

**Before:** only RL-Route vs. tuned Gradient Boosting was bootstrap-tested; the core claim
"RL-Route beats Best-Fixed" was reported as a point estimate with no significance test.

**After:** added `best_fixed_per_query_scores()` and a second `paired_bootstrap_test()` call
against whichever of BM25/Dense/Hybrid has the best mean on the test set. Output now
includes `significance_vs_best_fixed: {best_fixed_method, delta, p_value}` per dataset.
Smoke-tested end-to-end (1 seed, 2 epochs, real NQ data): field populates correctly.

### 3.3 `failure_mode_analysis.py` (new)

Qualitative examples where RL-Route routes correctly and tuned GBT doesn't (and vice
versa) — query text, each retriever's raw reward, both methods' decisions. Reuses
`canonical_evaluation.py`'s exact GBT tuning and train/val/test split so the comparison is
apples-to-apples with Table 1. Smoke-tested end-to-end (3 epochs, real NQ data) — ran
correctly, though 3 epochs is far too few to reflect real routing quality (that's expected;
it only validates the plumbing).

### 3.4 `summarize_online_adaptation.py` (new)

Turns `online_adaptation_results.json`'s `segment_averages` into a markdown table, a LaTeX
table, and a suggested prose paragraph — the online-adaptation effect size previously only
appeared inside Figure 1.

### 3.5 `estimate_pipeline_time.py` (new) — read-only timing calibration

Measures real retrieval throughput (on **synthetic, in-memory** queries/passages — never
touches `experiments/baseline_v2/data/` or `results/`) and real RL training throughput (by
loading an **existing** `*_env_data.json` read-only and training a throwaway agent for a
few epochs), then extrapolates a wall-clock estimate for the full pipeline. Built specifically
to avoid repeating the mistake in §2.9 — no path it touches can collide with real pipeline
output. Run for real during this session (not just written): on the development machine,
retrieval measured at ~1.03s/query (500-passage corpus) and RL training at ~1.39s/epoch
(NQ, 1392 train queries), extrapolating to:

| Stage | Estimate |
|---|---|
| 2. Retrieval (6000 queries × 500-passage corpora) | ~1h 43m |
| 4. `canonical_evaluation.py` (15 seeds × 150 epochs × 3 datasets) | ~2h 35m |
| 8. `23_multi_seed_ablation.py` (15 seeds × 200 epochs × 4 configs) | ~4h 37m — the single largest cost |
| **Total (stages 1+2+4+8)** | **~7h 14m – 11h 52m** |

Your hardware will differ — re-run the script itself to get a machine-specific number.

### 3.6 `run_on_cluster.sh` (new) — GPU cluster entry point

Environment check → calibration (with a 15s abort window) → backs up the current
`results/` directory (necessary because nothing under `results/` is in git history — see
§3.7 — so this backup is the *only* way to diff pre-fix vs. post-fix numbers afterward) →
runs `run_full_pipeline.py`, tee'd to a timestamped log file. Supports `nohup`/background
use and `SKIP_CALIBRATION`/`SKIP_ABLATION`/`SKIP_FAILURE_MODE`/`FORCE` env var overrides.

### 3.7 Bug found while validating the new tooling: `online_adaptation.py` reward reporting

**Before:** the online RL agent's action for a given query was sampled *stochastically*
(`evaluate=False`, required to get a `log_prob` for the gradient update), and that same
noisy sample's reward was what got appended to `results["rl_online"]` — i.e., reported as
"online RL performance." This conflates exploration noise with the reported metric.
Discovered because running `summarize_online_adaptation.py` against the existing (pre-fix)
`online_adaptation_results.json` showed online RL performing *worse* than a frozen,
randomly-initialized copy of itself in 2 of 3 stream segments — which cannot be a genuine
adaptation effect, since "frozen" is evaluated greedily and "online" wasn't.

**After:** each step now computes a separate **greedy** action (`evaluate=True`) purely for
the reported metric, in addition to the stochastic sample used for the actual policy
gradient update — matching how `rl_frozen` and `gbt` were already being evaluated
(deterministically). This makes the three-way comparison in Figure 1 / the online-
adaptation summary apples-to-apples for the first time. **This needs a full re-run to see
the corrected numbers** — the existing `results/online_adaptation_results.json` still
reflects the pre-fix (noisy-metric) behavior.

### 3.8 Also fixed: `17_latency_analysis.py` mock data broke silently

The mock `bm25`/`dense` dicts used to time feature extraction (`{"metrics_at_k": {"5":
{"mrr": 0.5}}}`) were written for the pre-fix feature schema. After the §2.1 fix,
`extract_lightweight_features` no longer reads `metrics_at_k`/`mrr` at all, so these mocks
silently degraded to always passing `0.0` — a silent-failure risk, not a crash. Updated to
the current raw-score schema (`top1_score`/`mean_score_top5`/`std_score_top5`). Only
affects the timing measurement's realism, not correctness of the timing numbers themselves.

### 3.9 Cleanup

Removed `results/online_adaptation_summary.md`, a stale artifact from testing
§3.4's script against the pre-fix `online_adaptation_results.json` — it would have been
confusing to find later since its numbers predate both the §2.1 leakage fix and the §3.7
reward-reporting fix.
