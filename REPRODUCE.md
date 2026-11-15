# Reproducing the paper

Each row maps a number in `paper/main.tex` or `paper/supplementary.tex` to the file it is read
from and the command that regenerates that file. Commands run from the repository root unless
stated otherwise. Scripts under `experiments/analyses/` refuse to overwrite an existing output
directory, so pass a new `--out-dir`.

Setting for all numbers: 100,000-passage Wikipedia pool, untruncated reciprocal rank as reward,
retrieval at k = 20, `split_seed = 42`, 15 training seeds. Every results file carries a
`_provenance` block recording the inputs it was computed from.

## 1. Setup and the leak (Sections 1, 3, 4)

| Claim | Source | Regenerate |
|---|---|---|
| Oracle beats the best fixed retriever by 21–46% relative (test split) | `results/canonical_results.json`, `results/headroom_analysis.md` | `python experiments/router_experiments/canonical_evaluation.py`, then `headroom_analysis.py` |
| 3,175 informative queries; majority 60.9%, label-free GB 59.9% | `results/learnability_analysis.md` | `python experiments/router_experiments/learnability_analysis.py --leaky-suffix _leaky` |
| Leak ladder accuracy 59.9 / 65.1 / 69.0 / 72.7 / 73.1% (Figure 1a) | `results/leak_gradient.json` | `python experiments/router_experiments/leak_gradient.py` |
| Apparent capture of the oracle gap per rung, −0.4 to +69.6%; exact leak 48–89% across datasets and splits (Figure 1b) | `results/analyses/leak_capture_summary/summary.json` | `python experiments/analyses/leak_capture_summary.py --out-dir <dir>` |
| Equality check catches only the one-arm and exact rungs (Figure 1 labels) | `results/analyses/ladder_checks/summary.json` (`ladder_verdicts`) | `python experiments/analyses/ladder_checks.py --out-dir <dir>` |
| Figure 1 | `results/figures/fig12_leak_ladder.pdf` | `python experiments/analyses/plot_leak_ladder.py` |
| MRR@20 equals the reward on 26–45% of rows | `results/analyses/paper_number_checks/summary.json` (`leak_equality`) | `python experiments/analyses/paper_number_checks.py --skip-gb-refit --out-dir <dir>` |
| Features 22–23 alone reach 73.4% | `results/paper_claim_checks.json` | `python experiments/router_experiments/verify_paper_claims.py` |
| Leak premium within each corpus: +11.9 / +12.4 / +16.8 | `results/learnability_analysis.json` (`per_dataset`) | `learnability_analysis.py` |
| Additive ±0.15 noise keeps 54% of the GB gain (33% under LR); a true-reward copy reaches 91.1% | `results/analyses/random_noise_sweep/summary.json` | `python experiments/analyses/random_noise_sweep.py --out-dir <dir>` |
| Value-keyed recoding keeps 92% at ±100%; at ±50% leakcheck v1 raises nothing while GB gains +12.3 | `results/analyses/classifier_noise_sweep/summary.json`, `results/analyses/leakcheck_noise_sweep/leakcheck_sweep.json` | `build_noise_variants.py`, then `classifier_noise_sweep.py` and `leakcheck_noise_sweep.py` |
| Multiplicative ±15% noise keeps 99–100% of the GB premium | `results/analyses/multiplicative_noise/` | `python experiments/analyses/multiplicative_noise.py --scales 0.15 --draws <d> --out-dir <dir>` for d = 0, 1, 2 |
| Standardised LR keeps 36% (recoding) and 29% (additive) | `results/analyses/scaled_logreg/summary.json` | `python experiments/analyses/scaled_logreg.py --out-dir <dir>` |
| Detection power (Figure 2) | `results/analyses/detection_power/table.json` | `python experiments/analyses/detection_power.py --out-dir <dir>` |
| Leak premium at 20K–100K passages: +9.2 → +10.6 (Figure 3) | `experiments/corpus_v3/results/leak_vs_scale.json` | `python experiments/corpus_v3/06_leak_vs_scale.py` (needs the corpus_v3 build; see its README) |
| RouterBench injection: +16.2 to +63.7 points; 15% multiplicative noise keeps 92–98% | `results/leak_injection_routerbench.json` | `python experiments/router_experiments/leak_injection_external.py --data <routerbench_0shot.pkl>` |

## 2. Routing without the leak (Section 5)

| Claim | Source | Regenerate |
|---|---|---|
| Per-arm effectiveness and capture per method (Table 3) | `results/canonical_results.json`, `results/main_results_table.tex` | `python experiments/router_experiments/canonical_evaluation.py`, then `generate_latex_tables.py` |
| Smallest p against the best fixed retriever: 0.26 | `results/analyses/ladder_checks/summary.json` (`method_pvalues`) | `ladder_checks.py` |
| Leaked router passes the same test (p = 0.016, < 0.001, 0.005) | `results/analyses/ladder_checks/summary.json` (`leaked_router_test`) | `ladder_checks.py` |
| Test-split MDE 68 / 43 / 72% (oracle variance), 47 / 20 / 54% (selective router) | `results/analyses/split_power/summary.json`, `ladder_checks` (`mde_bracket`) | `python experiments/analyses/split_power.py --out-dir <dir>` |
| Score-gap capture on NQ +0.4%, 95% CI 0.0–1.5% | `results/analyses/split_power/summary.json` (`capture`) | as above |
| RL row with the best-validation checkpoint | `results/analyses/rl_checkpoint_rerun/summary.json` | `python experiments/analyses/rl_checkpoint_rerun.py --out-dir <dir>` |
| Shuffled labels 59.1 ± 0.2 (p = 0.003); within-dataset shuffle p = 0.016 (Figure 4) | `results/perm_B1000/`, `results/analyses/permutation_within_dataset/summary.json` | `python experiments/router_experiments/permutation_control.py --n-perm 1000` |
| Frozen query embeddings: 59.7% | `results/embedding_control.json` | `python experiments/router_experiments/embedding_control.py` |
| NQC / WIG / SMV predictors: −1.34 points | `results/qpp_control_scale100000.json` | `python experiments/router_experiments/qpp_control.py` |
| Shared-index arm ordering (Hybrid best on TriviaQA and HotpotQA) | `results/analyses/ladder_checks/summary.json` (`shared_index_arms`) | `ladder_checks.py` |
| 27–46% of queries contain answer-bearing distractors | `results/corpus_census.json` | `python experiments/router_experiments/verify_paper_claims.py` |

## 3. The audit tool (Section 6)

| Claim | Source | Regenerate |
|---|---|---|
| leakcheck verdict rules (25% threshold, binary rewards, float tolerance) | `leakcheck/tests/` | `python -m pytest leakcheck/tests -q` |
| Detector comparison (Table 4) and `audit_v2` results | `results/analyses/detector_benchmark/`, `results/analyses/detector_benchmark_v0.2.1/` | `python experiments/analyses/detector_benchmark.py --out-dir <dir>` |
| RouterBench confound: \|ρ\| = 0.881 with no exact agreement | `results/external_audit_routerbench.md` | `python experiments/router_experiments/audit_external.py --data <routerbench_0shot.pkl>` |
| QueryRoute: 0 verdicts on 4 datasets, max \|ρ\| 0.51, max held-out R² 0.26 | `results/analyses/queryroute_audit/summary.json` | `python experiments/analyses/queryroute_audit.py --download --data-dir <dir outside the repo> --out-dir <dir>` |

## 4. Supplementary material

| Claim | Source | Regenerate |
|---|---|---|
| Positive control: genuine gain +14.0 / +17.4, noisy leak adds 26–34 points (S1) | `results/analyses/positive_control/summary.json` | `python experiments/analyses/positive_control.py --out-dir <dir>` |
| Corpus census (S2) | `results/corpus_census.json` | as above |
| Two-arm ablation | `results/two_arm_ablation.json` | `python experiments/router_experiments/two_arm_ablation.py` |
| Oracle inflation under label noise | `results/oracle_inflation.md` | `python experiments/router_experiments/oracle_inflation.py` |
| Fusion, cascade and per-retriever regression baselines | `results/fusion_baseline.md`, `results/cascade_router.md`, `results/per_retriever_router.md` | the scripts of the same name in `experiments/router_experiments/` |
| REINFORCE / PPO / GRPO span 0.0139 MRR on NQ (5 seeds, 300 epochs) | `experiments/router_experiments/results/algorithm_comparison.json` | `python experiments/router_experiments/rl_algorithm_comparison.py` |

## External data

RouterBench is distributed as pickles, and unpickling runs arbitrary code, so no script here
downloads it. Fetch `routerbench_0shot.pkl` from
https://huggingface.co/datasets/withmartian/routerbench, inspect it, and pass its path with
`--data`. The QueryRoute audit downloads JSON files only, into a directory you choose; their
data is not redistributed in this repository.

## Known limitations

All numbers use a single train/val/test split (`split_seed = 42`); reported spreads are over
training seeds. Defects found in our own artifacts are listed in `results/KNOWN_ISSUES.md`.
