# Reproducing every claim in the paper

Each row below names a claim from `paper/main_v2.tex` and the command that regenerates it. If a
number in the paper is not reachable from this table, it should not be in the paper.

All commands run from the repository root. Everything in §1 is read-only and needs no GPU.

Numbers below are from the run committed at `39996af`: 100,000-passage Wikipedia pool,
untruncated reciprocal rank, retrieval at k=20, `split_seed=42`, 15 training seeds.
Provenance for every results file is in its own `_provenance` block — check it before comparing
across files, because `canonical_results.json` (real Wikipedia gold) and
`canonical_results_all.json` (all gold types, including template-constructed) are different
experiments with different means.

## 1 · Read the committed artifacts

> **Corrected 2026-08.** Seven rows below carried numbers a later audit found were wrong in the
> paper and had been transcribed here too — the same superseded-run failure this paper is about,
> in the file whose job is to prevent it. Each is marked ⟲ with what it replaced. None is a
> re-run: the artifacts were always right, the transcriptions were not.

| Paper claim | Command |
|---|---|
| ⟲ Oracle beats best-fixed by **26–59%** relative (was 21–56%, which matched neither population) | `cat results/learnability_analysis.md` |
| Pooled: 3,175 informative queries, 60.9% majority, 59.9% honest GBT, lift −1.0 points | `cat results/learnability_analysis.md` |
| ⟲ Retrieval-score-only lift **−1.45 points**, 59.40% vs 60.85% (was −1.4% against rounded 59.4/60.9, which does not reconstruct) — the retracted "Lean Router" | `cat results/learnability_analysis.md` — Table 3 |
| Dose–response: 59.9 / 65.1 / 69.0 / 72.7 / 73.1 | `cat results/leak_gradient.md` |
| ⟲ Permutation + random-feature nulls at **B=1000** — NQ p=0.066, TriviaQA 0.181, HotpotQA 0.740, **pooled 0.003**. Was B=20, where NQ read 0.048, *exactly* the attainable floor; the apparent NQ effect was the floor, and pooling is what actually clears chance | `cat results/perm_B1000/permutation_control.md` |
| Why the B=20 run is superseded and what moved | `cat results/perm_B1000/README.md` |
| **QPP control**: NQC / WIG / SMV / score entropy per arm on the shared index — lift **negative on all three** (−3.6 NQ, −4.1 TriviaQA, −1.2 HotpotQA; pooled −1.34 at n=3,614). The IR field's own difficulty predictors do not beat a constant policy either | `cat results/qpp_control_scale100000.json` |
| Embedding-state control: frozen MiniLM does **not** rescue routing — 59.7% (384-d) / 59.7% (32 PCA) / 59.9% (hand-61) pooled, against a 60.9% majority | `cat results/embedding_control.json` |
| **The leak transplants**: injected into RouterBench's 11-arm reward table, jitter retains 92.1–97.8% of the premium (97% here); honest prompt features lift +0.94 over a 30.1% majority | `cat results/leak_injection_routerbench.json` |
| Two-arm ablation: lift −2.0 / −2.8 / −5.0 | `cat results/two_arm_ablation.md` — Table 1 |
| ⟲ **The two-arm majority baseline also moves** (63.4→69.6, 61.7→69.0, 56.4→63.0), so the lift change is not attributable to removing the arm | `results/two_arm_ablation.json` — `per_dataset.*.two.pred.majority` |
| ⟲ MDE (superseded in the camera-ready by §7: test-split, one-sided 68/43/72% three-arm, 85/51/78% two-arm): capture **53–69%** of headroom, three-arm, on whole-corpus SD and headroom | `cat results/two_arm_ablation.md` — Table 2 |
| ⟲ Arm-selection bias **+0.0022 / −0.0010 / +0.0042**; **16–22%** of headroom on ≤1% margins; +16–29% inflation at 10% noise | `cat results/oracle_inflation.md` |
| 34.3% of NQ queries contain answer-bearing distractors, mean 4.59 | `cat results/corpus_census.json` |
| Convex fusion loses on two of three datasets; captures +7.0% on TriviaQA | `cat results/fusion_baseline.md` |
| ⟲ Pool 20K→100K moved pooled lift **−2.0% → −1.0%** (was −1.3%→−1.0%); the fixed-pool learning curve attributes ~0.8 of that to the accompanying rise in *n* | `cat results/learning_curve.md` |
| RouterBench: 93.0% tied, 69.9% of headroom on ≤1% margins, +40% at 10% noise, no feature matrix shipped | `cat results/external_audit_routerbench.md` |
| ⟲ Per-arm effectiveness; **Hybrid loses to Dense on all three** | `cat results/canonical_results.json` |
| Capture table, every method × dataset (Table 3) | `cat results/main_results_table.tex` |
| **Leak premium grows with corpus size**: +9.2 → +10.6, 20K→100K | `cat experiments/corpus_v3/results/leak_vs_scale.json` |
| Jitter cost by scale: +0.21 / −0.17 / −0.12 (the argmax rebuttal) | `cat experiments/corpus_v3/results/leak_vs_scale.json` |
| Leak premium within each corpus: +11.9 / +12.4 / +16.8 | `results/learnability_analysis.json` — `per_dataset.*.honest`/`.leaky` |
| Features 22–23 alone reach 73.4%; all 61 minus the leak reach 59.9% | `cat results/paper_claim_checks.json` |
| Retrieval sanity per scale (Hit@5 among answerable, unanswerable share) | `cat experiments/corpus_v3/results/retrieval_summary.json` |
| RL algorithm span **0.0139 MRR** (REINFORCE/PPO/GRPO, NQ). **Separate sweep**: 5 seeds, 300 epochs — *not* the 15-seed/150-epoch run behind Table 3's RL row (0.1512), so the two are not directly comparable | `cat experiments/router_experiments/results/algorithm_comparison.json` |
| RouterBench benchmark count. **The artifact disagrees with itself**: `artifact_audit.n_benchmarks` = 86, its prose summary says 87. The paper cites 86 (the computed field). Confirm against the pkl | `cat results/external_audit_routerbench.json` |
| Detector audit: \|rho\| = 0.881 with 0.0% exact agreement. **No Pearson coefficient is recorded** — the artifact says only "Pearson agrees", so the paper claims no Pearson value | `cat results/external_audit_routerbench.md` |
| Per-retriever regression: negative on 11 of 15; 2 nominal p<0.05, both HotpotQA ridge variants; Bonferroni alpha = 0.05/15 = 0.0033 | `cat results/per_retriever_router.md` |
| Known-bad artifacts: a non-reproducible CV standard deviation, and the MDE test-split size | `cat results/KNOWN_ISSUES.md` |

## 2 · Re-run the checks

```bash
# The Score-Gap fix is present and the table agrees with the code that produced it.
# Recomputes compute_heuristic_baseline live and diffs it against canonical_results.json.
# Expect three "matches a live recomputation" lines.
python experiments/router_experiments/run_lab_finish.py --skip-canonical --skip-nulls

# No feature in the shipped env data encodes any arm's reward, and the detector has teeth
# (it must fire on the deliberately-leaky variants, or the clean result proves nothing).
python -m pytest tests/test_no_feature_reward_leak.py leakcheck/tests/ -q     # 28 leakcheck tests + 2 leak-guard tests

# Full suite.
python -m pytest tests/ leakcheck/tests/ -q
```

## 3 · Regenerate from scratch

Stage 1 needs a GPU machine (~90 min); the rest is CPU.

```bash
# Everything that is stale, in dependency order, with verification between stages.
python experiments/router_experiments/run_lab_finish.py --dry-run   # inspect the plan first
python experiments/router_experiments/run_lab_finish.py

# Individual analyses (all CPU, minutes each):
python experiments/router_experiments/learnability_analysis.py --leaky-suffix _leaky
python experiments/router_experiments/leak_gradient.py
python experiments/router_experiments/permutation_control.py --n-perm 20
python experiments/router_experiments/two_arm_ablation.py
python experiments/router_experiments/oracle_inflation.py
```

If `permutation_control.py` dies with `BrokenProcessPool`, run one dataset per process
(`--datasets nq --n-jobs 4`). The cause is peak memory in joblib's worker pool, not a bug in the
analysis; see the docstring in `_cv_score`.

## 4 · The external audit

RouterBench ships pickles, and unpickling executes arbitrary code, so `audit_external.py` never
downloads anything — fetch and vet the file yourself, then point the script at it.

```bash
# https://huggingface.co/datasets/withmartian/routerbench -> routerbench_0shot.pkl (~100 MB)
python experiments/router_experiments/audit_external.py --data /path/to/routerbench_0shot.pkl
```

## 5 · Auditing your own feature matrix

`leakcheck` is domain-independent. It needs no model fitting and only NumPy.

```python
import leakcheck
report = leakcheck.audit(X, R)      # X: (n, d) features; R: (n, k) per-arm rewards
print(report)
if report.leaked:                   # exact agreement only -- a fact about the arrays
    raise SystemExit("reward leakage in features")
print(report.suspect_features)      # rank/proxy evidence -- resolve these by ablation
```

Only **exact** agreement yields a verdict. Rank-based findings are suspicions, because a confound
through a shared upstream cause produces the same signature as a monotone leak — see
`leakcheck/CALIBRATION.md` for the RouterBench case that forced this design and the measured
thresholds.

## 6 · Camera-ready corrections (2026-10-31)

The CRC text corrects or adds the numbers below. The audit that found them is
`paper/crc_audit/CRC_PUBLISHABILITY_AUDIT.md` together with
`paper/crc_audit/CRC_METHODS_CLAIMS_LITERATURE_AUDIT.md`. The CRC experiments currently sit one
level deeper than `experiments/crc_v2/common.py` writes, at `results/crc_v2/crc_v2/`.

| Paper claim (CRC wording) | Was | Command |
|---|---|---|
| The leaked features are **MRR@20**, which equals the reward on 26–45% of rows (by dataset and arm). Figure 1's leak is **MRR@5**, zero on 76–95% of rows | "is the reward" | `python experiments/crc_v2/crc_text_checks.py --skip-gb-refit --out-dir <new dir>` (sections `leak_equality` and `scale_mrr5_zero`) |
| Independent additive noise of ±0.15 keeps **54%** of the GB gain (+7.1 of +13.3; 53–55% over three draws) and 33% under LR. A true-reward copy reaches 91.1% and keeps 22% | "15% noise costs 0.4 points" | `cat results/crc_v2/crc_v2/r5_random_noise/summary.json` |
| The value-keyed jitter is a relabelling: it keeps 92% at ±100%. At ±50% or more leakcheck raises no suspicion, yet GB gains +12.3 | split-point explanation | `results/crc_v2/crc_v2/r1_classifier_sweep/summary.json`, `results/crc_v2/crc_v2/r3_leakcheck_sweep/leakcheck_sweep.json` |
| Tuned GB matches Always-Dense on **169/174, 231/231, 138/138** | 172/174, 228/231 | `python experiments/crc_v2/crc_text_checks.py --out-dir <new dir>` (section `gb_test_actions`, refits GB, about 10 min) |
| RL row, with the best-validation checkpoint restored: **0.1618 ± .0061 / 0.2000 ± .0037 / 0.0722 ± .0036**. Mean capture −10.8 / −0.0 / −9.1%; single seeds −49% to +9% | 0.1512 / 0.1881 / 0.0683 (final epoch) | `cat results/crc_v2/crc_v2/rl_restore/summary.json`; capture from `crc_text_checks` `rl_restored` |
| No method beats Always-Dense at p < 0.05. The smallest p is 0.094 (score-gap, NQ) | — | `crc_text_checks` `bootstrap_vs_dense` |
| A train-selected comparator (Hybrid on HotpotQA) would show 12–35% "capture", all of it from always choosing Dense | — | `crc_text_checks` `train_selected_comparator` |
| Distractors: about half are mined by query-term overlap, the rest random (`negative_strategy = mixed`) | "499 hard negatives" | `cat results/corpus_census.json` (`_config`, `hard_negative_share`) |
| Census: 34.3% of the **1,989 prepared** NQ queries; 13.4 answer-bearing distractors per contaminated query | "mean 4.59 each" | `cat results/corpus_census.json` (9,125 / 682) |
| Convex fusion on HotpotQA: −45.2% | −45.3% | `cat results/paper_claim_checks.json` |
| Figure 11 (detection power): what the MRR@20 leak still buys vs what equality/leakcheck flag, per noise level | — (new) | `python experiments/crc_v2/c_detection_power.py --out-dir <new dir>` (reads R1/R3/R5; redraws `results/figures/fig11_detection_power.pdf`) |
| Appendix positive control: genuine gain +14.0/+17.4 with no verdict; noisy leak adds 26–34 pts; suspicions also land on genuine signal features | — (new) | `python experiments/crc_v2/b_positive_control.py --out-dir <new dir>` (~25 min CPU) |
| §6 leakcheck 0.2 `audit_v2` (decodability): flags slots 22–23 on the ±50%/±100% value-keyed leak where v1 flags nothing; labels a planted group confound in 10/10 trials; also flags honest overlap features 26–28 | — (new) | `python experiments/crc_v2/h_detector_benchmark.py --out-dir <new dir>` (~20 min CPU); see `results/crc_v2/h_detector_benchmark/README.md` |
| Appendix A provenance: leak introduced in `4d2dade` (23 Jul), fixed in `eaa39e7` (2 Aug), regression test `bab7e48`, graded arms `17577c2` | — (new) | `git show 4d2dade:src/router/prepare_env_data.py \| grep -n 'get("mrr"'` |

## 7 · Camera-ready corrections from review_2 (2026-11-09)

See `paper/review_2/CHANGES_APPLIED.md`. Run from `experiments/crc_v2/`.

| Paper claim | Was | Command |
|---|---|---|
| Test-split MDE, one-sided α 0.05, 80% power: **68 / 43 / 72%** of headroom (three arms), **85 / 51 / 78%** (two arms); two-sided 76/48/81% and 96/58/88% | 53–69% / 65–83% (whole-corpus SD and headroom) | `python r6_testsplit_power.py --out-dir <new dir>` (seconds) |
| Score-gap NQ capture **+0.4%, 95% CI 0.0–1.5%**; largest upper limit 45.6% (ridge, HotpotQA) | no interval | same, `capture` block (paired bootstrap, B = 10,000, seed 20261108) |
| Table 2 Capture column: −0.4 / +24.9 / +28.9 / +68.7 / +69.6% (tuned GB, 3 datasets × 5 split seeds) | — (new) | `python r9_leak_capture_summary.py --out-dir <new dir>` (aggregates R4/O1 rows; fits nothing) |
| Multiplicative ±15% noise keeps **99.6%** of the GB premium (+13.2; 99–100% across 3 draws), LR all of it; no verdict, 7 suspicions, 0 exact matches | — (new) | `python r7_multiplicative_noise.py --scales 0.15 --draws <d> --n-jobs 1 --anchors-from ../../results/crc_v2/crc_v2/r5_random_noise/summary.json --out-dir <dir>/draw<d>` for d = 0, 1, 2 (~4 min per fit; 5 fits per draw) |
| Standardised LR keeps 36% (recoding) and 29% (additive ±0.15) | — (new) | `python r8_scaled_logreg.py --out-dir <new dir>` |
| Within-dataset label shuffle: 59.18 ± 0.23, B = 60, p = 0.016 (floor) | not quoted | `cat results/crc_v2/perm_within_dataset/summary.json` |

Note: "Table 2 Capture column" above is now **Figure 1(b)** (`fig12_leak_ladder.py`); Table 2 in the
camera-ready is the arms table.

## 8 · Camera-ready corrections from review_3 (2026-11-12)

See `paper/review_3/00_editorial_decision_and_roadmap.md`. Run from `experiments/crc_v2/`.
One script, `python r10_review3_checks.py --out-dir <new dir>`, writes every number below to
`results/crc_v2/r10_review3_checks/summary.json` (the leaked-router test refits tuned GB, a few
minutes; `--skip-leaked` gives the rest in about a minute).

| Paper claim | Was | Block in `summary.json` |
|---|---|---|
| Shared 100K index: Dense best only on NQ; Hybrid best on TriviaQA and HotpotQA; BM25 > Dense on HotpotQA (answerable queries, reciprocal rank) | not reported | `shared_index_arms` |
| MDE under a selective-router variance model: **47 / 20 / 54%** (vs 68/43/72% under the oracle's variance) | "these limits are optimistic" | `mde_bracket` |
| Smallest Table 3 p = **0.26** (score-gap, NQ, which changes one test query); ties counted | 0.094 (a floating-point tie artifact) | `method_pvalues` |
| Equality check (leakcheck verdict) catches the *one arm* and *exact* rungs only | not shown | `ladder_verdicts` |
| RQ2's test applied to the leaked tuned router (split 42) | — (new) | `leaked_router_test` |
| Apparent capture of the exact leak 48–89% across datasets and splits | not quoted | `cat ../../results/crc_v2/r9_leak_capture_summary/summary.json` (`leaky` min/max) |
| 27–46% of queries contain answer-bearing distractors; gold kept for 1,157/1,989, 1,539/1,894, 916/1,995 queries | cited to arabzadeh2022shallow | `cat ../../results/corpus_census.json` |
| BM25's IDF is computed per 500-passage candidate set | not stated | `experiments/baseline_v2/02_run_baselines.py` (`bm25.fit(chunks)` per query) |
| leakcheck 0.2.1: relative tolerance (float32 copies keep their verdict); binary rewards get a verdict on an exact full copy | 0.2.0 missed both | `python -m pytest leakcheck/tests/test_review3_fixes.py -q` |
| QueryRoute audit (§6): 4 datasets, 43–1,406 queries, 484 features × 11 candidates, 5 metrics; 0 verdicts, 0 exact agreement; max \|ρ\| 0.51 (n ≥ 300; 0.71 on TREC DL 2019, n = 43), max held-out R² 0.26 | "has since shipped its selector's features" | `python r11_queryroute_audit.py --download --data-dir <dir outside repo> --out-dir <new dir>` (downloads ~25 MB of their release, Qwen2.5-7B backbone, BM25 outcomes; their data is not redistributed here). Artifact: `results/crc_v2/r11_queryroute_audit/summary.json` |

## Known gaps

- **`paper/main_v2.tex` has never been compiled.** No LaTeX toolchain was available where it was
  written. Structure was checked programmatically (environments balanced, braces balanced, tabular
  column counts match row widths) but a real `pdflatex` run is required, along with the official
  NeurIPS style file, before the page count means anything.
- ~~Two bib entries are cited but unread~~ **Resolved 2026-08-10.** `arabzadeh2021bertselect` and
  `tang2025mbarag` were verified against primary sources; the `TODO(verify)` markers are gone and
  the dated provenance now sits in `%` comments above each entry.
- **All bibliography provenance lives in `%` comments outside entries, never in `note={}`.**
  `\bibliographystyle{plain}` typesets `note`, so an audit trail placed there is published. Do not
  move it back.
- **One split seed.** Every number uses `split_seed=42`; the reported spread is training-seed
  variance only.
- **`results_backup_*/` is gitignored but the history still contains it**, so `.git` is ~1.9 GB.
  Deliberately not rewritten before the deadline.
