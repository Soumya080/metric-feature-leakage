# Validation of the CRC v2 results

**Validated:** 2026-10-30, independently, on a second machine. The result files were only read; the one write into this folder was this report. The local rebuild went to a scratch directory. Before-and-after hashes were not taken.

**Where the runs came from:** another machine, Python 3.11.9, path `<local-path> (2)\...`, on 2026-10-29. In this repository the folder is nested as `results/crc_v2/crc_v2/`. All of the scripts' default paths point to `results/crc_v2/`, so any reproduction command needs `--env-dir` and `--out-dir` set to this folder.

## 1. Integrity: all checks pass

| Check | Result |
|---|---|
| All expected outputs present (R0, R1 build, R1 sweep, R3, R4, R4 with extra split seeds, RL) | yes |
| The 21 input files hashed in R0 match this machine's `src/router/` byte for byte | 21/21 |
| R1 manifest: source hashes match local sources; the 54 variant files match their recorded hashes | yes / 54/54 |
| The 54 variant files rebuilt independently here are byte-identical, and every check in the build script passed locally | 54/54 |
| R0: Table 2 recomputed against the committed `results/leak_gradient.json` | 0 mismatches |
| R0: honest and leaky rows reproduce `learnability_analysis.json` | OK |
| R1: 23 arms × 5 seeds, no duplicates, n = 3175, majority 0.6085, 61 features | 115/115 |
| R1 Table 2 checks at CV seed 0: honest / noisy / leaky | 59.87 / 72.66 / 73.07 (paper 59.9 / 72.7 / 73.1); OK |
| R1: two cells re-fitted here (`rankreward_s000`, `mrr20_s100`, seed 0), gradient boosting and logistic regression | identical to the saved values to full float precision |
| R4: honest tuned gradient boosting at split 42 reproduces Table 3 | 0.1642 / 0.2000 / 0.0753; OK |
| R4: test sizes are 174 / 231 / 138 at every split seed | yes |
| RL: 45 runs, 15 seeds × 3 datasets, constants equal to `canonical_evaluation` | yes |
| RL: best epoch is the first maximum of validation reward; per-query means equal the stored means; one source hash | yes |
| RL: final-epoch weights (the old method) reproduce Table 3's RL row within one SD | 0.1507 / 0.1878 / 0.0694 vs 0.1512 / 0.1881 / 0.0683 |
| Unit tests `tests/test_crc_v2_setup.py` | 7 passed |

Nothing failed.

## 2. What the results say

### R1 / R2: accuracy against noise level
Gradient boosting (GB) and logistic regression (LR) accuracy, mean over cross-validation seeds 0–4, pooled over the three datasets. "Kept" is the share of the zero-noise gain that survives at that noise level.

| Noise half-width | MRR@20 leak, GB % (kept) | true reward, GB % (kept) | MRR@20 leak, LR kept | true reward, LR kept |
|---|---|---|---|---|
| 0 | 73.10 (1.00) | **91.08** (1.00) | 1.00 | 1.00 |
| 0.02 | 73.19 (1.01) | 76.38 (0.53) | 0.97 | 0.91 |
| 0.05 | 72.91 (0.99) | 72.77 (0.41) | 0.88 | 0.62 |
| 0.10 | 72.69 (0.97) | 68.82 (0.29) | 0.62 | 0.33 |
| 0.15 | 72.55 (0.96) | 66.79 (0.22) | 0.44 | 0.22 |
| 0.25 | 72.34 (0.94) | 65.59 (0.18) | 0.30 | 0.14 |
| 0.50 | 72.06 (0.92) | 65.20 (0.17) | 0.22 | 0.06 |
| 1.00 | 72.08 (0.92) | 64.36 (0.15) | −0.00 | −0.01 |

Honest features score 59.82 (GB) and 60.50 (LR) against a 60.85% majority. Clipping the noise to [0, 1] at 0.15 changes nothing material (72.52 and 66.63).

- **The true reward copy buys +31.3 points under GB at 61 features**, against +13.3 for the paper's "exact" MRR@20 rung.
- **Why the MRR@20 leak survives even 100% noise under GB: the "noise" is not noise.**
  - The jitter is a fixed function of the value (crc32 of the value), so every query with the same reward gets the same offset.
  - MRR@20 takes only 21 distinct values, and 60–69% of rows are 0.
  - Checked on the s = 1.00 files: no two distinct original values map to the same jittered value. The perturbed feature is a lossless relabelling, and a tree can decode it whatever the scale.
  - The true reward has 451–494 distinct values per arm, so the relabelling is much harder to learn and the GB gain falls off with noise.
  - Linear models cannot decode a non-monotone relabelling, which is why LR loses the leak in both families.
  - Consequence: the paper's "15% noise costs 0.4 points" and "retains 97%" describe a fixed, value-keyed relabelling, not random noise. These results do not measure how a leak with independent per-query noise behaves.

### R3: what leakcheck and the equality check report (pooled verdicts / suspicions)
- **Honest features:** 0 / 0 overall and on every dataset. No false alarm, and nothing is ever flagged outside slots 22–25.
- **Exact copies** (`leaky`, `mrr20_s000`, `rankreward_s000`): direct verdicts. The MRR@20 copy matches the true reward on 31% (BM25) and 40% (Dense) of rows, above the 0.25 threshold.
- **Every perturbed variant:** exact-match rate 0.00, so the equality check never fires.
- **Rank suspicions fade with noise:**
  - MRR@20: 7 suspicions at s ≤ 0.05, 6 at 0.10–0.15, 4 at 0.25, **0 at 0.50 and 1.00**.
  - True reward: 2 suspicions remain at 0.50 and 1.00 (NQ and TriviaQA only; none on HotpotQA).
- **Harmful but invisible:** `mrr20_s050` and `mrr20_s100` still buy +12.3 GB points while **neither the equality check nor leakcheck flags anything**.
- **`leak_noisy` and `mrr20_s015`:** suspicions in the pooled audit and on every dataset (HotpotQA: 3). The §6 claim holds at the paper's 15%.
- **Single-feature argmax lift** (exploratory): at most +0.045 and near 0 for most variants. It does **not** track the accuracy gains, so it is not usable as an inflation predictor in its current form.

### R4: how much of the oracle gap each router appears to capture (tuned GB)

**Table 3 split (seed 42).** Capture against the test-best fixed arm:

| Router | NQ | TriviaQA | HotpotQA |
|---|---|---|---|
| Honest | −3.8% | 0.0% | 0.0% |
| `leaky` | 48.5% | 69.6% | 63.8% |
| True-reward copy | 44.1% | 81.1% | 59.7% |

**The other four split seeds (7, 13, 99, 2024):**
- **Honest features** capture at most 0.0% against the test-best arm on every dataset and seed.
- **Leaked routers** capture 41–91%.
- **The fixed-arm choice matters.** Measured against the arm the validation set would pick instead of the test-best arm, honest capture can look positive: up to +36%, when validation picks Hybrid or BM25 and that arm then does worse on test.

### RL: Table 3's RL row with the best validation checkpoint actually restored
Mean ± SD over 15 seeds; capture is against Always-Dense.

| | NQ | TriviaQA | HotpotQA |
|---|---|---|---|
| Final-epoch weights (old protocol) | 0.1507 ± .0068 | 0.1878 ± .0037 | 0.0694 ± .0045 |
| **Best checkpoint restored** | **0.1618 ± .0061** | **0.2000 ± .0037** | **0.0722 ± .0036** |
| Always-Dense | 0.1655 | 0.2000 | 0.0753 |
| Mean capture after restoring | −10.8% | −0.00% | −9.1% |
| Seeds beating Always-Dense | 4/15 | 9/15 | 2/15 |
| Best single seed | +6.7% | +8.0% | +8.6% |

- **Restoring raises RL reward by more than one SD on NQ and TriviaQA**, as the pre-registered rule required checking. The mean still does not beat the best fixed arm, so the "≤ 0.4% capture" claim survives for the mean.
- **Best epochs** are early on NQ (1–24) and TriviaQA (16–81), but late on HotpotQA (63–150). HotpotQA's validation reward may still be rising at epoch 150.

## 3. Pre-registered decision rules

| Rule | Outcome |
|---|---|
| True reward at 15% keeps less than 50% of its gain under GB → narrow the abstract's "retains most" | **Triggered** (22%). Also qualify the 97% figure as a value-keyed relabelling (§2, R1/R2). |
| Equality check misses every arm with noise above 0 while GB still gains more than 2 points → keep the "equality audit does not catch it" claim | **Keep.** Exact-match is 0.00 for every perturbed arm. The GB gain exceeds 2 points everywhere, minimum +4.5 at `rankreward_s100`. |
| leakcheck raises a suspicion on `leak_noisy` and `mrr20_s015`, pooled and per dataset → keep §6's claim | **Keep at 15%.** Add that suspicions disappear for MRR@20 at noise ≥ 0.50 while the GB gain stays +12.3 points. |
| RL restored differs from final-epoch by more than one SD → state it in the text | **Triggered** on NQ and TriviaQA. Put the restored row in Table 3 and move the old row to a footnote. |

## 4. Caveats and open items
- Results are on one machine with one environment. Two cells were re-fitted here and match exactly.
- The R1 variants hold slot 25 honest, whereas the paper's rungs do not. `mrr20_s000` (73.10) matches `leaky` (73.15), so slot 25 adds about nothing.
- No variant with independent per-query noise has been run. Without one, the paper cannot claim that randomly noisy leaks survive. *(Resolved by R5, §5.)*
- The R4 and RL numbers come from single-split test sets of 138–231 queries; their uncertainty is large.

## 5. R5: independent per-query noise (`r5_random_noise/`)

Added after §2 showed that R1's jitter is a value-keyed relabelling. R5 adds Uniform(−s, s)
noise drawn independently per query and arm (numpy `default_rng([draw, arm])`, 3 draws, the same
draw reused across scales). Slots 22–24 change and slot 25 stays honest, as in R1. Evaluation
is Table 2's protocol (`leak_gradient.evaluate`, CV seeds 0–4). leakcheck runs on the pooled
matrix only; there is no per-dataset audit in R5.

### Integrity: all checks pass
| Check | Result |
|---|---|
| Grid complete | 225/225 fits = 45 (arm, draw) cells × 5 CV seeds; no duplicates or NaNs. 45/45 leakcheck rows. |
| Population | `n_informative` = 3175 and majority = 60.85% in every row, as in Table 2 |
| Built-in R1 regression check (s = 0 arms) | `honest_minimal`, `mrr20_exact`, `rankreward_exact` reproduce R1's GB means exactly (59.82 / 73.10 / 91.08) |
| Local re-fit on this machine (Python 3.10) | `mrr20_rand_s015` draw 1 CV seed 3, and `rankreward_rand_s050` draw 2 CV seed 4: GB, LR and the leakcheck row are bit-identical to the saved values |
| `r5_random_noise_quick/` (draw 0, CV seed 0) | All 17 rows and 17 audits are identical to the matching cells of the full run |
| Provenance | Run on the same external machine as §1 (Python 3.11.9, ~1.29 h of fit time). `run_header.json` records the external path, as in §1. |

### Gradient-boosting retention: random noise vs R1's value-keyed jitter
Pooled 61-D, mean of 5 CV seeds (× 3 draws for R5). Retention = (acc − honest)/(exact − honest).
Honest = 59.82; MRR@20 exact = 73.10 (+13.28); true-reward exact = 91.08 (+31.26).

| s | MRR@20, R1 value-keyed | **MRR@20, R5 random** | True reward, R1 value-keyed | **True reward, R5 random** |
|---|---|---|---|---|
| 0.02 | 73.19 (1.01) | 72.92 (0.99) | 76.38 (0.53) | 75.87 (0.51) |
| 0.05 | 72.91 (0.99) | 71.42 (0.87) | 72.77 (0.41) | 71.75 (0.38) |
| 0.10 | 72.69 (0.97) | 68.59 (0.66) | 68.82 (0.29) | 68.53 (0.28) |
| **0.15** | 72.55 (0.96) | **66.93 (0.54)** | 66.79 (0.22) | **66.75 (0.22)** |
| 0.25 | 72.34 (0.94) | 65.36 (0.42) | 65.59 (0.18) | 65.26 (0.17) |
| 0.50 | 72.06 (0.92) | 62.99 (0.24) | 65.20 (0.17) | 62.85 (0.10) |
| 1.00 | 72.08 (0.92) | 61.37 (0.12) | 64.36 (0.15) | 61.35 (0.05) |

- **The MRR@20 survival in Table 2 comes from the relabelling, not from robustness to noise.**
  With independent noise at s = 0.15, GB keeps 54% of the MRR@20 gain (+7.11 of +13.28 points;
  per-draw 0.53–0.55; per-fit 66.68–67.22). Value-keyed jitter keeps 96%.
- **For the true reward the two kinds of noise agree up to s = 0.25** (0.22 vs 0.22 at 0.15).
  With about 450 distinct values per arm, value-keyed jitter acts almost like random noise. At
  s ≥ 0.50 value-keyed noise retains more (0.17 vs 0.10). The likely reason is that frequent
  values such as 1.0 and 0.5 still share one code, but this is a conjecture and was not tested.
- **Under random noise the leaked metric and the true reward converge.** From s = 0.10 their GB
  accuracies differ by at most 0.2 points. Once the noise is comparable to the value spacing,
  what survives is the coarse signal both families share.
- **Logistic regression** keeps 33% (MRR@20) and 29% (true reward) at s = 0.15, and none
  (−0.01) at s = 1.0.
- **The noise realisation barely matters.** The draw-to-draw range of GB means is at most
  0.93 points across all arms.

### leakcheck and equality check under random noise (pooled, all 3 draws agree)
| Arms | Verdict | Suspicions | Exact-match 22 / 23 | \|ρ\| 22 / 23 (MRR@20) | GB gain (MRR@20 / true reward) |
|---|---|---|---|---|---|
| honest | no | 0 | 0.00 / 0.00 | 0.06 / 0.13 | — |
| exact (s = 0) | **yes** (2) | 5 | MRR@20 0.31 / 0.40; true reward 1.00 / 1.00 | 0.82 / 0.89 | +13.3 / +31.3 |
| s = 0.02–0.05 | no | 7 | 0.00 / 0.00 | 0.64–0.66 / 0.77–0.78 | +11.6–13.1 / +11.9–16.1 |
| s = 0.10–0.15 | no | 6 | 0.00 / 0.00 | 0.53–0.59 / 0.66–0.71 | +7.1–8.8 / +6.9–8.7 |
| s = 0.25 | no | 5 | 0.00 / 0.00 | 0.46–0.47 / 0.58–0.59 | +5.5 / +5.4 |
| s = 0.50 | no | 3 | 0.00 / 0.00 | 0.35–0.36 / 0.46–0.47 | +3.2 / +3.0 |
| s = 1.00 | no | **0** | 0.00 / 0.00 | 0.22–0.23 / 0.30–0.31 | +1.6 / +1.5 |

No finding falls outside slots 22–25 in any arm.

- **The equality audit misses every random-noise arm.** Exact-match is 0.00 everywhere, while
  GB gains at least +3.0 points for every s ≤ 0.50.
- **leakcheck's rank-agreement suspicion tracks the exploitable gain under random noise.** It
  fires at every s ≤ 0.50 (gain ≥ +3.0 points) and goes silent at s = 1.0 (gain +1.5 points).
  Compare R1/R3: with value-keyed jitter, suspicions vanish at s ≥ 0.50 while GB still gains
  +12.3. **leakcheck's blind spot is therefore non-monotone recoding, not noise.** A recoding
  that scrambles the order but keeps each value's identity defeats rank agreement, yet a tree
  can still decode it.

### Pre-registered rules, re-read with R5
| Claim | With R5 |
|---|---|
| Abstract: "a noise-perturbed leak retains most of that advantage" | **Narrow it.** With genuine noise at the paper's 0.15 scale, MRR@20 keeps 54% under GB (+7.1 of +13.3 points) and 33% under LR. The true reward keeps 22%. "Most" (96%) holds only for the value-keyed jitter the paper used. That should be described as a deterministic relabelling, or "jittered" defined at first use. |
| "An equality-based audit does not catch it" | **Keep, now stronger.** It holds for value-keyed and random noise alike, at every scale with a gain ≥ 3 points. |
| §6: leakcheck flags the jittered leak | **Keep at 15%.** It holds for both kinds of noise. Add the limitation: under value-keyed recoding, suspicions vanish at s ≥ 0.50 while the leak stays exploitable, whereas under random noise they vanish only once the gain is below 2 points. |

### Caveats
- R5's leakcheck is pooled only; per-dataset suspicions were not computed.
- R5 is classifier-only (Table 2's protocol). There is no R4-style capture and no RL run under random noise.
- Uniform noise only. Other distributions, and noise proportional to the value, were not tested.
