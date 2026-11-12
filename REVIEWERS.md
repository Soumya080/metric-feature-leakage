# Start here (reviewers)

Everything below runs on a laptop. No GPU, no downloads, no API keys.
Total time for all three steps: **about four minutes**, nearly all of it in step 2.

---

## 1. Install

```bash
pip install -e ".[test]"
```

Installs `leakcheck` (numpy-only at its core) plus scikit-learn and pytest for the demo and
the test suite. Python 3.8 or newer.

---

## 2. Reproduce the paper's central mechanism — one command, about three minutes

```bash
python examples/leak_demo.py
```

This builds a routing problem in which **the best arm is chosen independently of the
features**, so the target is unlearnable by construction. Any accuracy above the
majority-class rate can therefore only have come from leakage the script introduces on
purpose. It then injects each rung of the taxonomy into a single feature column and refits
the same classifier.

What to look for: the **direct vs. jittered** rows. Both leak the same arms into the same
columns; the only difference is the perturbation. The jittered leak still retains most of the
advantage. This is the paper's mechanism (Figure 1) reproduced on synthetic data from nothing,
not the paper's figure itself (for that, see step 2b), and it is why an accidental leak is
dangerous — it does not need to be a clean copy of the reward, and an equality check will not
find it.

The `honest` row is the control: it sits at or below the majority rate, which is what
guarantees everything above it is leakage rather than real signal.

### 2b. Rebuild the paper's Figure 1 from committed rows — about 30 seconds

```bash
cd experiments/crc_v2
python r9_leak_capture_summary.py --out-dir <new dir>   # panel (b): aggregates committed capture rows, fits nothing
python fig12_leak_ladder.py                             # writes results/figures/fig12_leak_ladder.{pdf,png}
```

Panel (a)'s accuracies are read from `results/leak_gradient.json`;
`python experiments/router_experiments/leak_gradient.py` recomputes them from the environment
files (slower). The "equality check" labels come from
`results/crc_v2/r10_review3_checks/summary.json` (`python r10_review3_checks.py --out-dir <new dir>`).

---

## 3. Check the detector works

```bash
python -m pytest leakcheck/tests/ -q
```

Expected: **32 passed** (leakcheck 0.2.1), about 15–80 seconds depending on the machine.
`python -m pytest tests/test_no_feature_reward_leak.py -q` adds the 2 pipeline leak-guard tests.

To run it on your own arrays:

```python
import leakcheck, numpy as np

X = ...          # (n_queries, n_features) feature matrix
R = ...          # (n_queries, n_arms)     per-arm reward table

print(leakcheck.audit(X, R))
```

`audit` returns a `Report`. It issues an automatic **verdict** only on exact agreement, which
is a fact about the arrays rather than a judgement: on more than 25% of the rows where the
reward is non-zero, or on every row for a binary reward (see `leakcheck/README.md`, "What a
clean report does and does not mean"). Rank-correlation findings are reported as
**suspicions carrying their evidence**, never as verdicts — §6 of the paper explains why, and
`leakcheck/CALIBRATION.md` records the external audit that falsified the first design.

**The tool is triage, not a certificate of cleanliness.** Separating a confound from a leak
requires dropping the column and refitting; that needs the pipeline, not the arrays.

---

## 4. Read the committed numbers without re-running anything

```bash
cat results/leak_gradient.md            # Figure 1(a), the leak ladder in accuracy
cat results/crc_v2/r9_leak_capture_summary/summary.json   # Figure 1(b), the ladder in capture
cat results/crc_v2/r6_testsplit_power/summary.json        # minimum detectable capture on the test splits (68/43/72%)
cat results/crc_v2/r10_review3_checks/summary.json        # shared-index arms, MDE bracket, p-values, leaked-router test
cat results/two_arm_ablation.md         # two-arm ablation (its MDE section is superseded by r6)
cat results/perm_B1000/permutation_control.md   # both null controls, B=1000
cat results/external_audit_routerbench.md       # the RouterBench transplant
cat results/KNOWN_ISSUES.md             # defects we found in our own artifacts
```

`REPRODUCE.md` maps **every** claim in the paper to the command that regenerates it. If a
number in the paper is not reachable from that table, it should not be in the paper.

---

## What this repository is not

The full pipeline (corpus construction, retrieval, RL training) is here under `experiments/`,
but reproducing it end to end needs a GPU and several hours, and it downloads Wikipedia
passages. **None of that is necessary to check the paper's claims.** Steps 2–4 above verify
the mechanism, the detector, and every reported number.

The Wikipedia passage pool is deliberately **not** redistributed. It is CC BY-SA text, and it
regenerates deterministically from its original source at a fixed seed. See `DATA_LICENSES.md`
for what is under which licence — the code is MIT, anything containing question, answer or
passage text is CC BY-SA 4.0.

---

## Was the original leak an accident?

The paper calls the leak an accidental bug. The commit history of the authors' working
repository shows the order of events (dates 2026; this anonymous mirror carries no history, so
this table is the record):

| Commit | Date | Event |
|---|---|---|
| `4d2dade` | 23 Jul | `prepare_env_data.py` reads `metrics["mrr"]` into the score features, with the comment "Use MRR at k as a proxy for retriever confidence scores" |
| `15fb214` | 2 Aug | Diagnosis: "the previously reported routing gains were reward leakage, not routing skill" |
| `bab7e48` | 5 Aug, 09:43 | Regression test against the leak (`tests/test_no_feature_reward_leak.py`) |
| `17577c2` | 5 Aug, 10:47 | The graded leak arms (the ladder), built after the bug was found and guarded |

The ladder's rungs are deliberate, as in any controlled experiment; the original leak was not.
