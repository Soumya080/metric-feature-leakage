"""Extract every numeric claim in main.tex, with its sentence, for manual checking.

Six wrong numbers have already been found in this file by hand, all of them inherited
from pre-fix or superseded runs. The failure mode is not arithmetic, it is a number that
was true of an earlier artifact and was never revisited -- which no amount of re-reading
the prose catches, because the prose is fluent either way.

This does not decide correctness. It produces the complete checklist, so "verified" can
mean "every number was looked at" rather than "every number I happened to notice".

Usage:
    python paper/audit_numbers.py              # all claims, grouped by section
    python paper/audit_numbers.py --unchecked  # only those not in the checked list
"""

import argparse
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "main.tex")

# Numbers already traced to a results file during the verification passes. Keyed by the
# distinctive literal as it appears in the tex, so a later edit that changes the value
# drops it out of this set and it shows up as unchecked again.
#
# KNOWN LIMITATION, and it has already bitten. This is a flat set of string literals with no
# notion of WHERE a literal appears. A value that is correct in one sentence therefore marks it
# checked in every sentence. The 2026-08-12 audit found "4.2" -- legitimately registered as
# gradient boosting's single-arm lift -- silently certifying an incorrect "+4.2" in the logistic
# regression paragraph, where the artifact says 3.943. Three further entries ("10.0", "42",
# "8.4") were registered against leak_gradient.json without appearing in it, and "0.801" was
# registered against external_audit_routerbench.json, which contains no Pearson value at all.
# Membership here means "this literal was checked somewhere", NOT "this sentence is right".
# The real fix is to key on (literal, sentence) or to generate this set from the artifacts.
# Until then, treat a green run as a floor, never as verification.
#
# Entries must be STRINGS. re.findall returns strings, so a bare float silently never matches.
CHECKED = {
    # dose-response (results/leak_gradient.json)
    "59.9", "65.1", "69.0", "72.7", "73.1", "60.9", "12.2", "11.8", "8.1", "4.2", "1.0",
    "3{,}175",
    # claim recomputation (results/paper_claim_checks.json)
    "73.4", "64.9", "57",
    # headroom / learnability (results/learnability_analysis.json)
    "26.5", "26--59", "1{,}157", "1{,}539", "916", "1{,}021", "1{,}307", "847",
    "61", "29", "32", "22",
    # per-dataset lifts + two-arm ablation (results/two_arm_ablation.json)
    "3.3", "3.1", "6.4", "2.0", "2.8", "5.0", "53--83", "137--231",
    # capture (results/headroom_analysis.json, results/paper_claim_checks.json)
    # 2026-10-31: "172/174", "228/231" REMOVED -- wrong (GB matches Always-Dense on 169/174
    # and 231/231; results/analyses/paper_number_checks/summary.json gb_test_actions). "45.3" -> "45.2"
    # (convex fusion HotpotQA is -45.25%, paper_claim_checks.json).
    "0.4", "0.1029", "0.1657", "138/138", "7.0", "2.4",
    "174", "231", "138", "21", "35", "46", "30",
    # permutation (results/permutation_control.json)
    "63.4", "60.1", "59.0", "0.048", "61.7", "58.6", "58.2", "0.238",
    "56.4", "50.1", "50.7", "0.810", "59.2", "0.095", "59.1", "0.857", "20",
    # oracle inflation (results/oracle_inflation.json)
    "0.0022", "0.0010", "0.0042", "16--22", "16--29",
    # corpus census (results/corpus_census.json)
    "34.3", "4.59", "27--46",
    # external audit (results/external_audit_routerbench.json).
    # "0.801" REMOVED 2026-08-12: no Pearson coefficient exists in that artifact -- it says only
    # "Pearson agrees". The paper no longer states a value. "87" -> "86": the computed field
    # artifact_audit.n_benchmarks is 86, while the file's own prose summary says 87. We cite the
    # computed field. Confirm against routerbench_0shot.pkl and make the two agree.
    "93.0", "69.9", "40", "0.881", "86",
    # RouterBench leak injection (results/leak_injection_routerbench.json)
    "36{,}497", "2{,}553", "0.94", "30.1", "16.2", "63.7", "92--98", "92.1--97.8",
    "11",
    # permutation controls at B=1000 (results/perm_B1000/permutation_control.json)
    "0.066", "0.090", "0.181", "0.187", "0.740", "0.820", "0.003", "0.002",
    "59.1", "58.2", "50.8", "1000", "0.001", "0.6", "0.5", "0.2",
    # embedding-state control (results/embedding_control.json). Per-dataset lift added to the
    # paper 2026-08-12: embed_pca32 beats hand_61 on ALL THREE datasets (-1.72/-2.27/-3.78 vs
    # -3.25/-3.12/-6.33) and only the pooled fit reverses the order. Reporting the pooled
    # number alone had hidden a counterexample that lives in the same artifact.
    "59.7", "61.7", "52.7", "384", "32",
    "1.7", "3.3", "2.3", "3.1", "3.8", "6.3",
    # seed stability (results/seed_stability.json). Per-seed jitter costs are
    # [0.4094, 0.5543, 0.4472, 0.5228, 0.3465], so the containing interval is [0.34, 0.56];
    # the caption previously said [0.35, 0.55], which excludes both extremes.
    # These were floats before 2026-08-12 and therefore never matched anything.
    "0.22", "0.34", "0.56",
    # QPP control (results/qpp_control_scale100000.json)
    "1.34", "53.5", "54.8", "3{,}614", "1.2", "4.1",
    # estimator sensitivity of the jitter finding (results/leak_gradient.json, logreg columns).
    # RECOMPUTED 2026-08-12 from logreg_lift: exact +9.745, jittered +3.943, hit +8.346,
    # cost 5.802, retention 40.5%, gbt retention 96.6%. The previous line registered
    # "10.0", "4.2", "42", "8.4" -- none of which appear in the artifact.
    "5.8", "9.7", "3.9", "40", "97", "8.3",
    # scale-claim caveats (experiments/corpus_v3/results/leak_vs_scale.json).
    # leak_hit flatness corrected 2026-08-12: flat 20K->50K against honest_minimal
    # (8.77 -> 8.77) but NOT against honest_calibrated (8.40 -> 9.69). The paper said
    # "under either", which is false for the calibrated baseline.
    "63.4", "64.2", "63.7", "8.8", "10.4", "10.3", "2{,}918", "3{,}634", "25", "8.4", "9.7",
    # setup constants, definitional rather than measured
    "22--25", "22--23", "0--21", "26--28", "5", "3", "15", "10", "500", "100",
    # leak-vs-scale, Section 3b (experiments/corpus_v3/results/leak_vs_scale.json)
    "5.5", "5.7", "5.8", "8.8", "10.1", "9.0", "9.6", "10.7", "9.2", "9.5", "10.6",
    "1.6", "1.9", "2{,}918", "3{,}345", "3{,}634", "233", "20K", "50K", "100K",
    # RL algorithm spread: experiments/router_experiments/results/algorithm_comparison.json --
    # a SEPARATE 5-seed / 300-epoch sweep on NQ (REINFORCE 0.1483), NOT the 15-seed / 150-epoch
    # run behind Table 3's RL row (0.1512). The appendix now says so explicitly.
    # Multiple-comparison scope corrected 2026-08-12 to the 15 DISTINCT configurations:
    # alpha = 0.05/15 = 0.0033, not 0.05/45 = 0.0011. The old denominator counted the same 15
    # three times, and per_retriever_router_base.json is a byte-identical copy of
    # per_retriever_router.json.
    "0.0139", "2{,}468", "0.8", "11", "15", "0.0033", "69", "39", "0.021", "0.024",
    "0.036", "0.029", "0.046", "300",
    # Table 3, every method x dataset. All 26 values verified equal to
    # results/canonical_results.json to four decimal places.
    "0.1029", "0.1501", "0.0468", "0.1655", "0.2000", "0.0753", "0.1464", "0.1907",
    "0.0572", "0.1657", "0.1968", "0.1639", "0.1990", "0.0635", "0.1640", "0.1594",
    "0.1981", "0.0750", "0.1642", "0.1512", "0.1881", "0.0683", "0.2001", "0.2694",
    "0.1096", "0.0046", "0.0043", "0.0055",
    # Abstract/Section 4, argmax rebuttal (experiments/corpus_v3/results/leak_vs_scale.json)
    "9.2", "10.6", "0.21", "0.17", "0.12",
    # Section 4, retracted contribution, at full precision so it reconstructs
    "59.40", "60.85", "1.45",
    # ---- Numbers added or corrected in the October 2026 revision ----
    # Table 2 caption: MRR@20 leak equals the reward on 26.3-44.8% of rows
    # (results/analyses/paper_number_checks/summary.json leak_equality).
    "26--45",
    # Section 4.1: value-keyed jitter keeps 92% at s=1.0 (results/analyses/classifier_noise_sweep);
    # independent noise keeps 54% (+7.1 of +13.3), 53-55% over draws, LR 33%, true-reward copy
    # 91.1% (+31.3) keeping 22% (results/analyses/random_noise_sweep/summary.json). MRR@20 has
    # 21 distinct values (0 and 1/1..1/20).
    "92", "54", "7.1", "13.3", "53--55", "33", "91.1", "31.3", "21",
    # Section 4 per-corpus premium (results/learnability_analysis.json; checked in verification pass).
    "11.9", "12.4", "16.8", "13.2",
    # Figure 1 caption: MRR@5 leak zero on 76-95% of rows at 20K/50K/100K (paper_number_checks).
    "76--95",
    # Section 5 / Table 3 (paper_number_checks): GB == Always-Dense counts; capture against the
    # train-selected arm (Hybrid on HotpotQA) 11.9-34.5%; smallest bootstrap p vs Dense 0.094;
    # restored RL row and per-seed capture range from results/analyses/rl_checkpoint_rerun.
    "169/174", "231/231", "12--35", "0.094", "0.1618", "0.0722", "49", "10.8", "9.1",
    "45.2",
    # Section 6 blind spot: value-keyed jitter at s>=0.50 raises no suspicion
    # (results/analyses/leakcheck_noise_sweep) while GB gains +12.25 (classifier_noise_sweep).
    "12.3", "50",
    # Appendix: census over all 1,989 prepared NQ queries; 9,125/682 = 13.4 per contaminated
    # query (results/corpus_census.json). RL settings are canonical_evaluation.py constants.
    "1{,}989", "13.4", "128", "150", "0.01",
    # Setup: Table 1 cells, 5-24% and 62-75% (canonical_results.json, verification pass); retriever
    # constants (src/retrieval/sparse.py, hybrid.py); split ratios (routing_env.py).
    "0.103", "0.166", "0.146", "0.200", "0.150", "0.191", "0.269", "0.047", "0.075", "0.057",
    "0.110", "5--24", "62--75", "1.5", "0.75", "60", "70/15",
    # Related work: verified against the sources in paper/NOVELTY_AUDIT.md; 2.5 is verbatim from
    # the RetrievalRouter abstract (arXiv:2608.25625, checked 2026-10-31).
    "17", "294", "2{,}047", "1.75", "2018", "94.9", "2.5",
    # Recomputed in verification pass but never registered: MDE and two-arm majorities
    # (results/two_arm_ablation.json), headroom shares (headroom_analysis.json), candidate-set
    # config (results/corpus_census.json _config), RouterBench m values, and definitional
    # constants (alpha, power).
    "53--69", "65--83", "69.6", "63.0", "70", "63", "174/231", "21/35", "26/30",
    "100{,}000", "499", "4", "8", "0.05", "80", "0.0",
    # Table 2 seed ranges (results/seed_stability.json, verification pass), headroom 59
    # (learnability_analysis.json), per-seed RL capture max +8.6 -> "+9" (rl_checkpoint_rerun).
    "0.15", "0.20", "0.13", "59", "9",
    # Not claims: the review URL's suffix and the "August 2026" date.
    "251", "2026",
    # Appendix A provenance: commit hashes and dates from `git log` (4d2dade 2026-07-23,
    # eaa39e7 2026-08-02, bab7e48 2026-08-05, 17577c2 2026-08-05).
    "23", "17577",
    # Appendix positive control (results/analyses/positive_control/summary.json): honest lift
    # -0.4 / +14.0 / +17.4 at alpha 0 / 0.5 / 1.0; random-noise leak adds 47.9-14.0 = 33.9 and
    # 43.2-17.4 = 25.8 -> "26--34". Figure 11 from results/analyses/detection_power/table.json.
    "14.0", "17.4", "26--34",
}

SECTION_RE = re.compile(r"\\section\{([^}]*)\}")
# Numbers that carry a claim: percentages, decimals, ratios, thousands-separated ints.
NUM_RE = re.compile(
    r"(?<![\w.])("
    r"\d+(?:\{,\}\d{3})+"        # 1{,}157
    r"|\d+/\d+"                   # 172/174
    r"|\d+(?:\.\d+)?--\d+(?:\.\d+)?"  # 26--59
    r"|\d+\.\d+"                  # 0.0022
    r"|\d+"                       # 61
    r")"
)


def sentences(block: str):
    text = " ".join(l for l in block.splitlines() if not l.lstrip().startswith("%"))
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?]) +", text) if s.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unchecked", action="store_true",
                    help="show only numbers not in the verified set")
    args = ap.parse_args()

    tex = io.open(TEX, encoding="utf-8").read()
    parts = SECTION_RE.split(tex)
    # parts = [preamble, name1, body1, name2, body2, ...]
    chunks = [("Abstract/preamble", parts[0])] + [
        (parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)
    ]

    total = unchecked_n = 0
    for name, body in chunks:
        rows = []
        for sent in sentences(body):
            nums = NUM_RE.findall(sent)
            # Skip LaTeX plumbing: widths, column specs, label numbers.
            if not nums or re.match(r"^\s*\\(begin|end|includegraphics|label|toprule|"
                                    r"midrule|bottomrule|centering|small)", sent):
                continue
            claim_nums = [n for n in nums if n not in {"0", "1", "2"}]
            if not claim_nums:
                continue
            new = [n for n in claim_nums if n not in CHECKED]
            if args.unchecked and not new:
                continue
            total += len(claim_nums)
            unchecked_n += len(new)
            marker = "NEW" if new else " ok"
            display = sent if len(sent) <= 150 else sent[:147] + "..."
            rows.append(f"  [{marker}] {', '.join(claim_nums)}\n        {display}")
        if rows:
            print(f"\n=== {name} ===")
            print("\n".join(rows))

    print(f"\n{'=' * 70}")
    print(f"numeric claims surfaced: {total}   not yet traced to a results file: {unchecked_n}")
    if unchecked_n:
        print("Rerun with --unchecked to see only those.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
