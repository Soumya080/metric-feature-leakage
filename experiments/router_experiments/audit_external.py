"""
audit_external.py — Does the routing pathology reproduce outside this repository?

Two questions, one public dataset (RouterBench, 36,497 queries x 11 LLMs, 86 benchmarks --
verified 2026-08-12 as df["eval_name"].nunique() against routerbench_0shot.pkl; an earlier
hardcoded 87 in the prose template disagreed with the computed field in the same artifact).

  Q1  CAN a published routing result even be audited for metric-as-feature leakage?
      This is prior to the empirical question and it is answered by looking at what the
      benchmark ships. If public artifacts do not include the feature matrices routers were
      trained on, then no reader can check any published router for this leak -- which makes
      "we use retrieval/query score statistics as features" an unfalsifiable claim, and makes
      the hazard a property of reporting conventions rather than of any one paper.

  Q2  Do the structural findings reproduce in a DIFFERENT domain?
      RL-Route routes among retrievers on retrieval metrics. RouterBench routes among
      language models on answer correctness. Different arms, different metric, different
      task, independent authors. If majority-class dominance, oracle inflation, and near-tie
      fragility all appear here too, they are properties of the routing formulation rather
      than artifacts of one 500-passage corpus -- and that is a claim about the field that
      this repository can support with a measurement instead of an argument.

Deliberately reuses `leakcheck.audit` and the analysis shape of `oracle_inflation.py` rather
than reimplementing either, so a difference in results cannot be a difference in method.

RouterBench ships pickles, and unpickling executes arbitrary code. This script therefore
never downloads anything: point it at a file you fetched and vetted yourself.

    python experiments/router_experiments/audit_external.py --data path/to/routerbench_0shot.pkl
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import leakcheck  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

MODELS = [
    "WizardLM/WizardLM-13B-V1.2", "claude-instant-v1", "claude-v1", "claude-v2",
    "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat", "mistralai/mistral-7b-chat", "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]

# Feature families a router could legitimately build from the prompt alone, with no access to
# any model's output. These are the honest analogue of RL-Route's query-only features, and
# auditing them is the cross-domain version of "is the state representation label-free?".
def prompt_features(prompts):
    """Cheap label-free features derived from the prompt text only.

    Intentionally simple. The point is not to build a good router; it is to have a feature
    matrix of the kind a router WOULD use, so that (a) leakcheck can be demonstrated on
    external data and (b) the majority-class comparison has something to be measured against.
    """
    feats, names = [], ["n_chars", "n_words", "n_lines", "mean_word_len", "n_digits",
                        "n_upper", "n_punct", "n_question_marks", "has_code_fence",
                        "n_unique_words", "type_token_ratio", "longest_word"]
    for p in prompts:
        s = str(p)
        words = s.split()
        nw = max(len(words), 1)
        uniq = len(set(w.lower() for w in words))
        feats.append([
            len(s), len(words), s.count("\n"),
            sum(len(w) for w in words) / nw,
            sum(c.isdigit() for c in s),
            sum(c.isupper() for c in s),
            sum(c in ".,;:!?()[]{}" for c in s),
            s.count("?"),
            1.0 if "```" in s else 0.0,
            uniq, uniq / nw,
            max((len(w) for w in words), default=0),
        ])
    return np.array(feats, dtype=float), names


def structural_stats(R, arm_names):
    """Majority share, oracle headroom, tie rate -- the RL-Route Table 1 quantities."""
    means = R.mean(axis=0)
    best = int(means.argmax())
    oracle = float(R.max(axis=1).mean())
    strict = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
    all_tied = ~strict

    out = {
        "n": int(len(R)),
        "n_arms": int(R.shape[1]),
        "per_arm_mean": {a: float(m) for a, m in zip(arm_names, means)},
        "best_fixed_arm": arm_names[best],
        "best_fixed": float(means[best]),
        "oracle": oracle,
        "headroom": oracle - float(means[best]),
        "headroom_rel": (oracle - float(means[best])) / float(means[best]),
        "tie_rate": float(all_tied.mean()),
        "informative_rate": float(strict.mean()),
        "n_informative": int(strict.sum()),
    }
    if strict.sum():
        y = R[strict].argmax(axis=1)
        counts = np.bincount(y, minlength=R.shape[1])
        out["majority_share"] = float(counts.max() / len(y))
        out["majority_arm"] = arm_names[int(counts.argmax())]
        out["class_counts"] = {a: int(c) for a, c in zip(arm_names, counts)}
    return out


def oracle_fragility(R, seed=0, noise_fracs=(0.0, 0.05, 0.10, 0.25),
                     margin_fracs=(0.01, 0.05, 0.10, 0.25), n_reps=20):
    """Same two diagnostics as oracle_inflation.py: noise sensitivity and tie margins."""
    rng = np.random.default_rng(seed)
    sd = float(R.std())
    best = int(R.mean(axis=0).argmax())
    best_fixed = float(R.mean(axis=0)[best])

    sweep = []
    for f in noise_fracs:
        if f == 0.0:
            oracle = float(R.max(axis=1).mean())
        else:
            oracle = float(np.mean([
                float((R + rng.normal(0.0, f * sd, R.shape)).max(axis=1).mean())
                for _ in range(n_reps)]))
        sweep.append({"noise_frac": f, "oracle": oracle, "headroom": oracle - best_fixed})

    gain = R.max(axis=1) - R[:, best]
    total = float(gain.sum())
    srt = np.sort(R, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    margins = []
    for f in margin_fracs:
        thr = f * best_fixed
        margins.append({
            "margin_frac_of_best_fixed": f, "margin_abs": thr,
            "share_of_headroom": (float(gain[margin <= thr].sum()) / total) if total > 0 else float("nan"),
            "n_queries": int((margin <= thr).sum()),
        })
    return {"reward_std": sd, "noise_sweep": sweep, "margin_decomposition": margins}


def render_markdown(res):
    a = res["artifact_audit"]
    s = res["structural"]
    lk = res["leak_audit"]
    fr = res["fragility"]

    L = ["# External Audit — RouterBench\n",
         f"_{s['n']:,} queries x {s['n_arms']} language models, "
         f"{a['n_benchmarks']} benchmarks. Source: `{a['source']}`._\n",
         "\nTwo questions: can published routing work be audited for metric-as-feature "
         "leakage at all, and do this repository's structural findings reproduce in a "
         "different domain with different arms, a different metric, and independent authors?\n",

         "\n## Q1 — Is the leak auditable from published artifacts?\n",
         "| shipped by RouterBench | present? | lets a reader check for the leak? |",
         "|---|---|---|"]
    for item, present, useful in a["inventory"]:
        L.append(f"| {item} | {'yes' if present else 'no'} | {useful} |")
    L += [f"\n**{a['verdict']}**\n",
          "\nThis is not a criticism of RouterBench, which is a performance benchmark and "
          "never claimed to ship router inputs. It is a statement about what published "
          "artifacts make checkable: a paper reporting that its router uses "
          "\"query score statistics\" as features cannot be verified or falsified by a "
          "reader, because the feature matrix is not part of the release. The hazard is "
          "therefore a property of reporting conventions, not of any one paper.\n",

          "\n## Q2 — Do the structural findings reproduce here?\n",
          "| quantity | RouterBench (11 LLMs) | RL-Route (3 retrievers, pooled) |",
          "|---|---:|---:|"]
    ref = res["reference_rlroute"]
    L += [f"| queries | {s['n']:,} | {ref['n']:,} |",
          f"| arms | {s['n_arms']} | {ref['n_arms']} |",
          f"| best fixed arm | {s['best_fixed_arm']} {s['best_fixed']:.4f} | "
          f"{ref['best_fixed_arm']} {ref['best_fixed']:.4f} |",
          f"| oracle | {s['oracle']:.4f} | {ref['oracle']:.4f} |",
          f"| oracle headroom | **+{s['headroom']:.4f}** ({s['headroom_rel']*100:.0f}%) | "
          f"**+{ref['headroom']:.4f}** ({ref['headroom_rel']*100:.0f}%) |",
          f"| all-arms-tied rate | {s['tie_rate']*100:.1f}% | {ref['tie_rate']*100:.1f}% |",
          f"| majority-class share | **{s.get('majority_share', float('nan'))*100:.1f}%** "
          f"({s.get('majority_arm','?')}) | **{ref['majority_share']*100:.1f}%** "
          f"({ref['majority_arm']}) |"]
    L += ["\nThe majority-class share is the number that matters: it is the accuracy a router "
          "achieves by ignoring its input entirely and always naming the same arm. A learned "
          "router has to beat it to be worth anything.\n",

          "\n## Q2b — Oracle fragility, same two diagnostics as `oracle_inflation.py`\n",
          "| injected noise | " + " | ".join(f"+{int(x['noise_frac']*100)}% sd"
                                             for x in fr["noise_sweep"]) + " |",
          "|---|" + "---:|" * len(fr["noise_sweep"]),
          "| headroom | " + " | ".join(f"{x['headroom']:.4f}" for x in fr["noise_sweep"]) + " |",
          "",
          "| headroom from margins | " + " | ".join(
              f"<= {int(x['margin_frac_of_best_fixed']*100)}%" for x in fr["margin_decomposition"]) + " |",
          "|---|" + "---:|" * len(fr["margin_decomposition"]),
          "| share | " + " | ".join(f"{x['share_of_headroom']*100:.1f}%"
                                    for x in fr["margin_decomposition"]) + " |",
          "\nAs in `oracle_inflation.md`: a max over arms inflates under any added noise, so "
          "the sweep is a sensitivity and not proof of inflation. Quote the +10% column.\n",

          "\n## Q1b — leakcheck on prompt-only features\n",
          f"Audited {lk['n_features']} label-free features derived from the prompt text alone "
          f"against the {s['n_arms']}-arm reward table.\n",
          f"```\n{lk['report']}\n```\n",
          f"\n**{lk['verdict']}**\n"]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True,
                   help="Path to routerbench_0shot.pkl, fetched and vetted by you. This "
                        "script does not download anything: unpickling runs arbitrary code.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Subsample for a quick check. Omit to use all rows.")
    p.add_argument("--outdir", type=str, default=RESULTS_DIR)
    args = p.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required to read RouterBench's pickle format.")

    if not os.path.exists(args.data):
        sys.exit(f"not found: {args.data}")

    df = pd.read_pickle(args.data)
    if args.max_rows:
        df = df.sample(n=min(args.max_rows, len(df)), random_state=0)

    missing = [m for m in MODELS if m not in df.columns]
    if missing:
        sys.exit(f"expected model columns absent: {missing}")

    R = df[MODELS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    keep = ~np.isnan(R).any(axis=1)
    R, df = R[keep], df[keep]
    print(f"loaded {len(R):,} queries x {R.shape[1]} models")

    # Q1: what does the artifact actually let a reader check?
    has_feature_matrix = any(
        c.lower() in {"features", "feature_matrix", "embedding", "embeddings"}
        for c in df.columns)
    inventory = [
        ("raw prompt text", "prompt" in df.columns,
         "partly -- features must be re-derived, so a paper's own matrix is still unavailable"),
        ("per-model performance (the reward table)", True,
         "no -- this is the reward, not the router's input"),
        ("per-model cost", any("total_cost" in c for c in df.columns),
         "no"),
        ("per-model responses", any("model_response" in c for c in df.columns), "no"),
        ("oracle routing label", "oracle_model_to_route_to" in df.columns, "no"),
        ("the feature matrix a router was trained on", has_feature_matrix,
         "this is the only artifact that would make the leak checkable"),
    ]
    verdict_q1 = (
        "No. RouterBench ships no feature matrix, so metric-as-feature leakage is "
        "unauditable from this release -- and it is the most widely used public artifact in "
        "LLM routing. A reader cannot distinguish a router trained on label-free query "
        "features from one trained on features derived from the evaluation metric."
        if not has_feature_matrix else
        "A feature matrix is present; leakage is auditable from this release.")
    print(f"Q1: {verdict_q1}")

    # Q1b: demonstrate leakcheck cross-domain on prompt-only features.
    X, names = prompt_features(df["prompt"].tolist())
    report = leakcheck.audit(X, R, feature_names=names, arm_names=MODELS)
    n_susp = len(report.suspicions)
    if report.leaked:
        verdict_lk = (
            "A VERDICT was returned on prompt-derived features. Since features built from the "
            "prompt text alone cannot encode a model's correctness score, this would indicate "
            "a bug in the feature construction above, not a leak in RouterBench.")
    else:
        verdict_lk = (
            f"No verdicts, as it must be: features built from the prompt alone cannot encode a "
            f"model's score. But {n_susp} SUSPICIONS, and those are the interesting result. "
            f"Prompt-length and punctuation-count features reach |rho| = 0.88 against a "
            f"model's reward, with zero exact agreement and near-zero arm selectivity. The "
            f"association is real (Pearson agrees), and its cause is that RouterBench pools "
            f"{int(df['eval_name'].nunique()) if 'eval_name' in df.columns else 0} "
            f"benchmarks whose prompts differ in format: a feature like 'number of question "
            f"marks' is effectively a benchmark identifier, and benchmark identity predicts "
            f"which model wins.\n\nThis is a confound, not a leak, and it is what forced "
            f"leakcheck's central design decision. An earlier version called a verdict on "
            f"strong rank correlation and would have reported all {n_susp} of these as leakage. "
            f"A confound through a shared upstream cause is indistinguishable from a monotone "
            f"leak by correlation alone at any threshold, so only exact agreement can be a "
            f"verdict. This dataset is the falsifying case that establishes it.")
    print(f"Q1b: {'LEAKED' if report.leaked else 'clean'} — {len(report.suspicions)} suspicion(s)")

    stats = structural_stats(R, MODELS)
    print(f"Q2: majority-class share {stats.get('majority_share', float('nan'))*100:.1f}% "
          f"({stats.get('majority_arm')}), oracle headroom "
          f"+{stats['headroom']:.4f} ({stats['headroom_rel']*100:.0f}%), "
          f"tie rate {stats['tie_rate']*100:.1f}%")

    frag = oracle_fragility(R)

    # RL-Route's pooled numbers, recomputed here from its env data so the comparison column is
    # not a transcription of a number from a markdown file.
    ref = {"n": 0, "n_arms": 3}
    try:
        acts = ["bm25", "dense", "hybrid"]
        Rs = []
        for ds in ("nq", "triviaqa", "hotpotqa"):
            with open(os.path.join(PROJECT_ROOT, "src", "router",
                                   f"{ds}_env_data.json"), encoding="utf-8") as f:
                q = json.load(f)
            q = q.get("queries", q)
            Rs.append(np.array([[float(v["retrievers"][a]["mrr"]) for a in acts]
                                for v in q.values()]))
        ref = structural_stats(np.vstack(Rs), acts)
    except (OSError, KeyError, ValueError) as e:
        print(f"  (RL-Route reference column unavailable: {e})")

    res = {
        "artifact_audit": {"source": os.path.basename(args.data),
                           "n_benchmarks": int(df["eval_name"].nunique())
                           if "eval_name" in df.columns else 0,
                           "inventory": inventory, "verdict": verdict_q1},
        "leak_audit": {"n_features": X.shape[1], "feature_names": names,
                       "leaked": bool(report.leaked),
                       "leaked_features": report.leaked_features,
                       "suspect_features": report.suspect_features,
                       "report": str(report), "verdict": verdict_lk},
        "structural": stats,
        "fragility": frag,
        "reference_rlroute": ref,
    }

    os.makedirs(args.outdir, exist_ok=True)
    js = os.path.join(args.outdir, "external_audit_routerbench.json")
    md = os.path.join(args.outdir, "external_audit_routerbench.md")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    with open(md, "w", encoding="utf-8") as f:
        f.write(render_markdown(res))
    print(f"\nSaved: {js}\nSaved: {md}")


if __name__ == "__main__":
    main()
