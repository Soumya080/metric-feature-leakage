"""
03_pilot_negatives.py — Decide the negative-sampling strategy from data, not a guess.

Why this exists
---------------
Until 2026-08-02, `01_prepare_data.py` sampled its 499 distractors uniformly at random,
while the paper and `README.md` both described them as "hard negatives".
Fixing that claim means actually mining topically-related negatives — but hardness is a
knob with a cliff on both ends, and this project's headline claim is a *gap*, not a level:

  * Too easy (uniform random): any passage sharing a query term stands out, so BM25 and
    Dense both find the gold and AGREE. Agreement collapses routing headroom.
  * Too hard (all mined, high fraction): every retriever fails on every query, MRR goes to
    ~0 for all three, and headroom collapses again — from the other direction.

Both failure modes destroy the paper's finding, and discovering either after a multi-hour
cluster run is expensive. So: regenerate a small sample under several strategies, run real
retrieval on it, and report oracle-vs-best-fixed headroom for each. Pick the winner, then
commit the full run.

What it measures
----------------
For each strategy, per dataset: mean MRR@5 per retriever, best-fixed MRR, oracle MRR, and
HEADROOM = oracle - best_fixed (the quantity RL router exists to capture). It also reports
headroom restricted to `real_wikipedia` gold queries, since those are what the headline
results filter to (see prepare_env_data.py --gold-type).

Usage
-----
    # Default: 200 queries/dataset, four strategies. Read-only w.r.t. the real data —
    # writes only into experiments/baseline_v2/pilot/.
    python experiments/baseline_v2/03_pilot_negatives.py

    python experiments/baseline_v2/03_pilot_negatives.py --num_questions 100 --datasets nq
    python experiments/baseline_v2/03_pilot_negatives.py --strategies random,mixed

This never touches experiments/baseline_v2/data/ or results/ — run it before the full
pipeline, compare, then set --negative_strategy / --hard_negative_fraction on
01_prepare_data.py accordingly.
"""

import argparse
import importlib.util
import json
import os
import random
import statistics
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

PILOT_DIR = os.path.join(PROJECT_ROOT, "experiments", "baseline_v2", "pilot")
CACHE_DIR = os.path.join(PROJECT_ROOT, "experiments", "baseline_v2", "cache")

# 01_prepare_data.py starts with a digit, so it cannot be imported by name.
_PREP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_prepare_data.py")
_spec = importlib.util.spec_from_file_location("prepare_data_mod", _PREP_PATH)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)

# Reuse 02_run_baselines.py's own retrieval + metric code rather than reimplementing it,
# so the pilot measures exactly what the real pipeline measures. (Same digit-prefixed
# filename problem, same importlib workaround.)
_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_run_baselines.py")
_bspec = importlib.util.spec_from_file_location("run_baselines_mod", _BASE_PATH)
baselines = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(baselines)

RETRIEVERS = ["bm25", "dense", "hybrid"]

# (label, strategy, hard_fraction) — spans "what we had" to "probably too hard".
DEFAULT_STRATEGIES = [
    ("random (current)", "random", 0.0),
    ("mixed-25%", "mixed", 0.25),
    ("mixed-50%", "mixed", 0.50),
    ("hard-100%", "hard", 1.0),
]

DATASET_LOADERS = {
    "nq": ("google-research-datasets/nq_open", None, "validation", 42),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", "validation", 43),
    "hotpotqa": ("hotpotqa/hotpot_qa", "fullwiki", "validation", 44),
}


def build_pilot_data(ds_name, dataset, wiki_pool, word_index, n_questions, seed,
                     strategy, hard_fraction):
    """Regenerate a small prepared-data sample under one negative-sampling strategy.

    Mutates 01_prepare_data.py's module-level knobs, which is how that script is
    parameterised. Restored by the caller between strategies.
    """
    prep.NEGATIVE_STRATEGY = strategy
    prep.HARD_NEGATIVE_FRACTION = hard_fraction
    data, stats = prep.prepare_dataset(
        ds_name, dataset, wiki_pool, word_index, n_questions, seed
    )
    return data, stats


def evaluate(prepared, encoder, k=5):
    """Run all three retrievers over a prepared sample; return per-query MRR@k.

    Delegates to 02_run_baselines.run_single_query_all_retrievers and its compute_mrr, so
    any change to the real retrieval/matching logic is automatically reflected here.
    """
    from tqdm import tqdm

    per_query = []
    for item in tqdm(prepared, desc="    retrieving", leave=False):
        res = baselines.run_single_query_all_retrievers(
            item["question"], item["corpus"], encoder, top_k=k
        )
        scores = {
            name: baselines.compute_mrr(res[name]["retrieved"][:k], item["answers"])
            for name in RETRIEVERS
        }
        per_query.append({"gold_type": item["gold_type"], "scores": scores})
    return per_query


def summarize(per_query, subset=None):
    rows = per_query if subset is None else [q for q in per_query if q["gold_type"] == subset]
    if not rows:
        return None
    means = {r: statistics.mean(q["scores"][r] for q in rows) for r in RETRIEVERS}
    best_name = max(means, key=means.get)
    best_fixed = means[best_name]
    oracle = statistics.mean(max(q["scores"].values()) for q in rows)

    # Signal DENSITY, not just size. Aggregate headroom can look healthy while almost every
    # query is unlearnable: if all three retrievers score 0 the actions are tied, the
    # gradient is zero, and argmax-based supervised labels degenerate into a tie-break
    # artifact (numpy breaks ties toward index 0, i.e. always BM25). Only queries with a
    # strictly-best action carry routing signal, so that rate is what actually bounds how
    # much any router can learn — and picking a negative-sampling strategy on headroom
    # alone, as the first version of this script did, can silently trade it away.
    all_zero = sum(1 for q in rows if max(q["scores"].values()) == 0.0)
    unique_best = sum(
        1 for q in rows
        if list(q["scores"].values()).count(max(q["scores"].values())) == 1
    )
    n = len(rows)
    return {
        "n": n,
        "per_retriever": means,
        "best_fixed": best_fixed,
        "best_fixed_name": best_name,
        "oracle": oracle,
        "headroom": oracle - best_fixed,
        "headroom_pct": (oracle - best_fixed) / best_fixed * 100 if best_fixed > 0 else 0.0,
        "all_zero_rate": all_zero / n,
        "unique_best_rate": unique_best / n,
        "n_learnable": unique_best,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_questions", type=int, default=200,
                        help="Queries per dataset per strategy (default 200). Small on "
                             "purpose — this is a direction check, not a result.")
    parser.add_argument("--datasets", type=str, default="nq,triviaqa,hotpotqa")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated subset of: random,mixed-25%%,mixed-50%%,hard-100%%")
    parser.add_argument("--output", type=str,
                        default=os.path.join(PILOT_DIR, "pilot_negatives.json"))
    args = parser.parse_args()

    os.makedirs(PILOT_DIR, exist_ok=True)
    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]

    strategies = DEFAULT_STRATEGIES
    if args.strategies:
        keep = {s.strip().lower() for s in args.strategies.split(",")}
        strategies = [s for s in DEFAULT_STRATEGIES
                      if s[0].lower() in keep or s[1].lower() in keep]
        if not strategies:
            print(f"No strategies matched {args.strategies!r}. Options: "
                  f"{[s[0] for s in DEFAULT_STRATEGIES]}")
            sys.exit(1)

    print("=" * 74)
    print("PILOT: negative-sampling strategy vs. oracle routing headroom")
    print(f"{args.num_questions} queries/dataset | strategies: {[s[0] for s in strategies]}")
    print("=" * 74)

    wiki_pool = prep.load_wiki_pool(os.path.join(CACHE_DIR, "wiki_passage_pool.json"))
    word_index = prep.build_word_index(wiki_pool)

    from datasets import load_dataset
    from src.embeddings.encoder import EmbeddingEncoder

    print("\nLoading embedding model...")
    encoder = EmbeddingEncoder()

    results = {}
    t0 = time.time()

    for ds_name in wanted:
        if ds_name not in DATASET_LOADERS:
            print(f"Unknown dataset {ds_name!r}, skipping.")
            continue
        path, config, split, seed = DATASET_LOADERS[ds_name]
        print(f"\nLoading {ds_name}...")
        dataset = load_dataset(path, config, split=split) if config else \
            load_dataset(path, split=split)

        results[ds_name] = {}
        for label, strategy, frac in strategies:
            print(f"\n--- {ds_name} | {label} ---")
            random.seed(seed)
            prepared, stats = build_pilot_data(
                ds_name, dataset, wiki_pool, word_index,
                args.num_questions, seed, strategy, frac,
            )
            per_query = evaluate(prepared, encoder)
            entry = {
                "strategy": strategy,
                "hard_fraction": frac,
                "prep_stats": stats,
                "all": summarize(per_query),
                "real_wikipedia": summarize(per_query, subset="real_wikipedia"),
            }
            results[ds_name][label] = entry
            a = entry["all"]
            print(f"    all (n={a['n']}): best-fixed={a['best_fixed_name']}:"
                  f"{a['best_fixed']:.3f} oracle={a['oracle']:.3f} "
                  f"HEADROOM={a['headroom']:+.3f} ({a['headroom_pct']:.0f}%)")
            r = entry["real_wikipedia"]
            if r:
                print(f"    real_wikipedia (n={r['n']}): best-fixed="
                      f"{r['best_fixed_name']}:{r['best_fixed']:.3f} "
                      f"oracle={r['oracle']:.3f} HEADROOM={r['headroom']:+.3f} "
                      f"({r['headroom_pct']:.0f}%)")

    print("\n" + "=" * 74)
    print("SUMMARY — headroom on real_wikipedia queries (what the headline run uses)")
    print("=" * 74)
    print(f"{'dataset':<10} {'strategy':<18} {'best-fix':>9} {'oracle':>7} {'headroom':>9} "
          f"{'all-zero':>9} {'learnable':>10}")
    for ds_name, by_strategy in results.items():
        for label, entry in by_strategy.items():
            r = entry["real_wikipedia"] or entry["all"]
            print(f"{ds_name:<10} {label:<18} {r['best_fixed']:>9.3f} {r['oracle']:>7.3f} "
                  f"{r['headroom']:>+9.3f} {r['all_zero_rate']:>8.1%} "
                  f"{r['unique_best_rate']:>9.1%}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {args.output}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")
    print("\nPick on BOTH columns, not headroom alone:")
    print("  - headroom  : how much routing could gain if it routed perfectly")
    print("  - learnable : share of queries with a strictly-best action. Everything else is")
    print("                a tie (usually all-zero), contributing no gradient and degrading")
    print("                argmax-based supervised labels into a tie-break artifact.")
    print("A strategy with big headroom but few learnable queries gives a router nothing to")
    print("train on. Prefer the best headroom among strategies that keep 'learnable' high.")
    print("If 'random' wins on both, keep random and fix the wording in")
    print("the paper and README.md instead of the data.")
    print("Then: python experiments/baseline_v2/01_prepare_data.py "
          "--negative_strategy <s> --hard_negative_fraction <f>")


if __name__ == "__main__":
    main()
