"""Grow the cached Wikipedia pool by APPENDING, so the old and new runs stay comparable.

Why this exists rather than just raising WIKI_POOL_SIZE. 01_prepare_data.py builds the
pool with

    indices = random.sample(range(len(wiki)), min(WIKI_POOL_SIZE * 2, len(wiki)))

so changing WIKI_POOL_SIZE draws a completely different sample. And --wiki_pool_size only
takes effect when the cache is absent, which means the obvious route -- delete the cache,
re-run -- replaces the pool wholesale. A query that found a real Wikipedia gold passage at
20K might not find one at 100K, and vice versa, so the two runs would differ in pool size
AND pool identity at once and nothing could be attributed to either.

Appending keeps the original 20,000 passages at indices 0..19,999. That buys three things:

  * gold assignment is MONOTONE -- a query whose gold was found in the old pool still finds
    it in the new one, so queries can only flip constructed -> real_wikipedia, never back
  * a matched subset exists (queries with real gold in BOTH runs), which separates the
    effect of harder negatives from the effect of gold type
  * the pool-size axis becomes a dose-response curve rather than two unrelated points

What it does NOT hold fixed: negatives get harder, because find_hard_negatives mines a
larger pool for topical overlap. Expect absolute MRR to fall across the board. That is the
intended change; compare relative quantities (headroom, lift, headroom capture) and let the
matched subset attribute the rest.

Usage:
    python experiments/baseline_v2/append_wiki_pool.py --target 100000
    python experiments/baseline_v2/append_wiki_pool.py --target 100000 --dry-run
"""

import argparse
import json
import os
import random
import shutil
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_PATH = os.path.join(PROJECT_ROOT, "experiments", "baseline_v2", "cache",
                          "wiki_passage_pool.json")

# Mirrors 01_prepare_data.py's filter so appended passages are drawn from the same
# population as the originals. A pool whose second half is systematically longer or
# shorter than its first half would confound pool size with passage length.
MIN_PASSAGE_LENGTH = 100
MAX_PASSAGE_LENGTH = 1500


def load_pool(path):
    with open(path, "r", encoding="utf-8") as f:
        pool = json.load(f)
    if not isinstance(pool, list):
        raise SystemExit(f"{path}: expected a JSON list of passage strings, "
                         f"got {type(pool).__name__}")
    return pool


def stream_new_passages(existing, wanted, seed=20260805):
    """Yield passages from wiki_snippets that are not already in the pool.

    Sampled from a shuffled index range rather than taken in file order: consecutive
    wiki40b snippets come from the same article, so appending them in order would add
    20,000 passages covering far fewer topics than 20,000 random ones.
    """
    from datasets import load_dataset

    seen = set(existing)
    wiki = load_dataset("community-datasets/wiki_snippets", "wiki40b_en_100_0", split="train",
                        streaming=False, trust_remote_code=True)

    order = list(range(len(wiki)))
    random.Random(seed).shuffle(order)

    produced = 0
    for idx in order:
        if produced >= wanted:
            return
        text = (wiki[idx].get("passage_text") or "").strip()
        if not (MIN_PASSAGE_LENGTH <= len(text) <= MAX_PASSAGE_LENGTH):
            continue
        if text in seen:
            continue
        seen.add(text)
        produced += 1
        yield text


def append_pool(pool, new_passages):
    """Return pool + new passages. The prefix is returned unchanged, by construction."""
    return list(pool) + list(new_passages)


def verify(old, new, target):
    """The invariant the whole design rests on. Loud, and checked before anything is written."""
    problems = []
    if len(new) < len(old):
        problems.append(f"pool shrank: {len(old)} -> {len(new)}")
    if new[:len(old)] != old:
        first = next((i for i in range(len(old)) if new[i] != old[i]), None)
        problems.append(f"prefix changed at index {first} -- gold assignment is no longer "
                        f"monotone and the two runs are NOT comparable")
    if len(set(new)) != len(new):
        problems.append(f"{len(new) - len(set(new))} duplicate passages")
    if len(new) != target:
        problems.append(f"size {len(new)} != requested {target} (source exhausted?)")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=100000, help="final pool size")
    ap.add_argument("--cache", default=CACHE_PATH)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and exit without writing")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        raise SystemExit(f"{args.cache} not found. Nothing to append to -- if you want a "
                         f"fresh pool, run 01_prepare_data.py, but note that a fresh pool "
                         f"is NOT comparable to the existing run.")

    old = load_pool(args.cache)
    need = args.target - len(old)
    print(f"Existing pool: {len(old):,} passages")
    print(f"Target:        {args.target:,}  (need {need:,} more)")

    if need <= 0:
        print("Pool already at or above target; nothing to do.")
        return
    if args.dry_run:
        print("\n--dry-run: no download, no write. Re-run without it to apply.")
        return

    backup = args.cache.replace(".json", f"_{len(old)}.backup.json")
    if not args.no_backup and not os.path.exists(backup):
        shutil.copy2(args.cache, backup)
        print(f"Backed up original -> {backup}")

    print(f"\nStreaming wiki_snippets for {need:,} unseen passages "
          f"({MIN_PASSAGE_LENGTH}-{MAX_PASSAGE_LENGTH} chars)...")
    added = list(stream_new_passages(old, need, seed=args.seed))
    print(f"  collected {len(added):,}")

    new = append_pool(old, added)

    problems = verify(old, new, args.target)
    if problems:
        print("\nVERIFICATION FAILED -- nothing written:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(1)

    with open(args.cache, "w", encoding="utf-8") as f:
        json.dump(new, f)

    print(f"\nVerified: prefix [0:{len(old):,}] byte-identical, no duplicates, "
          f"size {len(new):,}")
    print(f"Wrote {args.cache}")
    print("\nNext: re-run the pipeline. Every query that had real Wikipedia gold at "
          f"{len(old):,} still has it; some 'constructed' queries will now find a real one.")


if __name__ == "__main__":
    main()
