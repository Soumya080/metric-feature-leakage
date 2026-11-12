"""01 — Build the single shared corpus that every query retrieves from.

Reads the cached Wikipedia passage pool, deduplicates it, and writes it once as
`artifacts/corpus.jsonl` in a fixed order. Every later stage indexes a PREFIX of a
seeded permutation of that order, which is what makes 5K a subset of 20K a subset
of 50K a subset of 100K.

Two differences from `experiments/baseline_v2/01_prepare_data.py` matter:

  * There are no per-query corpora here. One corpus, shared by all ~6,000 queries.
  * `construct_gold()` is not used and has no counterpart. In the 500-passage
    setting, 41.8% of NQ gold passages are template-generated filler containing the
    answer string (see results/corpus_census.json); those passages were in the pool
    during retrieval and perturbed real queries' rankings too. corpus_v3 contains
    only genuine Wikipedia text, which makes it the cleaner of the two settings —
    worth stating in the paper rather than burying.

A query whose answer appears in no corpus passage is NOT dropped. It stays and
scores zero for every retriever, which is the honest open-domain condition; the
share of such queries is reported by 04_run_retrieval.py.

Usage:
    python experiments/corpus_v3/01_build_shared_corpus.py
    python experiments/corpus_v3/01_build_shared_corpus.py --smoke
    python experiments/corpus_v3/01_build_shared_corpus.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402

# Guards against pool entries that are boilerplate or truncated fragments. These
# bounds match MIN/MAX_PASSAGE_LENGTH in baseline_v2/01_prepare_data.py so the two
# settings draw from text of the same shape.
MIN_CHARS = 80
MAX_CHARS = 1500


def normalise(text: str) -> str:
    """Whitespace-collapsed, case-folded key used only for duplicate detection.

    The stored text keeps its original casing and spacing — this is a comparison
    key, not a transformation of the corpus.
    """
    return " ".join(text.lower().split())


def build(smoke: bool = False, force: bool = False) -> dict:
    common.ensure_dirs()

    if os.path.exists(common.CORPUS_PATH) and not force:
        manifest = common.read_manifest(common.CORPUS_MANIFEST_PATH)
        if manifest:
            print(f"Corpus already built: {common.fmt_int(manifest['n_documents'])} documents")
            print(f"  {common.CORPUS_PATH}")
            print("  (pass --force to rebuild)")
            return manifest

    if not os.path.exists(common.POOL_PATH):
        raise FileNotFoundError(
            f"passage pool not found at {common.POOL_PATH}\n"
            "Run experiments/baseline_v2/01_prepare_data.py to populate the cache first."
        )

    print(f"Reading pool: {common.POOL_PATH}")
    with open(common.POOL_PATH, "r", encoding="utf-8") as f:
        pool = json.load(f)
    print(f"  {common.fmt_int(len(pool))} passages in pool")

    target = max(common.SMOKE_SCALES) if smoke else max(common.SCALES)

    seen: set[str] = set()
    kept: list[str] = []
    n_dupe = n_short = n_long = 0

    for text in pool:
        if not isinstance(text, str):
            continue
        text = text.strip()
        if len(text) < MIN_CHARS:
            n_short += 1
            continue
        if len(text) > MAX_CHARS:
            n_long += 1
            continue
        key = normalise(text)
        if key in seen:
            n_dupe += 1
            continue
        seen.add(key)
        kept.append(text)
        if len(kept) >= target:
            break

    if len(kept) < target:
        print(
            f"  WARNING: pool yielded {common.fmt_int(len(kept))} usable passages, "
            f"short of the {common.fmt_int(target)} requested. Scales above "
            f"{common.fmt_int(len(kept))} will be skipped downstream."
        )

    print(f"  kept {common.fmt_int(len(kept))} | dropped: "
          f"{common.fmt_int(n_dupe)} duplicate, {common.fmt_int(n_short)} too short, "
          f"{common.fmt_int(n_long)} too long")

    tmp_path = common.CORPUS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(kept):
            f.write(json.dumps({"doc_id": i, "text": text}, ensure_ascii=False) + "\n")
    os.replace(tmp_path, common.CORPUS_PATH)

    manifest = {
        "n_documents": len(kept),
        "source": os.path.relpath(common.POOL_PATH, common.PROJECT_ROOT),
        "source_pool_size": len(pool),
        "corpus_seed": common.CORPUS_SEED,
        "min_chars": MIN_CHARS,
        "max_chars": MAX_CHARS,
        "dropped_duplicate": n_dupe,
        "dropped_short": n_short,
        "dropped_long": n_long,
        "smoke": smoke,
        "synthetic_passages": 0,  # corpus_v3 never synthesises gold — see module docstring
        "sha256": common.sha256_of_file(common.CORPUS_PATH),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    common.write_manifest(common.CORPUS_MANIFEST_PATH, manifest)

    print(f"\nWrote {common.CORPUS_PATH}")
    print(f"  sha256 {manifest['sha256'][:16]}...")
    print(f"  manifest {common.CORPUS_MANIFEST_PATH}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--smoke", action="store_true",
                        help=f"build only {max(common.SMOKE_SCALES)} documents for a fast end-to-end check")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the corpus already exists")
    args = parser.parse_args()

    build(smoke=args.smoke, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
