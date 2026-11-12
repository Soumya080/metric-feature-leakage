"""02 — Embed the shared corpus ONCE and cache the vectors to disk.

This is the only stage that wants a GPU, and it is the reason the whole sweep is
cheap. The 500-passage pipeline re-embeds a private corpus per query — roughly
6,000 queries x 500 passages = 3.0M passage embeddings per full run, of passages
drawn from a pool of only 100K, so ~97% of that work is re-embedding duplicates.

Here the corpus is embedded once. Every scale (5K/20K/50K/100K) is then a slice of
this one cached array, so growing the corpus costs nothing extra in embedding time.

Query embeddings are computed here too: 04_run_retrieval needs them for dense
retrieval, and the same vectors serve the embedding-state control (GBT on frozen
MiniLM query embeddings vs. the hand-crafted features).

Storage: float32 on disk, L2-normalised at index-build time rather than here, so
the cache stays usable if a later stage wants un-normalised vectors.

Usage:
    python experiments/corpus_v3/02_embed_corpus.py
    python experiments/corpus_v3/02_embed_corpus.py --smoke
    python experiments/corpus_v3/02_embed_corpus.py --batch-size 512 --force
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402


def _load_encoder(batch_size: int):
    """SentenceTransformer plus a note about where it will actually run.

    torch in this venv may be a CPU-only build, in which case embedding 100K
    passages takes roughly 20-40 minutes instead of 3-5. That is still perfectly
    acceptable for a one-off, but it should be visible in the log rather than
    discovered by watching a progress bar crawl.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {common.EMBEDDING_MODEL} on {device}")
    if device == "cpu":
        print("  NOTE: CUDA is not available to this interpreter (torch is a CPU build).")
        print("        Embedding will still complete, just slower. To use the GPU:")
        print("        pip install --force-reinstall torch --index-url "
              "https://download.pytorch.org/whl/cu121")

    model = SentenceTransformer(common.EMBEDDING_MODEL, device=device)
    return model, device


def _encode(model, texts, batch_size: int, label: str) -> np.ndarray:
    t0 = time.time()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    elapsed = time.time() - t0
    rate = len(texts) / elapsed if elapsed > 0 else 0.0
    print(f"  {label}: {common.fmt_int(len(texts))} in {elapsed:.1f}s ({rate:.0f}/s)")
    return vectors


def embed(batch_size: int = 256, force: bool = False, smoke: bool = False) -> dict:
    common.ensure_dirs()

    corpus_manifest = common.read_manifest(common.CORPUS_MANIFEST_PATH)
    if corpus_manifest is None:
        raise FileNotFoundError(
            "corpus manifest missing. Run 01_build_shared_corpus.py first."
        )

    existing = common.read_manifest(common.EMBEDDINGS_MANIFEST_PATH)
    corpus_done = (
        existing is not None
        and not force
        and os.path.exists(common.EMBEDDINGS_PATH)
        # Rebuild if the corpus changed underneath us — a stale embedding cache
        # silently paired with a new corpus would corrupt every downstream number.
        and existing.get("corpus_sha256") == corpus_manifest["sha256"]
    )

    model = device = None

    if corpus_done:
        print(f"Corpus embeddings already cached: {common.EMBEDDINGS_PATH}")
        print("  (pass --force to recompute)")
        manifest = existing
    else:
        texts = common.load_corpus()
        print(f"Embedding {common.fmt_int(len(texts))} corpus passages")
        model, device = _load_encoder(batch_size)
        vectors = _encode(model, texts, batch_size, "corpus")

        tmp = common.EMBEDDINGS_PATH + ".tmp.npy"
        np.save(tmp, vectors)
        os.replace(tmp, common.EMBEDDINGS_PATH)

        manifest = {
            "model": common.EMBEDDING_MODEL,
            "device": device,
            "n_documents": int(vectors.shape[0]),
            "dimension": int(vectors.shape[1]),
            "dtype": str(vectors.dtype),
            "batch_size": batch_size,
            "normalized_on_disk": False,
            "corpus_sha256": corpus_manifest["sha256"],
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        common.write_manifest(common.EMBEDDINGS_MANIFEST_PATH, manifest)
        size_mb = os.path.getsize(common.EMBEDDINGS_PATH) / 1024 / 1024
        print(f"  wrote {common.EMBEDDINGS_PATH} ({size_mb:.0f} MB)")

    # --- Query embeddings, one file per dataset --------------------------------
    limit = 200 if smoke else None
    for dataset in common.DATASETS:
        out_path = common.QUERY_EMBEDDINGS_PATH.format(dataset=dataset)
        try:
            queries = common.load_queries(dataset, limit=limit)
        except FileNotFoundError as exc:
            print(f"  skipping {dataset}: {exc}")
            continue

        if os.path.exists(out_path) and not force:
            cached = np.load(out_path, mmap_mode="r")
            cached_shape = cached.shape[0]
            del cached  # Release file handle immediately so Windows doesn't lock it
            if cached_shape == len(queries):
                print(f"  {dataset}: query embeddings cached ({common.fmt_int(len(queries))})")
                continue
            print(f"  {dataset}: cached query embeddings are stale "
                  f"({cached_shape} vs {len(queries)}), recomputing")

        if model is None:
            model, device = _load_encoder(batch_size)

        vectors = _encode(model, [q["question"] for q in queries], batch_size, dataset)
        tmp = out_path + ".tmp.npy"
        np.save(tmp, vectors)
        os.replace(tmp, out_path)

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--smoke", action="store_true",
                        help="embed only the first 200 queries per dataset")
    parser.add_argument("--force", action="store_true",
                        help="recompute even if a valid cache exists")
    args = parser.parse_args()

    embed(batch_size=args.batch_size, force=args.force, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
