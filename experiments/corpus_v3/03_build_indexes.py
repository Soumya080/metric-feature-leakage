"""03 — Build one BM25 + one FAISS index per corpus scale.

The scales are NESTED. A single seeded permutation of document ids is computed once
(common.build_scale_order); scale N indexes the first N entries of it. So

    docs(5K) subset of docs(20K) subset of docs(50K) subset of docs(100K)

and the leak-vs-scale curve varies corpus size with the smaller corpora being
literal subsets of the larger. Drawing an independent sample per scale would
confound "bigger corpus" with "different documents" and the curve would mean
nothing.

Dense index is `IndexFlatIP` over L2-normalised vectors — exact inner product, i.e.
exact cosine. Deliberately not HNSW/IVF: at 100K x 384 the flat index is 154 MB and
searches in milliseconds, so there is no reason to introduce approximate-recall as
a second thing varying with corpus size.

BM25 is the vectorised Okapi implementation in bm25_sparse.py, which is asserted
equal to `src/retrieval/sparse.BM25Retriever` by tests/test_bm25_parity.py. Same
ranking function as the 500-passage setting; only the corpus differs.

Usage:
    python experiments/corpus_v3/03_build_indexes.py
    python experiments/corpus_v3/03_build_indexes.py --smoke
    python experiments/corpus_v3/03_build_indexes.py --scale 5000 --force
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from bm25_sparse import SparseBM25  # noqa: E402


def _paths(scale: int):
    d = common.index_dir(scale)
    return {
        "dir": d,
        "faiss": os.path.join(d, "dense.faiss"),
        "bm25": os.path.join(d, "bm25.pkl"),
        "doc_ids": os.path.join(d, "doc_ids.npy"),
        "manifest": os.path.join(d, "index_manifest.json"),
    }


def build_scale(scale: int, corpus: list[str], embeddings: np.ndarray,
                order: np.ndarray, force: bool = False) -> dict | None:
    import faiss

    paths = _paths(scale)
    manifest = common.read_manifest(paths["manifest"])
    if manifest and not force and all(
        os.path.exists(paths[k]) for k in ("faiss", "bm25", "doc_ids")
    ):
        print(f"  scale {common.fmt_int(scale)}: already built (--force to rebuild)")
        return manifest

    if scale > len(corpus):
        print(f"  scale {common.fmt_int(scale)}: SKIPPED — corpus has only "
              f"{common.fmt_int(len(corpus))} documents")
        return None

    os.makedirs(paths["dir"], exist_ok=True)
    doc_ids = order[:scale].astype(np.int64)
    texts = [corpus[i] for i in doc_ids]

    print(f"  scale {common.fmt_int(scale)}: indexing {common.fmt_int(len(texts))} documents")

    # --- Dense -----------------------------------------------------------------
    t0 = time.time()
    vectors = np.ascontiguousarray(embeddings[doc_ids].astype(np.float32))
    # Normalise here rather than in the cache, so the cached array stays raw.
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, paths["faiss"])
    dense_secs = time.time() - t0

    # --- Sparse ----------------------------------------------------------------
    t0 = time.time()
    bm25 = SparseBM25().fit(texts)
    with open(paths["bm25"] + ".tmp", "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(paths["bm25"] + ".tmp", paths["bm25"])
    bm25_secs = time.time() - t0

    np.save(paths["doc_ids"], doc_ids)

    manifest = {
        "scale": scale,
        "n_documents": len(texts),
        "corpus_seed": common.CORPUS_SEED,
        "dimension": int(vectors.shape[1]),
        "faiss_index": "IndexFlatIP",
        "bm25_k1": bm25.k1,
        "bm25_b": bm25.b,
        "bm25_vocab_size": len(bm25.vocab),
        "bm25_nnz": int(bm25.matrix.nnz),
        "dense_build_seconds": round(dense_secs, 2),
        "bm25_build_seconds": round(bm25_secs, 2),
        "first_doc_id": int(doc_ids[0]),
        "last_doc_id": int(doc_ids[-1]),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    common.write_manifest(paths["manifest"], manifest)

    mb = os.path.getsize(paths["faiss"]) / 1024 / 1024
    print(f"    dense {dense_secs:.1f}s ({mb:.0f} MB) | "
          f"bm25 {bm25_secs:.1f}s (vocab {common.fmt_int(len(bm25.vocab))}, "
          f"nnz {common.fmt_int(bm25.matrix.nnz)})")
    return manifest


def verify_nesting(scales) -> bool:
    """Assert docs(smaller) is a strict subset of docs(larger), for every pair.

    This is the property the whole leak-vs-scale reading rests on, so it is checked
    against what is actually on disk rather than trusted from the construction.
    """
    available = [s for s in scales if os.path.exists(_paths(s)["doc_ids"])]
    if len(available) < 2:
        print("  nesting check: fewer than two scales present, nothing to compare")
        return True

    ok = True
    for smaller, larger in zip(available, available[1:]):
        a = set(np.load(_paths(smaller)["doc_ids"]).tolist())
        b = set(np.load(_paths(larger)["doc_ids"]).tolist())
        if a.issubset(b):
            print(f"  nesting OK: {common.fmt_int(smaller)} subset of {common.fmt_int(larger)}")
        else:
            print(f"  NESTING VIOLATED: {common.fmt_int(smaller)} is not a subset of "
                  f"{common.fmt_int(larger)} ({len(a - b)} documents missing)")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scale", type=int, nargs="+", default=None,
                        help="specific scale(s) to build (default: all)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    common.ensure_dirs()
    scales = common.resolve_scales(args.smoke, tuple(args.scale) if args.scale else None)

    corpus = common.load_corpus()
    if not os.path.exists(common.EMBEDDINGS_PATH):
        raise FileNotFoundError(
            "corpus embeddings missing. Run 02_embed_corpus.py first."
        )
    embeddings = np.load(common.EMBEDDINGS_PATH, mmap_mode="r")
    if embeddings.shape[0] != len(corpus):
        raise ValueError(
            f"embedding cache has {embeddings.shape[0]} rows but the corpus has "
            f"{len(corpus)} documents — rerun 02_embed_corpus.py --force"
        )

    order = common.build_scale_order(len(corpus))
    print(f"Corpus: {common.fmt_int(len(corpus))} documents | "
          f"scales: {', '.join(common.fmt_int(s) for s in scales)}")

    for scale in scales:
        build_scale(scale, corpus, embeddings, order, force=args.force)

    print("\nVerifying nested subsets")
    if not verify_nesting(scales):
        print("\nFAILED: subsets are not nested. The leak-vs-scale curve would be "
              "uninterpretable; rebuild with --force.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
