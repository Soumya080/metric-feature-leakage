"""04 — Run BM25 / Dense / Hybrid over the shared index, one shard per (scale, dataset).

Output is byte-compatible with `experiments/baseline_v2/results/{ds}_per_query_results.json`,
so `src/router/prepare_env_data.py --feature-mode ...` consumes it unchanged and every
leak arm at every scale is built by exactly the same code that built the 500-passage arms.

Three things are done differently from the 500-passage runner, all for scale:

  * Scoring is batched. Dense is one BLAS matmul per query batch against the cached
    embedding matrix; BM25 is one sparse matmul against the precomputed CSR weights.
    Ranking function and constants are unchanged — see bm25_sparse.py and its parity test.

  * RRF is vectorised. `reciprocal_rank_fusion` builds a Python dict over the full
    ranked list, which at 100K documents x 6,000 queries is ~600M dict operations.
    The arithmetic here is identical (1/(60+rank), summed over the lists a document
    appears in, BM25 contributing only where it scored > 0) and is checked against the
    reference implementation by --verify.

  * `gold_rank` is resolved through an inverted index instead of walking the ranked
    list. `first_answer_bearing_rank` normalises and scans every retrieved passage,
    which is fine over 500 but is 100K string comparisons per query per retriever here.
    Answer-bearing documents are instead identified once per query, then the rank is a
    lookup. Same word-boundary matching, same answer, same result — also checked by
    --verify.

Latency is measured against the PREBUILT index. The 500-passage runner times
`bm25.fit()` and `faiss.add_chunks()` inside the measured region, which is why it
reports Dense as faster than BM25; those numbers describe index construction, not
query serving, and are not comparable to these.

Usage:
    python experiments/corpus_v3/04_run_retrieval.py
    python experiments/corpus_v3/04_run_retrieval.py --smoke --verify
    python experiments/corpus_v3/04_run_retrieval.py --scale 100000 --dataset nq
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402

RRF_K = 60  # matches src/retrieval/hybrid.reciprocal_rank_fusion's default
QUERY_BATCH = 128


# =====================================================================
# Answer-bearing document lookup
# =====================================================================

class AnswerIndex:
    """Inverted index over normalised corpus text, for locating answer-bearing docs.

    Used only to answer "which documents contain this answer as a whole word/phrase",
    which is what `first_answer_bearing_rank` determines by linear scan. Matching is
    delegated to the reference `_word_boundary_contains`, so a document is accepted
    here if and only if the reference would have accepted it.
    """

    def __init__(self, normalised_docs: list[str], metrics):
        self._docs = normalised_docs
        self._contains = metrics._word_boundary_contains
        self._normalize = metrics.normalize_answer

        postings: dict[str, list[int]] = defaultdict(list)
        for doc_id, text in enumerate(normalised_docs):
            for word in set(text.split()):
                postings[word].append(doc_id)
        self._postings = {w: np.asarray(ids, dtype=np.int64) for w, ids in postings.items()}

    def answer_docs(self, answers: list[str]) -> np.ndarray:
        """Sorted doc ids containing any of `answers`. Empty if none do."""
        found: set[int] = set()
        for answer in answers:
            needle = self._normalize(answer)
            if not needle:
                continue
            words = needle.split()
            if not words:
                continue
            # Start from the rarest word's posting list: the phrase can only occur
            # where every one of its words occurs, so this is the smallest candidate
            # set to verify.
            postings = [self._postings.get(w) for w in words]
            if any(p is None for p in postings):
                continue
            candidates = min(postings, key=len)
            for doc_id in candidates:
                if doc_id in found:
                    continue
                if self._contains(needle, self._docs[doc_id]):
                    found.add(int(doc_id))
        return np.asarray(sorted(found), dtype=np.int64)


def _normalised_corpus(texts: list[str], metrics, cache_path: str) -> list[str]:
    """normalize_answer() over the whole corpus, cached — it is pure and reused per scale."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if len(cached) == len(texts):
            return cached
    normalised = [metrics.normalize_answer(t) for t in texts]
    with open(cache_path + ".tmp", "wb") as f:
        pickle.dump(normalised, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(cache_path + ".tmp", cache_path)
    return normalised


# =====================================================================
# Ranking
# =====================================================================

def _rank_positions(order: np.ndarray, n_docs: int) -> np.ndarray:
    """position[doc_id] = 0-based rank of that doc in `order`; n_docs if absent."""
    pos = np.full(n_docs, n_docs, dtype=np.int64)
    pos[order] = np.arange(len(order), dtype=np.int64)
    return pos


def _rrf_full(dense_order: np.ndarray, sparse_order: np.ndarray, n_docs: int):
    """Vectorised reciprocal rank fusion over the two full ranked lists.

    Equivalent to reciprocal_rank_fusion(dense, sparse, k=60, top_n=n): a document's
    score is the sum of 1/(60 + rank) over the lists it appears in, with rank 1-based.
    Documents BM25 scored 0 are absent from `sparse_order` and contribute nothing,
    matching sparse.py dropping them.

    Ties are broken by dense rank then sparse rank, which reproduces the reference's
    behaviour: it inserts dense hits into the score dict first, sparse-only hits after,
    and Python's sort is stable.
    """
    scores = np.zeros(n_docs, dtype=np.float64)
    scores[dense_order] += 1.0 / (RRF_K + 1.0 + np.arange(len(dense_order)))
    scores[sparse_order] += 1.0 / (RRF_K + 1.0 + np.arange(len(sparse_order)))

    insertion = np.full(n_docs, 2 * n_docs, dtype=np.int64)
    insertion[sparse_order] = n_docs + np.arange(len(sparse_order))
    insertion[dense_order] = np.arange(len(dense_order))

    seen = scores > 0
    candidates = np.flatnonzero(seen)
    order = candidates[np.lexsort((insertion[candidates], -scores[candidates]))]
    return order, scores


def _summary_scores(scores: np.ndarray) -> dict:
    top5 = scores[:5]
    return {
        "top1_score": float(scores[0]) if scores.size else 0.0,
        "mean_score_top5": float(np.mean(top5)) if top5.size else 0.0,
        "std_score_top5": float(np.std(top5)) if top5.size else 0.0,
        "scores_top10": [float(s) for s in scores[:10]],
    }


def _cross_scores(indices: np.ndarray, bm25_scores: np.ndarray,
                  dense_scores: np.ndarray, top_n: int = 5) -> dict:
    """This retriever's own top-n, scored under both scoring functions.

    Same quantity as cross_scoring_features(); computed by indexing the full score
    vectors rather than by dict lookup over candidate lists. BM25 scores of exactly 0
    are recorded as 0, as in the reference.
    """
    idx = indices[:top_n]
    if idx.size == 0:
        empty = {"mean": 0.0, "max": 0.0}
        return {"bm25": empty, "dense": dict(empty), "n_scored": 0}
    b = bm25_scores[idx]
    d = dense_scores[idx]
    return {
        "bm25": {"mean": float(b.mean()), "max": float(b.max())},
        "dense": {"mean": float(d.mean()), "max": float(d.max())},
        "n_scored": int(idx.size),
    }


# =====================================================================
# Per-scale, per-dataset run
# =====================================================================

def run_shard(scale: int, dataset: str, *, corpus_texts, answer_index,
              embeddings, bm25, doc_ids, metrics, limit=None, force=False,
              verify=False) -> dict | None:
    out_path = common.shard_path(scale, dataset)
    if os.path.exists(out_path) and not force:
        print(f"    {dataset}: shard exists, skipping (--force to rerun)")
        with open(out_path, "r", encoding="utf-8") as f:
            return {"n_queries": len(json.load(f)), "skipped": True}

    queries = common.load_queries(dataset, limit=limit)
    q_emb_path = common.QUERY_EMBEDDINGS_PATH.format(dataset=dataset)
    if not os.path.exists(q_emb_path):
        raise FileNotFoundError(
            f"query embeddings missing for {dataset}. Run 02_embed_corpus.py first."
        )
    query_vectors = np.load(q_emb_path).astype(np.float32)
    if query_vectors.shape[0] < len(queries):
        raise ValueError(
            f"{dataset}: {query_vectors.shape[0]} query embeddings for "
            f"{len(queries)} queries — rerun 02_embed_corpus.py --force"
        )
    query_vectors = query_vectors[: len(queries)]
    # Cosine == inner product on normalised vectors, matching FAISS IndexFlatIP.
    norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_vectors = query_vectors / np.maximum(norms, 1e-12)

    n_docs = len(doc_ids)
    max_k = max(common.TOP_K_VALUES)
    results = []
    n_unanswerable = 0

    t_start = time.time()
    for start in range(0, len(queries), QUERY_BATCH):
        batch = queries[start:start + QUERY_BATCH]
        qv = query_vectors[start:start + QUERY_BATCH]

        t0 = time.time()
        dense_batch = np.asarray(embeddings @ qv.T, dtype=np.float64)  # (n_docs, batch)
        dense_batch_ms = (time.time() - t0) * 1000 / max(len(batch), 1)

        for j, query in enumerate(batch):
            dense_scores = dense_batch[:, j]

            t0 = time.time()
            dense_order = np.argsort(-dense_scores, kind="stable")
            dense_ms = dense_batch_ms + (time.time() - t0) * 1000

            t0 = time.time()
            sparse_scores = bm25.score(query["question"])
            positive = np.flatnonzero(sparse_scores > 0)
            sparse_order = positive[np.argsort(-sparse_scores[positive], kind="stable")]
            bm25_ms = (time.time() - t0) * 1000

            t0 = time.time()
            hybrid_order, hybrid_scores = _rrf_full(dense_order, sparse_order, n_docs)
            hybrid_ms = (time.time() - t0) * 1000

            answers = query["answers"]
            answer_docs = answer_index.answer_docs(answers)
            # Corpus-relative ids: the index holds a subset of the corpus, so map the
            # global answer-bearing ids onto local index positions.
            local_answer = np.flatnonzero(np.isin(doc_ids, answer_docs))
            if local_answer.size == 0:
                n_unanswerable += 1

            record = {
                "query_id": query["query_id"],
                "question": query["question"],
                "answers": answers,
                "corpus_size": n_docs,
                # corpus_v3 contains no synthetic passages; gold_type is carried from
                # the 500-passage setting purely so results can be stratified the same
                # way. See 01_build_shared_corpus.py.
                "gold_type": "real_wikipedia",
                "gold_type_v2": query["gold_type_v2"],
                "n_corpus": n_docs,
                "scale": scale,
                "cross_score_norm": {
                    "bm25_max": float(sparse_scores[sparse_order[0]]) if sparse_order.size else 0.0,
                    "dense_max": float(dense_scores[dense_order[0]]) if dense_order.size else 0.0,
                },
                # Only needed by the fusion baseline, which stays at the 500-passage
                # setting; computing 11 full-corpus re-rankings per query here would
                # dominate the run for a result §3b does not use.
                "convex_fusion_gold_rank": None,
                "retrievers": {},
            }

            ranked = {
                "bm25": (sparse_order, sparse_scores[sparse_order], bm25_ms),
                "dense": (dense_order, dense_scores[dense_order], dense_ms),
                "hybrid": (hybrid_order, hybrid_scores[hybrid_order], hybrid_ms),
            }

            for name, (order, ordered_scores, latency_ms) in ranked.items():
                top = order[:max_k]
                contents = [corpus_texts[doc_ids[i]] for i in top]

                if local_answer.size:
                    positions = _rank_positions(order, n_docs)[local_answer]
                    best = int(positions.min())
                    gold_rank = best + 1 if best < n_docs else None
                else:
                    gold_rank = None

                block = {
                    "latency_ms": latency_ms,
                    **_summary_scores(ordered_scores),
                    "retrieved_ids_top10": [int(doc_ids[i]) for i in top[:10]],
                    "gold_rank": gold_rank,
                    "cross_scores": _cross_scores(top, sparse_scores, dense_scores),
                    "metrics_at_k": {},
                }

                for k in common.TOP_K_VALUES:
                    top_k_contents = contents[:k]
                    concat = " ".join(top_k_contents)
                    block["metrics_at_k"][str(k)] = {
                        "hit": metrics.check_hit(top_k_contents, answers),
                        "mrr": metrics.compute_mrr(top_k_contents, answers),
                        "recall": metrics.compute_recall_at_k(contents, answers, k),
                        "f1": metrics.f1_score(concat, answers),
                        "em": metrics.substring_match_score(concat, answers),
                    }

                record["retrievers"][name] = block

            if verify and len(results) < 5:
                _verify_record(record, ranked, corpus_texts, doc_ids, metrics)

            results.append(record)

        done = min(start + QUERY_BATCH, len(queries))
        if done % (QUERY_BATCH * 8) == 0 or done == len(queries):
            rate = done / max(time.time() - t_start, 1e-9)
            print(f"    {dataset}: {done}/{len(queries)} ({rate:.1f} q/s)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(results, f)
    os.replace(out_path + ".tmp", out_path)

    # Two Hit@5 numbers, and the distinction matters.
    #
    # In the 500-passage setting a gold passage was INSERTED into every query's corpus,
    # so every query was answerable by construction. A shared corpus makes no such
    # promise: if no passage happens to contain the answer, every retriever scores 0 and
    # the query says nothing about retrieval quality. Overall Hit@5 therefore mixes "the
    # retriever failed" with "there was nothing to find", and the mixture changes with
    # corpus size — which is precisely the axis being studied.
    #
    # `answerable` conditions on at least one answer-bearing passage existing, and is the
    # number to compare across scales. The unanswerable share is reported alongside it
    # because it is itself a finding, not a nuisance.
    def _hit5(rows):
        return float(np.mean([r["retrievers"]["dense"]["metrics_at_k"]["5"]["hit"]
                              for r in rows])) if rows else 0.0

    answerable = [r for r in results
                  if any(r["retrievers"][x]["gold_rank"] is not None
                         for x in common.RETRIEVERS)]
    hit5_all = _hit5(results)
    hit5_answerable = _hit5(answerable)

    summary = {
        "scale": scale,
        "dataset": dataset,
        "n_queries": len(results),
        "n_answerable": len(answerable),
        "unanswerable_in_corpus": n_unanswerable,
        "unanswerable_share": round(n_unanswerable / max(len(results), 1), 4),
        "dense_hit_at_5": round(hit5_all, 4),
        "dense_hit_at_5_answerable": round(hit5_answerable, 4),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"    {dataset}: dense Hit@5 = {hit5_all:.3f} overall, "
          f"{hit5_answerable:.3f} among answerable "
          f"({len(answerable)}/{len(results)}) | "
          f"{summary['elapsed_seconds']:.0f}s")
    return summary


def _verify_record(record, ranked, corpus_texts, doc_ids, metrics) -> None:
    """Check the fast paths against the reference implementations on one query.

    Only the two places where this file does not simply call the reference are
    checked: gold_rank (inverted index vs. linear scan) and the RRF ordering
    (vectorised vs. dict-based). Everything else already calls 02_run_baselines.
    """
    from src.retrieval.hybrid import reciprocal_rank_fusion

    answers = record["answers"]
    for name in ("bm25", "dense"):
        order, scores, _ = ranked[name]
        depth = min(len(order), 2000)
        contents = [corpus_texts[doc_ids[i]] for i in order[:depth]]
        expected = metrics.first_answer_bearing_rank(contents, answers)
        actual = record["retrievers"][name]["gold_rank"]
        if expected is not None:
            assert actual == expected, (
                f"gold_rank mismatch on {name} for query {record['query_id']}: "
                f"fast={actual} reference={expected}"
            )
        elif actual is not None:
            assert actual > depth, (
                f"gold_rank {actual} for {name} not found by reference within {depth}"
            )

    d_order, d_scores, _ = ranked["dense"]
    s_order, s_scores, _ = ranked["bm25"]

    # The reference must be given the FULL ranked lists, because that is what
    # 02_run_baselines fuses (top_n=n_corpus) and what _rrf_full reproduces. Feeding it
    # a truncated candidate pool changes every rank beyond the cut and makes the
    # comparison ill-posed rather than failing. Only run it where the O(n) dict merge
    # is affordable.
    if len(d_order) > 20_000:
        return

    dense_hits = [{"chunk_id": f"c{int(i)}", "chunk_index": int(i), "score": float(sc)}
                  for i, sc in zip(d_order, d_scores)]
    sparse_hits = [{"chunk_id": f"c{int(i)}", "chunk_index": int(i), "score": float(sc)}
                   for i, sc in zip(s_order, s_scores)]
    reference = reciprocal_rank_fusion(dense_hits, sparse_hits, k=RRF_K,
                                       top_n=len(dense_hits))
    ref_ids = [h["chunk_index"] for h in reference[:10]]

    fast_ids = [int(i) for i in ranked["hybrid"][0][:10]]
    assert fast_ids == ref_ids, (
        f"RRF ordering mismatch for query {record['query_id']}:\n"
        f"  fast      {fast_ids}\n  reference {ref_ids}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scale", type=int, nargs="+", default=None)
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        choices=list(common.DATASETS))
    parser.add_argument("--limit", type=int, default=None,
                        help="cap queries per dataset (debugging)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="assert the fast gold_rank/RRF paths match the reference")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    common.ensure_dirs()
    scales = common.resolve_scales(args.smoke, tuple(args.scale) if args.scale else None)
    datasets = tuple(args.dataset) if args.dataset else common.DATASETS
    limit = args.limit if args.limit is not None else (200 if args.smoke else None)

    metrics = common.load_metric_module()
    corpus_texts = common.load_corpus()
    embeddings_all = np.load(common.EMBEDDINGS_PATH, mmap_mode="r")

    print("Preparing answer-bearing index (normalising corpus once)")
    t0 = time.time()
    normalised = _normalised_corpus(
        corpus_texts, metrics, os.path.join(common.ARTIFACT_DIR, "corpus_normalised.pkl")
    )
    answer_index = AnswerIndex(normalised, metrics)
    print(f"  ready in {time.time() - t0:.1f}s "
          f"({common.fmt_int(len(answer_index._postings))} distinct terms)")

    all_summaries = []
    failures = []

    for scale in scales:
        idx_dir = common.index_dir(scale)
        bm25_path = os.path.join(idx_dir, "bm25.pkl")
        ids_path = os.path.join(idx_dir, "doc_ids.npy")
        if not (os.path.exists(bm25_path) and os.path.exists(ids_path)):
            print(f"\nscale {common.fmt_int(scale)}: index missing, skipping")
            continue

        print(f"\nscale {common.fmt_int(scale)}")
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        doc_ids = np.load(ids_path)
        # Normalised subset matrix, in index order — the same vectors FAISS holds.
        vectors = np.ascontiguousarray(embeddings_all[doc_ids].astype(np.float32))
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

        for dataset in datasets:
            try:
                summary = run_shard(
                    scale, dataset,
                    corpus_texts=corpus_texts,
                    answer_index=answer_index, embeddings=vectors, bm25=bm25,
                    doc_ids=doc_ids, metrics=metrics, limit=limit,
                    force=args.force, verify=args.verify,
                )
                if summary and not summary.get("skipped"):
                    all_summaries.append(summary)
            except Exception as exc:  # one bad cell must not lose the other eleven
                print(f"    {dataset}: FAILED — {type(exc).__name__}: {exc}")
                failures.append({"scale": scale, "dataset": dataset, "error": str(exc)})

    # --- Sanity gate ---------------------------------------------------------
    # Gated on the answerable subset, against chance. A low overall Hit@5 on a shared
    # corpus usually means the answer is not in the corpus at all, not that retrieval
    # is broken; and the absolute figure legitimately falls as N grows. Chance for
    # top-5 over N documents is 5/N, so what a working retriever must clear is a large
    # multiple of that. See common.SANITY_CHANCE_MULTIPLE.
    print(f"\n{'=' * 78}")
    print(f"Sanity gate: Dense Hit@5 among answerable queries must exceed "
          f"{common.SANITY_CHANCE_MULTIPLE:.0f}x chance (5/N)")
    bad = []
    for s in all_summaries:
        chance = 5.0 / max(s["scale"], 1)
        ratio = s["dense_hit_at_5_answerable"] / chance if chance > 0 else 0.0
        s["chance_hit_at_5"] = round(chance, 6)
        s["chance_multiple"] = round(ratio, 1)
        if ratio < common.SANITY_CHANCE_MULTIPLE:
            bad.append(s)
            flag = "FAIL"
        elif s["dense_hit_at_5_answerable"] < common.WARN_HIT5_FLOOR:
            flag = "warn"
        else:
            flag = "ok"
        print(f"  {flag:>4}  scale {common.fmt_int(s['scale']):>7}  {s['dataset']:<9} "
              f"Hit@5 {s['dense_hit_at_5_answerable']:.3f} answerable "
              f"({s['n_answerable']}/{s['n_queries']}) = "
              f"{ratio:>6.0f}x chance | {s['dense_hit_at_5']:.3f} overall")
    print("  'warn' = above chance but low in absolute terms; look at it, but it is not "
          "on its own\n         evidence of a broken pipeline on a shared corpus.")

    if all_summaries:
        common.write_manifest(
            os.path.join(common.RESULTS_DIR, "retrieval_summary.json"),
            {"shards": all_summaries, "failures": failures,
             "sanity_chance_multiple": common.SANITY_CHANCE_MULTIPLE,
             "warn_hit5_floor": common.WARN_HIT5_FLOOR},
        )

    if failures:
        print(f"\n{len(failures)} shard(s) failed; the rest are on disk and usable.")
    if bad:
        print("\nSANITY GATE FAILED — do not use these numbers until the corpus or the "
              "gold labelling is fixed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
