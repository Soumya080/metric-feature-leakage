"""SparseBM25 must rank identically to src/retrieval/sparse.BM25Retriever.

This is the guard that keeps §3b (shared index, 5K-100K) comparable to §3
(500-passage pools). If the two settings used different ranking functions, any
change in the leak-vs-scale curve could be the retriever changing rather than the
corpus growing, and the whole experiment would be uninterpretable.
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for path in (PROJECT_ROOT, os.path.dirname(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from bm25_sparse import SparseBM25, tokenize  # noqa: E402
from src.retrieval.sparse import BM25Retriever  # noqa: E402


DOCS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Beijing is the capital of China and a major cultural centre.",
    "The Karate Kid (2010) was filmed on location in Beijing, China.",
    "Reinforcement learning optimises a policy against a reward signal.",
    "BM25 is a bag-of-words ranking function used by search engines.",
    "Dense retrieval encodes queries and passages into a shared vector space.",
    "The river bank was crowded; the dog slept while the fox ran.",
    "Punctuation, hyphens and CAPITALS should not change tokenisation!",
    "china china china repeated terms exercise term frequency saturation.",
    "A short doc.",
]

QUERIES = [
    "where does the karate kid 2010 take place",
    "capital of china",
    "china china",                      # repeated query term -> multiplicity path
    "bm25 ranking function",
    "quick brown fox",
    "term that appears nowhere zzzzz",  # all-OOV -> empty result
    "PUNCTUATION, hyphens!",
    "the",                              # ultra-common term
]


def _reference(docs):
    chunks = [
        {"content": t, "chunk_id": f"chunk_{i}", "chunk_index": i}
        for i, t in enumerate(docs)
    ]
    ref = BM25Retriever()
    ref.fit(chunks)
    return ref


@pytest.mark.parametrize("query", QUERIES)
def test_scores_match_reference(query):
    """Same score, to floating-point tolerance, for every retrieved document."""
    ref = _reference(DOCS)
    fast = SparseBM25().fit(DOCS)

    ref_hits = ref.retrieve(query, top_k=len(DOCS))
    idx, scores = fast.retrieve(query, top_k=len(DOCS))

    assert len(ref_hits) == len(idx), (
        f"different number of positive-scoring docs for {query!r}: "
        f"reference {len(ref_hits)} vs sparse {len(idx)}"
    )

    fast_by_doc = dict(zip(idx.tolist(), scores.tolist()))
    for hit in ref_hits:
        doc = hit["chunk_index"]
        assert doc in fast_by_doc, f"doc {doc} missing from sparse result for {query!r}"
        np.testing.assert_allclose(
            fast_by_doc[doc], hit["score"], rtol=1e-9, atol=1e-12,
            err_msg=f"score mismatch on doc {doc} for query {query!r}",
        )


@pytest.mark.parametrize("query", QUERIES)
def test_ranking_order_matches_reference(query):
    """Identical ordering wherever the reference's scores are not tied.

    Ties are excluded because sparse.py leaves them to Python's sort and we impose
    an explicit index tiebreak; agreeing on tied positions is not required for the
    comparison to be sound, agreeing on distinct scores is.
    """
    ref = _reference(DOCS)
    fast = SparseBM25().fit(DOCS)

    ref_hits = ref.retrieve(query, top_k=len(DOCS))
    idx, _ = fast.retrieve(query, top_k=len(DOCS))

    ref_order = [h["chunk_index"] for h in ref_hits]
    ref_scores = [h["score"] for h in ref_hits]

    for rank in range(len(ref_order) - 1):
        if abs(ref_scores[rank] - ref_scores[rank + 1]) < 1e-12:
            continue  # tied — order is not determined by the formula
        assert idx[rank] == ref_order[rank], (
            f"rank {rank} differs for {query!r}: "
            f"sparse={idx[rank]} reference={ref_order[rank]}"
        )


def test_tokenizer_matches_reference():
    ref = BM25Retriever()
    for text in DOCS + QUERIES:
        assert tokenize(text) == ref._tokenize(text), f"tokenisation differs on {text!r}"


def test_empty_result_for_all_oov_query():
    fast = SparseBM25().fit(DOCS)
    idx, scores = fast.retrieve("zzzzz qqqqq wwwww", top_k=5)
    assert idx.size == 0 and scores.size == 0


def test_top_k_truncates_without_changing_order():
    fast = SparseBM25().fit(DOCS)
    full_idx, _ = fast.retrieve("capital of china", top_k=len(DOCS))
    small_idx, _ = fast.retrieve("capital of china", top_k=3)
    assert small_idx.tolist() == full_idx[: len(small_idx)].tolist()
