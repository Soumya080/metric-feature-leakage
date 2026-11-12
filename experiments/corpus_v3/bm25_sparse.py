"""Vectorised Okapi BM25, numerically identical to `src/retrieval/sparse.py`.

Why not just use BM25Retriever
------------------------------
`src/retrieval/sparse.py` scores by looping over every document in Python for every
query. At the 500-passage scale that is fine. At 100K documents x ~6,000 queries it
is ~600M inner iterations per dataset, which is hours.

Why not bm25s / rank_bm25
-------------------------
Neither is installed, and more importantly both implement a DIFFERENT BM25 variant
(different IDF floor, different tokeniser). If the shared-index setting used a
different ranking function than the 500-passage setting, then any difference in the
leak-vs-scale curve would be confounded with the change of retriever. The whole
point of §3b is that only the corpus size changes.

So: same formula, same constants, same tokeniser as sparse.py — just precomputed
into a sparse matrix so scoring is one matmul.

The equivalence is asserted, not assumed: see tests/test_bm25_parity.py.

Formulation (verbatim from sparse.py)
    idf(t)      = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1.0)
    score(q, d) = SUM over occurrences of t in q of
                      idf(t) * tf(t,d) * (k1 + 1)
                      / (tf(t,d) + k1 * (1 - b + b * len(d) / avgdl))

Note the sum is over *occurrences*, not distinct terms: sparse.py iterates
`for token in query_tokens` without deduplicating, so a query term appearing twice
contributes twice. Multiplying by the query term count reproduces that exactly.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
from scipy import sparse


def tokenize(text: str) -> List[str]:
    """Character-for-character the tokeniser in src/retrieval/sparse.py."""
    cleaned = "".join([c.lower() if c.isalnum() or c.isspace() else " " for c in text])
    return cleaned.split()


class SparseBM25:
    """BM25 over a fixed corpus, precomputed as a CSR term-weight matrix."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}
        self.matrix: sparse.csr_matrix | None = None
        self.n_docs = 0

    def fit(self, documents: Sequence[str]) -> "SparseBM25":
        """Build the CSR matrix W with W[d, t] = the per-(doc, term) BM25 weight.

        Because the weight depends only on tf(t,d) and len(d) — never on the query —
        it can be fully precomputed, leaving scoring as W @ q.
        """
        self.n_docs = len(documents)
        if self.n_docs == 0:
            raise ValueError("cannot fit BM25 on an empty corpus")

        doc_term_counts: List[Counter] = []
        doc_lengths = np.empty(self.n_docs, dtype=np.float64)
        df: Counter = Counter()

        for i, text in enumerate(documents):
            tokens = tokenize(text)
            counts = Counter(tokens)
            doc_term_counts.append(counts)
            doc_lengths[i] = len(tokens)
            df.update(counts.keys())

        avgdl = float(doc_lengths.sum() / self.n_docs) if self.n_docs else 0.0
        self.vocab = {term: j for j, term in enumerate(sorted(df))}

        idf = np.empty(len(self.vocab), dtype=np.float64)
        for term, j in self.vocab.items():
            freq = df[term]
            idf[j] = np.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

        # Length normalisation is constant within a document.
        denom_len = self.k1 * (1.0 - self.b + self.b * (doc_lengths / (avgdl or 1.0)))

        n_nonzero = sum(len(c) for c in doc_term_counts)
        rows = np.empty(n_nonzero, dtype=np.int32)
        cols = np.empty(n_nonzero, dtype=np.int32)
        vals = np.empty(n_nonzero, dtype=np.float64)

        cursor = 0
        for i, counts in enumerate(doc_term_counts):
            if not counts:
                continue
            span = len(counts)
            term_idx = np.fromiter(
                (self.vocab[t] for t in counts), dtype=np.int32, count=span
            )
            tf = np.fromiter(counts.values(), dtype=np.float64, count=span)
            weight = idf[term_idx] * tf * (self.k1 + 1.0) / (tf + denom_len[i])
            rows[cursor:cursor + span] = i
            cols[cursor:cursor + span] = term_idx
            vals[cursor:cursor + span] = weight
            cursor += span

        self.matrix = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(self.n_docs, len(self.vocab))
        )
        return self

    def score(self, query: str) -> np.ndarray:
        """Dense score vector over all documents. Unseen query terms score 0."""
        if self.matrix is None:
            raise RuntimeError("call fit() before score()")

        counts = Counter(tokenize(query))
        cols, vals = [], []
        for term, count in counts.items():
            j = self.vocab.get(term)
            if j is not None:
                cols.append(j)
                # Query term multiplicity, matching sparse.py's un-deduplicated loop.
                vals.append(float(count))

        if not cols:
            return np.zeros(self.n_docs, dtype=np.float64)

        q = sparse.csr_matrix(
            (vals, ([0] * len(cols), cols)), shape=(1, len(self.vocab))
        )
        return np.asarray((self.matrix @ q.T).todense()).ravel()

    def retrieve(self, query: str, top_k: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Top-k document indices and their scores, descending.

        Only strictly-positive scores are returned, matching sparse.py, which drops
        `score > 0` failures. Ties broken by ascending document index so the ranking
        is deterministic across runs and machines.
        """
        scores = self.score(query)
        positive = int((scores > 0).sum())
        if positive == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        take = min(top_k, positive)
        # argpartition then a stable sort of the shortlist: O(n) rather than O(n log n),
        # which matters at 100K docs x 6K queries.
        shortlist = np.argpartition(-scores, take - 1)[:take] if take < self.n_docs \
            else np.arange(self.n_docs)
        order = np.lexsort((shortlist, -scores[shortlist]))
        idx = shortlist[order][:take]
        return idx.astype(np.int64), scores[idx]
