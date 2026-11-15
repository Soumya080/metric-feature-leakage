"""Is the negative result a property of the task, or of our hand-crafted features?

The standard reply to any negative learnability result is that the representation was too
weak. Ours invites it: the label-free retrieval features are two summary statistics (top-5
score mean and standard deviation), where the query-performance-prediction literature offers
considerably more. So the objection has to be answered rather than deflected.

This swaps the representation and holds everything else fixed. Instead of the 61 hand-crafted
dimensions, the classifier sees frozen sentence-embedding vectors of the query text --- the
same encoder the dense retriever uses, so the features are exactly what a router would have
available at inference with no extra machinery. Same informative-query filter, same repeated
stratified CV, same estimator, same seed, same majority baseline.

Three outcomes, and they mean different things:

  * embeddings also land at the majority baseline --- the negative is about the target, not
    the features, and the objection is closed.
  * embeddings do substantially better --- the negative is about OUR features, the paper's
    claim must narrow.
  * embeddings do worse --- expected if 384 dimensions overfit at n~3,000; reported as such
    rather than as evidence for anything.

Raw 384-dim vectors are also reduced by PCA (fit on training folds only, inside the CV) as a
second arm, since the paper's own feature set uses 32 PCA components and the comparison should
not turn on dimensionality alone.

Usage:
    python experiments/router_experiments/embedding_control.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (PROJECT_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from learnability_analysis import (  # noqa: E402
    ACTIONS, N_REPEATS, N_SPLITS, load_queries, reward_matrix,
)
from canonical_evaluation import get_dataset_paths  # noqa: E402

SEED = 0
DATASETS = ("nq", "triviaqa", "hotpotqa")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
EMB_PATH = os.path.join(PROJECT_ROOT, "experiments", "corpus_v3", "artifacts",
                        "query_embeddings_{dataset}.npy")


def _score(X, y, estimator):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    s = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy")
    return float(s.mean()), float(s.std())


_ENCODER = [None]


def _query_embeddings(queries, cache_path):
    """Frozen embeddings of the query text, encoded here if the cache does not match.

    The cache under corpus_v3/artifacts/ is written by 02_embed_corpus.py, which has a
    --smoke mode that stops at 200 queries. A partial cache silently paired with a full
    query list would align 200 vectors against ~1,200 labels and produce a number that
    looks plausible and means nothing, so the count is checked and a mismatch triggers a
    recompute rather than a skip.
    """
    if os.path.exists(cache_path):
        cached = np.load(cache_path, mmap_mode="r")
        n = cached.shape[0]
        if n == len(queries):
            out = np.asarray(cached, dtype=np.float64)
            del cached
            return out
        del cached
        print(f"    cache has {n} vectors for {len(queries)} queries — re-encoding")

    if _ENCODER[0] is None:
        from sentence_transformers import SentenceTransformer
        # Same encoder the dense retriever uses, so these are features a router would
        # genuinely have at inference rather than an unavailable oracle representation.
        _ENCODER[0] = SentenceTransformer("all-MiniLM-L6-v2")

    # `queries` is a dict keyed by query id, and reward_matrix() builds X and R by iterating
    # list(queries). Encoding in that same order is what keeps row i of the embedding matrix
    # aligned with row i of the labels; iterating the dict any other way would silently
    # scramble the correspondence.
    qids = list(queries)
    texts = [queries[q].get("question", "") for q in qids]
    missing = sum(1 for t in texts if not t)
    if missing:
        raise ValueError(f"{missing}/{len(texts)} queries have no 'question' text in the env "
                         f"data; cannot build a text representation from it")
    vectors = _ENCODER[0].encode(texts, batch_size=256, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp.npy"
    np.save(tmp, vectors)
    os.replace(tmp, cache_path)
    return vectors.astype(np.float64)


def main() -> int:
    paths = get_dataset_paths("")
    per_ds, pooled = {}, {"hand": [], "emb": [], "R": []}

    for ds in DATASETS:
        env, emb_file = paths.get(ds), EMB_PATH.format(dataset=ds)
        if not (env and os.path.exists(env)):
            print(f"  {ds}: env data missing, skipped")
            continue

        queries = load_queries(env)
        _, X_hand, R = reward_matrix(queries, "mrr")
        X_emb = _query_embeddings(queries, emb_file)

        # Belt and braces: _query_embeddings guarantees the count, but a silent misalignment
        # here would produce a plausible-looking number from mismatched rows.
        assert X_emb.shape[0] == X_hand.shape[0], (
            f"{ds}: {X_emb.shape[0]} embeddings vs {X_hand.shape[0]} queries")

        mask = (R == R.max(axis=1, keepdims=True)).sum(axis=1) == 1
        Xh, Xe, y = X_hand[mask], X_emb[mask], R[mask].argmax(axis=1)
        counts = np.bincount(y, minlength=len(ACTIONS))
        if (counts[counts > 0] < N_SPLITS).any():
            print(f"  {ds}: a class has fewer than {N_SPLITS} members, skipped")
            continue
        majority = float(counts.max() / len(y))

        gbt = GradientBoostingClassifier(random_state=SEED)
        # PCA inside the pipeline so it is fit on training folds only; fitting it on all rows
        # first would be exactly the preprocessing leak this project is about.
        pca = make_pipeline(PCA(n_components=32, random_state=SEED),
                           GradientBoostingClassifier(random_state=SEED))

        h_acc, h_sd = _score(Xh, y, gbt)
        e_acc, e_sd = _score(Xe, y, gbt)
        p_acc, p_sd = _score(Xe, y, pca)

        per_ds[ds] = {
            "n_informative": int(len(y)), "majority": majority,
            "hand_61": {"acc": h_acc, "std": h_sd, "lift": h_acc - majority},
            "embed_384": {"acc": e_acc, "std": e_sd, "lift": e_acc - majority},
            "embed_pca32": {"acc": p_acc, "std": p_sd, "lift": p_acc - majority},
        }
        print(f"  {ds:9s} n={len(y):5d} majority={majority*100:5.2f}%  "
              f"hand={h_acc*100:5.2f}({(h_acc-majority)*100:+.2f})  "
              f"emb384={e_acc*100:5.2f}({(e_acc-majority)*100:+.2f})  "
              f"embPCA={p_acc*100:5.2f}({(p_acc-majority)*100:+.2f})")

        pooled["hand"].append(Xh)
        pooled["emb"].append(Xe)
        pooled["R"].append(R[mask])

    out = {"seed": SEED, "cv": f"{N_SPLITS}x{N_REPEATS}", "per_dataset": per_ds}

    if len(pooled["hand"]) == len(DATASETS):
        Xh = np.vstack(pooled["hand"]); Xe = np.vstack(pooled["emb"])
        y = np.vstack(pooled["R"]).argmax(axis=1)
        counts = np.bincount(y, minlength=len(ACTIONS))
        majority = float(counts.max() / len(y))
        gbt = GradientBoostingClassifier(random_state=SEED)
        pca = make_pipeline(PCA(n_components=32, random_state=SEED),
                            GradientBoostingClassifier(random_state=SEED))
        h_acc, h_sd = _score(Xh, y, gbt)
        e_acc, e_sd = _score(Xe, y, gbt)
        p_acc, p_sd = _score(Xe, y, pca)
        out["pooled"] = {
            "n_informative": int(len(y)), "majority": majority,
            "hand_61": {"acc": h_acc, "std": h_sd, "lift": h_acc - majority},
            "embed_384": {"acc": e_acc, "std": e_sd, "lift": e_acc - majority},
            "embed_pca32": {"acc": p_acc, "std": p_sd, "lift": p_acc - majority},
        }
        print(f"\n  {'pooled':9s} n={len(y):5d} majority={majority*100:5.2f}%  "
              f"hand={h_acc*100:5.2f}({(h_acc-majority)*100:+.2f})  "
              f"emb384={e_acc*100:5.2f}({(e_acc-majority)*100:+.2f})  "
              f"embPCA={p_acc*100:5.2f}({(p_acc-majority)*100:+.2f})")

    path = os.path.join(RESULTS_DIR, "embedding_control.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
