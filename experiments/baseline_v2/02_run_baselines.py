"""
02_run_baselines.py — Baseline v2: Run BM25, Dense, and Hybrid on real Wikipedia passages.

Identical experiment design to v1 but using the real-passage corpus from v2 data prep.
This is the production-grade experiment for the paper.

Usage:
    cd <repo>
    python experiments/baseline_v2/02_run_baselines.py
    python experiments/baseline_v2/02_run_baselines.py --dataset nq --max_questions 10
    python experiments/baseline_v2/02_run_baselines.py --dataset both --top_k 1 3 5 10 20
"""

import json
import os
import re
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from src.embeddings.encoder import EmbeddingEncoder
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense_faiss import FAISSRetriever
from src.retrieval.hybrid import reciprocal_rank_fusion

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "experiments/baseline_v2/data"
RESULTS_DIR = "experiments/baseline_v2/results"
TOP_K_VALUES = [1, 3, 5, 10, 20]   # Extended k values for thorough analysis
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================================
# METRICS — Standard QA retrieval metrics (paper-grade)
# ============================================================
def normalize_answer(text):
    """Lowercase, strip articles and punctuation for fair comparison."""
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def _word_boundary_contains(needle_norm, haystack_norm):
    """Whole-word/whole-phrase containment check on already-normalized text.

    Plain `needle in haystack` substring checks produce false positives whenever the
    gold answer is a short or common token that happens to appear inside a longer word
    (e.g. "may" matching inside "maybe", "china" matching inside "chinatown", "a"
    matching almost anything). This mirrors the word-boundary regex already used by
    src/evaluation/metrics.py's `_answer_matches_context`, so hit/MRR computed here are
    consistent with the rest of the codebase.
    """
    if not needle_norm:
        return False
    pattern = r'(?<!\w)' + re.escape(needle_norm) + r'(?!\w)'
    return bool(re.search(pattern, haystack_norm))

def exact_match_score(prediction, ground_truths):
    """Standard Exact Match: prediction must exactly equal a ground truth (normalized)."""
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == pred_norm:
            return 1.0
    return 0.0

def substring_match_score(prediction, ground_truths):
    """Whole-word/phrase containment: any ground truth appears in prediction."""
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        if _word_boundary_contains(normalize_answer(gt), pred_norm):
            return 1.0
    return 0.0

def f1_score(prediction, ground_truths):
    """Token-level F1 between prediction and best-matching ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gt_tokens) if gt_tokens else 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)
    return best_f1

def check_hit(retrieved_contents, answers):
    """Check if any answer appears in any retrieved passage (standard Hit@K)."""
    for content in retrieved_contents:
        content_norm = normalize_answer(content)
        for ans in answers:
            if _word_boundary_contains(normalize_answer(ans), content_norm):
                return True
    return False

def compute_mrr(retrieved_contents, answers):
    """Mean Reciprocal Rank — rank of first relevant result."""
    for rank, content in enumerate(retrieved_contents, start=1):
        content_norm = normalize_answer(content)
        for ans in answers:
            if _word_boundary_contains(normalize_answer(ans), content_norm):
                return 1.0 / rank
    return 0.0

def compute_recall_at_k(retrieved_contents, answers, k):
    """Recall@K — fraction of unique answers found in top-k passages."""
    if not answers:
        return 0.0
    context = " ".join(retrieved_contents[:k])
    context_norm = normalize_answer(context)
    found = sum(1 for ans in set(answers) if normalize_answer(ans) in context_norm)
    return found / len(set(answers))


# ============================================================
# RETRIEVER RUNNERS
# ============================================================
def build_chunk_list(corpus_texts):
    """Convert passage strings to chunk dicts matching pipeline format."""
    chunks = []
    for i, text in enumerate(corpus_texts):
        chunks.append({
            "content": text,
            "chunk_id": f"chunk_{i}",
            "chunk_index": i,
        })
    return chunks

def first_answer_bearing_rank(ranked_contents, answers):
    """1-based rank of the first retrieved passage containing an answer, else None.

    Same word-boundary matching as compute_mrr, so this is exactly "the rank MRR would
    have used" — just measured over the FULL corpus ranking instead of being truncated at
    k and collapsed to 0. That truncation is what makes the reward degenerate: at k=5,
    70-83%% of queries score 0 for every retriever, every action ties, and there is nothing
    for a router to learn from. Ranking the whole corpus keeps the signal graded (a gold
    passage at rank 40 is genuinely better than one at rank 300) and is defined for
    essentially every query, since the gold passage always contains the answer.
    """
    for rank, content in enumerate(ranked_contents, start=1):
        content_norm = normalize_answer(content)
        for ans in answers:
            if _word_boundary_contains(normalize_answer(ans), content_norm):
                return rank
    return None


def cross_scoring_features(results, sparse_candidates, dense_candidates, top_n=5):
    """Score each retriever's own top-n passages under the OTHER retriever's function.

    Every score feature the router currently sees is a retriever's opinion of its own
    output: BM25's margin over BM25's top-1, Dense's spread over Dense's ranking. That is
    circular — a retriever is always confident about what it just ranked first, so the
    feature describes the retriever's habits more than whether it actually succeeded.

    The informative question is the reverse one: does the OTHER scorer agree? If BM25's
    top passages also sit close to the query in embedding space, BM25 found something
    semantically real rather than a lexical coincidence. If Dense's top passages also
    score well under BM25, Dense found passages that genuinely contain the query's terms.
    Disagreement is evidence that one of them failed — and it is evidence no
    single-retriever confidence score can express, because it requires evaluating one
    system's output with a different system's notion of relevance.

    Whether BM25 or Dense wins on a query is a property of the query-passage interaction,
    not of the query alone, which is why query-side features cannot predict it. These
    features are measured after retrieval, so the interaction is observable.

    Costs no retrieval: sparse_candidates and dense_candidates already rank the FULL
    corpus (both were computed for the Hybrid fusion just above), so scoring any passage
    under either function is a dict lookup. BM25 drops zero-score passages
    (src/retrieval/sparse.py), so a chunk absent from bm25_map genuinely scored 0 under
    BM25 and is recorded as 0 rather than as missing.
    """
    bm25_map = {r["chunk_index"]: r["score"] for r in sparse_candidates}
    dense_map = {r["chunk_index"]: r["score"] for r in dense_candidates}

    def summarize(indices, score_map):
        vals = [float(score_map.get(i, 0.0)) for i in indices]
        if not vals:
            return {"mean": 0.0, "max": 0.0}
        return {"mean": float(np.mean(vals)), "max": float(max(vals))}

    for name in ("bm25", "dense", "hybrid"):
        idx = results[name].get("chunk_indices", [])[:top_n]
        results[name]["cross_scores"] = {
            "bm25": summarize(idx, bm25_map),
            "dense": summarize(idx, dense_map),
            "n_scored": len(idx),
        }

    # Corpus-wide maxima so downstream can divide the raw BM25 scale away. BM25 scores are
    # unbounded and depend on the query's IDF profile, making raw values incomparable
    # across queries; cosine similarity is already bounded. Both lists are sorted
    # descending, so element 0 is the maximum over the whole corpus.
    results["cross_score_norm"] = {
        "bm25_max": float(sparse_candidates[0]["score"]) if sparse_candidates else 0.0,
        "dense_max": float(dense_candidates[0]["score"]) if dense_candidates else 0.0,
    }
    return results


# Weights swept for the convex-combination fusion. 0.0 = pure Dense, 1.0 = pure BM25.
# Every alpha's gold rank is persisted so the weight can be CHOSEN on the training split
# post-hoc; picking it here, against all queries, would be exactly the kind of test-set
# tuning this codebase is trying to stop doing.
CONVEX_ALPHAS = tuple(round(a / 10.0, 1) for a in range(11))


def convex_fusion_gold_ranks(sparse_candidates, dense_candidates, answers,
                             alphas=CONVEX_ALPHAS):
    """Gold rank under alpha * norm(BM25) + (1 - alpha) * norm(Dense), per alpha.

    Why this exists: the paper's fusion baseline. Hybrid is already Reciprocal Rank Fusion
    over these same two lists, so "why not fuse instead of route?" is partly answered
    already -- but Bruch, Gai & Ingber (TOIS 2023) report that a tuned convex combination
    of normalised scores beats RRF both in- and out-of-domain, so RRF alone is not the
    strongest fusion to compare against.

    Min-max normalised per retriever because BM25 (~0-30) and cosine similarity (~0.2-0.8)
    are on incommensurable scales; an unnormalised sum is just BM25 with a rounding error.

    BM25 drops zero-score passages (src/retrieval/sparse.py), so a passage absent from the
    sparse list scores 0 after normalisation -- the bottom of BM25's range, which is what
    "BM25 gave it nothing" means. Dense returns the full corpus, so it defines the universe.
    """
    if not answers:
        return None

    dense_score = {c["chunk_index"]: float(c["score"]) for c in dense_candidates}
    sparse_score = {c["chunk_index"]: float(c["score"]) for c in sparse_candidates}
    content = {c["chunk_index"]: c["content"] for c in dense_candidates}
    for c in sparse_candidates:                      # in case sparse saw anything dense did not
        content.setdefault(c["chunk_index"], c["content"])

    def _norm(scores):
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        if hi - lo < 1e-12:
            return {k: 0.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    nd, ns = _norm(dense_score), _norm(sparse_score)
    ids = list(content)

    out = {}
    for alpha in alphas:
        combined = [(alpha * ns.get(i, 0.0) + (1.0 - alpha) * nd.get(i, 0.0), i) for i in ids]
        # Sort by score desc, then chunk_index asc: ties must not depend on dict order, or
        # the fusion's reported rank changes between runs on the same data.
        combined.sort(key=lambda t: (-t[0], t[1]))
        ranked = [content[i] for _, i in combined]
        out[f"{alpha:.1f}"] = first_answer_bearing_rank(ranked, answers)
    return out


def run_single_query_all_retrievers(question, corpus_texts, encoder, top_k=20,
                                    answers=None):
    """
    Run BM25, Dense, and Hybrid on a single question against a corpus.
    Returns per-retriever results.

    When `answers` is given, each retriever additionally reports `gold_rank`: the rank of
    the first answer-bearing passage across the ENTIRE corpus (not just top_k). That is the
    dense training reward — see first_answer_bearing_rank above.
    """
    chunks = build_chunk_list(corpus_texts)

    # Embed all chunks (batched for speed)
    texts = [c["content"] for c in chunks]
    embeddings = encoder.encode_batch(texts)
    for i, chunk in enumerate(chunks):
        chunk["chunk_embedding"] = embeddings[i]

    # Embed query
    query_embedding = encoder.encode(question)

    results = {}

    # --- BM25 ---
    t0 = time.time()
    bm25 = BM25Retriever()
    bm25.fit(chunks)
    bm25_results = bm25.retrieve(question, top_k=top_k)
    bm25_time = time.time() - t0
    results["bm25"] = {
        "retrieved": [r["content"] for r in bm25_results],
        "scores": [r["score"] for r in bm25_results],
        # Corpus positions of the retrieved passages. Needed to measure whether BM25 and
        # Dense actually agree on WHICH passages are relevant — a label-free signal, since
        # it compares the two result lists to each other and never to the gold answer.
        # See _cross_retriever_features() in src/router/prepare_env_data.py.
        "chunk_indices": [r["chunk_index"] for r in bm25_results],
        "latency_ms": bm25_time * 1000,
    }

    # --- Dense (FAISS) ---
    t0 = time.time()
    faiss_ret = FAISSRetriever(encoder.dimension)
    faiss_ret.add_chunks(chunks)
    faiss_results = faiss_ret.retrieve(query_embedding, top_k=top_k)
    faiss_time = time.time() - t0
    results["dense"] = {
        "retrieved": [r["content"] for r in faiss_results],
        "scores": [r["score"] for r in faiss_results],
        "chunk_indices": [r["chunk_index"] for r in faiss_results],
        "latency_ms": faiss_time * 1000,
    }

    # --- Hybrid (RRF) ---
    # Fuses the retrievers' FULL ranked lists, not a truncated candidate pool. The previous
    # version fused only the top max(20, 2*top_k) of each, which made "Hybrid" a different
    # system depending on the requested k — and left its full-corpus ranking inconsistent
    # with its own reported MRR@k. Fusing the complete lists makes Hybrid one well-defined
    # retriever and removes an arbitrary depth hyperparameter. Cost is negligible at this
    # corpus size (RRF is a dict merge over a few hundred items).
    n_corpus = len(chunks)
    t0 = time.time()
    dense_candidates = faiss_ret.retrieve(query_embedding, top_k=n_corpus)
    sparse_candidates = bm25.retrieve(question, top_k=n_corpus)
    hybrid_full = reciprocal_rank_fusion(dense_candidates, sparse_candidates, top_n=n_corpus)
    hybrid_results = hybrid_full[:top_k]
    hybrid_time = time.time() - t0
    results["hybrid"] = {
        "retrieved": [r["content"] for r in hybrid_results],
        "scores": [r["score"] for r in hybrid_results],
        "chunk_indices": [r["chunk_index"] for r in hybrid_results],
        "latency_ms": hybrid_time * 1000,
    }

    # --- Full-corpus ranks, for the dense reward ---
    # Reuses the complete ranked lists already computed for the Hybrid fusion above, so
    # this adds no retrieval work and is guaranteed consistent with the reported MRR@k.
    if answers:
        full_lists = {
            "bm25": sparse_candidates,
            "dense": dense_candidates,
            "hybrid": hybrid_full,
        }
        for name, full in full_lists.items():
            rank = first_answer_bearing_rank([r["content"] for r in full], answers)
            # None has a real meaning: the retriever never ranked an answer-bearing passage
            # at all. BM25 in particular drops zero-score passages (src/retrieval/sparse.py),
            # so a query sharing no terms with any answer passage returns nothing for it.
            # Downstream treats None as rank n_corpus+1 — strictly worse than any achieved
            # rank — rather than dropping the query or inventing a finite rank.
            results[name]["gold_rank"] = rank
            results[name]["n_ranked"] = len(full)
        results["n_corpus"] = n_corpus

        # Convex-combination fusion, swept over alpha. Computed here because it needs the
        # FULL normalised score vectors, which exist only at this point -- the per-query
        # record persists just the top 10. Reuses the ranked lists above, so no extra
        # retrieval. The weight is chosen later, on the training split.
        results["convex_fusion_gold_rank"] = convex_fusion_gold_ranks(
            sparse_candidates, dense_candidates, answers)

    # --- Cross-retriever scoring, for post-retrieval routing evidence ---
    # Label-free (never touches `answers`), and reuses the full ranked lists above, so it
    # adds no retrieval work. See cross_scoring_features for why this is the one signal
    # that is not circular.
    cross_scoring_features(results, sparse_candidates, dense_candidates)

    # The query embedding is already computed above for dense retrieval; persisting it
    # costs nothing here and gives the router a semantic view of the query that the
    # handcrafted surface features (length, stopword ratio, question word) cannot express.
    results["query_embedding"] = [float(x) for x in np.asarray(query_embedding).ravel()]

    return results


def run_baseline_experiment(dataset_name, data, encoder, top_k_values):
    """Run all retrievers on all questions, collect per-query results."""
    print(f"\n{'='*60}")
    print(f"RUNNING BASELINE v2: {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"  Questions: {len(data)}")
    print(f"  Top-K values: {top_k_values}")
    print(f"  Retrievers: BM25, Dense (FAISS), Hybrid (RRF)")
    print(f"  Corpus type: Real Wikipedia passages")
    print()

    all_results = []
    gold_type_counts = {"real_wikipedia": 0, "constructed": 0}

    for i, item in enumerate(tqdm(data, desc=f"Evaluating {dataset_name}")):
        question = item["question"]
        answers = item["answers"]
        corpus = item["corpus"]
        gold_type = item.get("gold_type", "unknown")
        gold_type_counts[gold_type] = gold_type_counts.get(gold_type, 0) + 1

        max_k = max(top_k_values)
        retriever_results = run_single_query_all_retrievers(
            question, corpus, encoder, top_k=max_k, answers=answers
        )

        query_result = {
            "query_id": i,
            "question": question,
            "answers": answers,
            "corpus_size": len(corpus),
            "gold_type": gold_type,
            "query_embedding": retriever_results.get("query_embedding", []),
            "n_corpus": retriever_results.get("n_corpus", len(corpus)),
            # Per-query maxima of each scoring function over the whole corpus. BM25 scores
            # are unbounded and query-dependent, so cross-scores must be normalised by
            # these before they are comparable across queries.
            "cross_score_norm": retriever_results.get("cross_score_norm", {}),
            # {alpha: gold_rank} for the convex-combination fusion baseline. Persisted per
            # alpha rather than as a single tuned number so the weight can be selected on
            # the training split downstream; selecting it here would tune on the test set.
            "convex_fusion_gold_rank": retriever_results.get("convex_fusion_gold_rank"),
            "retrievers": {},
        }

        for ret_name in ["bm25", "dense", "hybrid"]:
            ret_data = retriever_results[ret_name]

            # Raw, label-free retrieval confidence scores (available at inference time,
            # unlike Hit/MRR/F1/EM below which require knowing the gold answer). These
            # are what the router's "avg_bm25"/"avg_dense" features must be built from —
            # see src/router/prepare_env_data.py. Previously these scores were computed
            # (see `results["bm25"]["scores"]` etc. above) but discarded before being
            # written to disk, so prepare_env_data.py fell back to reusing the
            # label-dependent MRR value as a "proxy," which leaks 2 of 3 possible
            # rewards directly into the router's input features.
            raw_scores = ret_data.get("scores", [])
            top5_scores = raw_scores[:5]
            top1_score = float(raw_scores[0]) if raw_scores else 0.0
            mean_score_top5 = float(np.mean(top5_scores)) if top5_scores else 0.0
            std_score_top5 = float(np.std(top5_scores)) if top5_scores else 0.0
            # Full top-10 distribution (not just the top1/mean/std summary above) — needed
            # for scale-free within-retriever confidence signals (margin, entropy) that
            # compare meaningfully across BM25 (~0-30) and cosine similarity (~0.2-0.8),
            # which the summary stats alone cannot support. See PLAN_leakage_harness.md.
            scores_top10 = [float(s) for s in raw_scores[:10]]
            # Which passages were retrieved (corpus positions), for cross-retriever
            # agreement. Also label-free — see the note where chunk_indices is built.
            retrieved_ids_top10 = [int(i) for i in ret_data.get("chunk_indices", [])[:10]]

            query_result["retrievers"][ret_name] = {
                "latency_ms": ret_data["latency_ms"],
                "top1_score": top1_score,
                "mean_score_top5": mean_score_top5,
                "std_score_top5": std_score_top5,
                "scores_top10": scores_top10,
                "retrieved_ids_top10": retrieved_ids_top10,
                # Rank of the first answer-bearing passage over the FULL corpus. Basis of
                # the dense reward 1/gold_rank, which unlike MRR@k is non-zero for
                # essentially every query. None = no answer-bearing passage exists.
                "gold_rank": ret_data.get("gold_rank"),
                # This retriever's own top-5 scored under BOTH scoring functions. The
                # cross term (BM25's passages under cosine, Dense's under BM25) is the
                # only label-free evidence available about whether this retriever's
                # output is actually good, as opposed to whether it is confident.
                "cross_scores": ret_data.get("cross_scores", {}),
                "metrics_at_k": {},
            }

            for k in top_k_values:
                top_k_contents = ret_data["retrieved"][:k]

                hit = check_hit(top_k_contents, answers)
                mrr = compute_mrr(ret_data["retrieved"][:k], answers)
                recall = compute_recall_at_k(ret_data["retrieved"], answers, k)

                # F1/EM on concatenated retrieved text
                concat_text = " ".join(top_k_contents)
                f1 = f1_score(concat_text, answers)
                em = substring_match_score(concat_text, answers)

                query_result["retrievers"][ret_name]["metrics_at_k"][str(k)] = {
                    "hit": hit,
                    "mrr": mrr,
                    "recall": recall,
                    "f1": f1,
                    "em": em,
                }

        all_results.append(query_result)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(data)} queries...")

    print(f"\n  Gold passage types: {gold_type_counts}")
    return all_results


def print_summary(results, dataset_name, k=5):
    """Print a summary table of results at a specific k."""
    print(f"\n{'='*75}")
    print(f"RESULTS SUMMARY: {dataset_name.upper()} (top_k={k})")
    print(f"{'='*75}")

    retrievers = ["bm25", "dense", "hybrid"]
    metrics = ["hit", "mrr", "recall", "f1", "em"]

    header = f"{'Retriever':<12}"
    for m in metrics:
        header += f" | {m.upper():>8}"
    header += f" | {'Latency':>10}"
    print(header)
    print("-" * 75)

    for ret in retrievers:
        row = f"{ret:<12}"
        for m in metrics:
            values = [r["retrievers"][ret]["metrics_at_k"][str(k)][m] for r in results]
            mean_val = np.mean(values)
            if m == "hit":
                row += f" | {mean_val:>8.1%}"
            else:
                row += f" | {mean_val:>8.4f}"

        latencies = [r["retrievers"][ret]["latency_ms"] for r in results]
        row += f" | {np.mean(latencies):>8.2f}ms"
        print(row)

    print("-" * 75)

    # Quick oracle preview
    oracle_hits = []
    for r in results:
        any_hit = any(
            r["retrievers"][ret]["metrics_at_k"][str(k)]["hit"]
            for ret in retrievers
        )
        oracle_hits.append(int(any_hit))

    best_fixed_hit = max(
        np.mean([r["retrievers"][ret]["metrics_at_k"][str(k)]["hit"] for r in results])
        for ret in retrievers
    )
    oracle_hit = np.mean(oracle_hits)
    headroom = oracle_hit - best_fixed_hit

    print(f"\n  Oracle Hit@{k}: {oracle_hit:.1%} | Best Fixed: {best_fixed_hit:.1%} | "
          f"HEADROOM: {headroom:.1%} ({headroom/max(0.001,best_fixed_hit)*100:.1f}% relative)")


def main():
    parser = argparse.ArgumentParser(description="Run baseline v2 retrieval experiments")
    parser.add_argument("--dataset", type=str, default="all", choices=["nq", "triviaqa", "hotpotqa", "both", "all"])
    parser.add_argument("--top_k", type=int, nargs="+", default=TOP_K_VALUES)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override where *_prepared.json is read from (default: "
                             "experiments/baseline_v2/data). Use with a scratch dir for "
                             "smoke tests so the real ~577MB datasets are neither read "
                             "(slow, memory-heavy) nor depended on.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override the output directory (default: experiments/baseline_v2/results). "
                             "ALWAYS use this for smoke tests / --max_questions runs — the default "
                             "directory holds the real per-query results, and a small run will silently "
                             "overwrite the full dataset's results file.")
    args = parser.parse_args()

    results_dir = args.output_dir or RESULTS_DIR
    if args.max_questions and not args.output_dir:
        print(f"WARNING: --max_questions was passed without --output_dir. This will overwrite "
              f"the FULL per-query results in {RESULTS_DIR} with a {args.max_questions}-query file. "
              f"Pass --output_dir to write elsewhere instead.")

    data_dir = args.data_dir or DATA_DIR

    os.makedirs(results_dir, exist_ok=True)

    print("Loading embedding model...")
    encoder = EmbeddingEncoder(EMBEDDING_MODEL)

    datasets_to_run = []
    if args.dataset in ["nq", "both", "all"]:
        nq_path = os.path.join(data_dir, "nq_prepared.json")
        if os.path.exists(nq_path):
            with open(nq_path, "r", encoding="utf-8") as f:
                nq_data = json.load(f)
            if args.max_questions:
                nq_data = nq_data[:args.max_questions]
            datasets_to_run.append(("nq", nq_data))
        else:
            print(f"WARNING: {nq_path} not found. Run 01_prepare_data.py first.")

    if args.dataset in ["triviaqa", "both", "all"]:
        tqa_path = os.path.join(data_dir, "triviaqa_prepared.json")
        if os.path.exists(tqa_path):
            with open(tqa_path, "r", encoding="utf-8") as f:
                tqa_data = json.load(f)
            if args.max_questions:
                tqa_data = tqa_data[:args.max_questions]
            datasets_to_run.append(("triviaqa", tqa_data))
        else:
            print(f"WARNING: {tqa_path} not found. Run 01_prepare_data.py first.")

    if args.dataset in ["hotpotqa", "all"]:
        hotpot_path = os.path.join(data_dir, "hotpotqa_prepared.json")
        if os.path.exists(hotpot_path):
            with open(hotpot_path, "r", encoding="utf-8") as f:
                hotpot_data = json.load(f)
            if args.max_questions:
                hotpot_data = hotpot_data[:args.max_questions]
            datasets_to_run.append(("hotpotqa", hotpot_data))
        else:
            print(f"WARNING: {hotpot_path} not found. Run 01_prepare_data.py first.")

    if not datasets_to_run:
        print("No datasets found. Run 01_prepare_data.py first.")
        return

    for ds_name, ds_data in datasets_to_run:
        t_start = time.time()
        results = run_baseline_experiment(ds_name, ds_data, encoder, args.top_k)
        total_time = time.time() - t_start

        output_path = os.path.join(results_dir, f"{ds_name}_per_query_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n>>> Saved per-query results: {output_path}")

        for k in args.top_k:
            print_summary(results, ds_name, k=k)

        print(f"\nTotal time for {ds_name}: {total_time:.1f}s")

    print(f"\n>>> ALL BASELINES v2 COMPLETE")
    print(f"  Next step: python -m src.router.prepare_env_data")


if __name__ == "__main__":
    main()
