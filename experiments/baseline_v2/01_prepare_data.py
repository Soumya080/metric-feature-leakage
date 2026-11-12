"""
01_prepare_data.py — Baseline v2: Real Wikipedia Passage Retrieval Corpus (FAST version)

Uses pre-built word index for O(1) answer-in-passage lookup instead of O(N) linear scan.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

from datasets import load_dataset
from tqdm import tqdm

# ============================================================
# CONFIG (all overridable via CLI for quick timing-calibration runs —
# see --num_questions / --num_distractors / --wiki_pool_size)
# ============================================================
NUM_NQ_QUESTIONS = 2000
NUM_TRIVIAQA_QUESTIONS = 2000
NUM_HOTPOTQA_QUESTIONS = 2000
NUM_DISTRACTORS = 499
# Matches the passage count actually in cache/wiki_passage_pool.json, which is what every
# committed *_prepared.json was built from. This was 50000 while the cache held 20000, and
# load_wiki_pool() returns a cached pool without checking its size — so the constant silently
# disagreed with the data. Report 20k in the paper, not 50k.
WIKI_POOL_SIZE = 20000
# Negative sampling. "random" is the historical behaviour (and what every result before
# 2026-08-02 was built on, despite the paper/README describing "hard negatives").
# "mixed" is the default: topically-mined negatives plus random filler.
NEGATIVE_STRATEGY = "mixed"
HARD_NEGATIVE_FRACTION = 0.5
# A word appearing in more than this many pool passages is treated as non-topical.
MAX_POSTINGS_FOR_NEGATIVES = 2000

MIN_PASSAGE_LENGTH = 80
MAX_PASSAGE_LENGTH = 1500
RANDOM_SEED = 42
OUTPUT_DIR = "experiments/baseline_v2/data"
CACHE_DIR = "experiments/baseline_v2/cache"


def normalize(text):
    return re.sub(r'\s+', ' ', text.lower().strip())


def build_word_index(passages):
    """Build inverted index: word -> set of passage indices. O(N) build, O(1) lookup."""
    print("  Building passage word index for fast lookup...")
    index = defaultdict(set)
    for i, p in enumerate(passages):
        words = set(normalize(p).split())
        for w in words:
            if len(w) >= 3:
                index[w].add(i)
    print(f"  Index built: {len(index)} unique words across {len(passages)} passages")
    return index


# Words too common to carry topical signal — including them would rank passages by
# stopword count rather than by subject overlap with the question.
# Deliberately function words ONLY. Words like "first", "new", "year", "work" look common
# but carry real topical signal in questions ("first president", "New Zealand", "work of
# art") — excluding them measurably weakens negative mining.
_STOPWORDS = frozenset("""
the and for are but not you all any can had her was our has him his how who did its let
say she too use that with have this from they will would there their what about which
when make like just know take into some could them than then only come over also back
after where much
""".split())


def question_content_words(question):
    """Content words from a question, for topical overlap scoring."""
    return {w for w in normalize(question).split() if len(w) >= 3 and w not in _STOPWORDS}


def find_hard_negatives(question, word_index, exclude, n_wanted):
    """Passages sharing the most content words with the question, excluding gold matches.

    These are *hard* negatives: topically related to the query but not containing the
    answer. Uniformly random passages (the previous behaviour) are trivially separable —
    any passage sharing a query term stands out, so BM25 and Dense both find the gold and
    AGREE, which collapses the per-query retriever disagreement this project is about.

    Returns indices ordered by descending overlap. May return fewer than n_wanted when the
    question has little vocabulary in common with the pool; callers pad with random picks.
    """
    words = question_content_words(question)
    if not words:
        return []

    overlap = Counter()
    for w in words:
        # Very common words match a huge share of the pool and add no topical signal,
        # while costing the most to iterate — skip them.
        postings = word_index.get(w)
        if not postings or len(postings) > MAX_POSTINGS_FOR_NEGATIVES:
            continue
        for idx in postings:
            if idx not in exclude:
                overlap[idx] += 1

    if not overlap:
        return []
    return [idx for idx, _ in overlap.most_common(n_wanted)]


def sample_distractors(question, wiki_pool, word_index, exclude, n_wanted, strategy, rng):
    """Pick n_wanted distractor indices under the given negative-sampling strategy.

    Returns (indices, n_hard) so the caller can record how many were topically mined —
    the paper must describe what was actually done, per query, not an intended design.
    """
    if strategy == "random":
        available = [i for i in range(len(wiki_pool)) if i not in exclude]
        return rng.sample(available, min(n_wanted, len(available))), 0

    n_hard_target = n_wanted if strategy == "hard" else int(n_wanted * HARD_NEGATIVE_FRACTION)
    hard = find_hard_negatives(question, word_index, exclude, n_hard_target)

    chosen = list(hard)
    n_hard = len(chosen)

    # Pad with random passages: partly to reach n_wanted when mining came up short, and
    # for "mixed" by design — an all-hard corpus is unrepresentative of a real index,
    # where most passages are simply unrelated.
    if len(chosen) < n_wanted:
        taken = exclude | set(chosen)
        available = [i for i in range(len(wiki_pool)) if i not in taken]
        chosen.extend(rng.sample(available, min(n_wanted - len(chosen), len(available))))

    rng.shuffle(chosen)
    return chosen, n_hard


def find_gold_fast(answer, passages, word_index):
    """Find passages containing answer using inverted index. Much faster than linear scan."""
    answer_norm = normalize(answer)
    answer_words = [w for w in answer_norm.split() if len(w) >= 3]
    
    if not answer_words:
        return []
    
    # Get candidate passages that contain ALL answer words
    candidates = None
    for w in answer_words:
        if w in word_index:
            if candidates is None:
                candidates = word_index[w].copy()
            else:
                candidates &= word_index[w]
        else:
            return []  # Word not in any passage
    
    if not candidates:
        return []
    
    # Verify exact substring match on candidates only
    matches = []
    for idx in candidates:
        if answer_norm in normalize(passages[idx]):
            matches.append(idx)
            if len(matches) >= 3:
                break
    return matches


def count_answer_bearing(passages, answers):
    """How many of these 'negatives' actually contain an answer string.

    find_gold_fast stops after 3 matches, and only those 3 are excluded from distractor
    sampling — out of a pool of tens of thousands. Distractors are then mined FOR topical
    overlap with the question, which raises the odds they contain the answer. So some of
    the "499 hard negatives" are positives, and every metric scores any answer-bearing
    passage as relevant (first_answer_bearing_rank). A query's reward is therefore partly
    determined by how many un-excluded answer-bearing distractors it happened to draw.

    The mechanism was verified long before the magnitude was; this measures the magnitude.
    Uses the same normalised-substring test as find_gold_fast, because that is the
    criterion by which the builder decided what to exclude — a second definition of
    "contains the answer" would measure something other than what slipped through.

    Expected to RISE with pool size: a larger pool yields harder, more topical negatives.
    That makes it a number to watch across the 20K and 100K runs, not just report once.
    """
    norm_answers = [normalize(a) for a in answers if a and normalize(a)]
    if not norm_answers:
        return 0
    n = 0
    for p in passages:
        pn = normalize(p)
        if any(a in pn for a in norm_answers):
            n += 1
    return n


def construct_gold(question, answer):
    """Natural-sounding constructed passage when no real one found."""
    templates = [
        f"{answer} is widely recognized in this context. According to various sources, "
        f"the topic relates to {answer}, which has been documented extensively in academic "
        f"and popular literature. Multiple references confirm this established fact.",
        f"Historical records and encyclopedic references indicate that {answer} plays a "
        f"significant role. This has been confirmed through multiple independent sources "
        f"and scholarly publications across different time periods.",
        f"Research and documentation confirm that {answer} is the established answer to "
        f"this topic. This fact has been verified across multiple reliable sources including "
        f"academic databases and reference materials.",
    ]
    return random.choice(templates)


def load_wiki_pool(cache_path):
    """Load cached wiki pool or download fresh."""
    if os.path.exists(cache_path):
        print(f"  Loading cached passage pool...")
        with open(cache_path, "r", encoding="utf-8") as f:
            passages = json.load(f)
        print(f"  Loaded {len(passages)} cached passages")
        if len(passages) != WIKI_POOL_SIZE:
            print(f"  WARNING: cached pool has {len(passages)} passages but WIKI_POOL_SIZE="
                  f"{WIKI_POOL_SIZE}. Hard negatives will be mined from the cached pool, so "
                  f"{len(passages)} is the number to report. Delete the cache to rebuild at "
                  f"{WIKI_POOL_SIZE}.")
        return passages

    print("\nLoading wiki_snippets to gather distractor passages...")
    t0 = time.time()
    wiki = load_dataset("community-datasets/wiki_snippets", "wiki40b_en_100_0", split="train", trust_remote_code=True)
    
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(wiki)), min(WIKI_POOL_SIZE * 2, len(wiki)))
    
    passages = []
    for idx in tqdm(indices, desc="Extracting"):
        item = wiki[idx]
        text = ""
        for key in ["snippet_text", "passage_text", "section_content", "text"]:
            if key in item and isinstance(item[key], str) and len(item[key]) > 50:
                text = item[key].strip()
                break
        if not text:
            for key in item:
                if isinstance(item[key], str) and len(item[key]) > 50:
                    text = item[key].strip()
                    break
        if MIN_PASSAGE_LENGTH <= len(text) <= MAX_PASSAGE_LENGTH:
            passages.append(text)
        if len(passages) >= WIKI_POOL_SIZE:
            break

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False)
    print(f"  Saved {len(passages)} passages in {time.time()-t0:.1f}s")
    return passages


def prepare_dataset(ds_name, dataset, wiki_pool, word_index, num_questions, seed):
    """Prepare one dataset with fast gold passage lookup."""
    print(f"\n{'='*60}")
    print(f"PREPARING {ds_name} — v2 Real Passages")
    print(f"{'='*60}")

    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_questions, len(dataset)))
    
    prepared = []
    stats = {"real_gold": 0, "constructed": 0, "skipped": 0,
             "hard_negatives": 0, "total_negatives": 0, "duplicates": 0,
             "answer_bearing_distractors": 0,
             "queries_with_contaminated_negatives": 0}
    # TriviaQA's validation split repeats ~5% of its questions verbatim. A duplicate that
    # lands on opposite sides of the train/test boundary is a genuine leak: the model has
    # already seen the test item. Drop repeats, keeping the first occurrence.
    seen_questions = set()

    for idx in tqdm(indices, desc=f"Preparing {ds_name}"):
        item = dataset[idx]
        question = item["question"]

        q_key = normalize(question)
        if q_key in seen_questions:
            stats["duplicates"] += 1
            continue
        seen_questions.add(q_key)

        # Extract answers
        if ds_name == "nq":
            answers = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
        elif ds_name == "hotpotqa":
            answers = [item["answer"]] if isinstance(item["answer"], str) else [str(item["answer"])]
        else:  # triviaqa
            answer_data = item.get("answer", {})
            if isinstance(answer_data, dict):
                aliases = answer_data.get("aliases", [])
                value = answer_data.get("value", "")
                answers = list(set([value] + aliases)) if value else aliases
            elif isinstance(answer_data, str):
                answers = [answer_data]
            else:
                answers = [str(answer_data)]
        
        answers = [a for a in answers if a and len(a.strip()) >= 2]
        if not answers:
            stats["skipped"] += 1
            continue

        # Fast gold passage lookup
        gold_idx = None
        matched_answer = None
        for ans in answers:
            matches = find_gold_fast(ans, wiki_pool, word_index)
            if matches:
                gold_idx = matches[0]
                matched_answer = ans
                break

        if gold_idx is not None:
            gold_passage = wiki_pool[gold_idx]
            gold_type = "real_wikipedia"
            stats["real_gold"] += 1
            exclude = set(matches)
        else:
            gold_passage = construct_gold(question, answers[0])
            gold_type = "constructed"
            stats["constructed"] += 1
            exclude = set()

        # Sample distractors (real Wikipedia passages)
        distractor_indices, n_hard = sample_distractors(
            question, wiki_pool, word_index, exclude, NUM_DISTRACTORS,
            NEGATIVE_STRATEGY, random,
        )
        distractors = [wiki_pool[i] for i in distractor_indices]
        stats["hard_negatives"] += n_hard
        stats["total_negatives"] += len(distractor_indices)

        # Census of false negatives — see count_answer_bearing for why these exist.
        n_answer_bearing = count_answer_bearing(distractors, answers)
        stats["answer_bearing_distractors"] += n_answer_bearing
        if n_answer_bearing:
            stats["queries_with_contaminated_negatives"] += 1

        # Insert gold at random position
        corpus = distractors.copy()
        gold_pos = random.randint(0, len(corpus))
        corpus.insert(gold_pos, gold_passage)

        prepared.append({
            "question": question,
            "answers": answers,
            "gold_passage": gold_passage,
            "gold_type": gold_type,
            "gold_position": gold_pos,
            "corpus": corpus,
            "corpus_size": len(corpus),
            "negative_strategy": NEGATIVE_STRATEGY,
            "n_hard_negatives": n_hard,
            # Per-query so results can be checked against contamination, not just
            # aggregated: if reward correlates with this, the metric is partly measuring
            # how many extra answer-bearing passages a query happened to draw.
            "n_answer_bearing_distractors": n_answer_bearing,
        })

    total = len(prepared)
    print(f"\n  Prepared: {total} questions")
    print(f"  Real Wikipedia gold: {stats['real_gold']} ({stats['real_gold']/max(1,total)*100:.1f}%)")
    print(f"  Constructed gold:    {stats['constructed']} ({stats['constructed']/max(1,total)*100:.1f}%)")
    _tot_neg = max(1, stats["total_negatives"])
    print(f"  Duplicate questions dropped: {stats['duplicates']}")
    print(f"  Negative strategy:   {NEGATIVE_STRATEGY} - "
          f"{stats['hard_negatives']}/{stats['total_negatives']} "
          f"({stats['hard_negatives']/_tot_neg*100:.1f}%) topically mined")
    print(f"  Skipped:             {stats['skipped']}")
    # False-negative census. find_gold_fast excludes at most 3 answer-bearing passages, so
    # anything above zero here is a "negative" the metrics will score as relevant.
    _ab = stats["answer_bearing_distractors"]
    print(f"  Answer-bearing distractors: {_ab}/{stats['total_negatives']} "
          f"({_ab/_tot_neg*100:.2f}%), affecting "
          f"{stats['queries_with_contaminated_negatives']}/{total} queries "
          f"({stats['queries_with_contaminated_negatives']/max(1,total)*100:.1f}%) "
          f"- mean {_ab/max(1,total):.2f} per query")
    return prepared, stats


def main():
    global NUM_DISTRACTORS, WIKI_POOL_SIZE, NEGATIVE_STRATEGY, HARD_NEGATIVE_FRACTION

    parser = argparse.ArgumentParser(description="Prepare NQ/TriviaQA/HotpotQA + 500-passage corpora.")
    parser.add_argument("--num_questions", type=int, default=None,
                        help=f"Questions per dataset (default {NUM_NQ_QUESTIONS}). Use a small value (e.g. 20) for a quick timing-calibration run.")
    parser.add_argument("--num_distractors", type=int, default=None,
                        help=f"Hard negatives per query (default {NUM_DISTRACTORS}). Corpus size = num_distractors + 1.")
    parser.add_argument("--wiki_pool_size", type=int, default=None,
                        help=f"Wikipedia passages to sample into the shared pool (default "
                             f"{WIKI_POOL_SIZE}). Only takes effect when the cache is absent — "
                             f"load_wiki_pool() returns an existing cache as-is. Delete "
                             f"experiments/baseline_v2/cache/wiki_passage_pool.json to rebuild. "
                             f"A larger pool raises the share of queries that get a REAL "
                             f"Wikipedia gold passage instead of a constructed template.")
    parser.add_argument("--negative_strategy", type=str, default=NEGATIVE_STRATEGY,
                        choices=["random", "hard", "mixed"],
                        help=f"How to pick the {NUM_DISTRACTORS} distractors (default "
                             f"{NEGATIVE_STRATEGY}). 'random' = uniform sampling, the "
                             f"pre-2026-08-02 behaviour that the paper wrongly described as "
                             f"hard negatives. 'hard' = all topically mined. 'mixed' = "
                             f"topically mined + random filler.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help=f"Where to write *_prepared.json (default {OUTPUT_DIR}). ALWAYS "
                             f"pass this together with a small --num_questions: the real "
                             f"files are ~577MB each and a smoke-test run would otherwise "
                             f"silently overwrite them.")
    parser.add_argument("--hard_negative_fraction", type=float, default=HARD_NEGATIVE_FRACTION,
                        help=f"Share of distractors to mine topically under 'mixed' "
                             f"(default {HARD_NEGATIVE_FRACTION}). Raising this makes "
                             f"retrieval harder; too high and every retriever fails "
                             f"equally, which collapses routing headroom. Pilot it.")
    args = parser.parse_args()

    if args.wiki_pool_size is not None:
        WIKI_POOL_SIZE = args.wiki_pool_size
    NEGATIVE_STRATEGY = args.negative_strategy
    HARD_NEGATIVE_FRACTION = args.hard_negative_fraction

    out_dir = args.output_dir or OUTPUT_DIR
    if args.num_questions and not args.output_dir:
        print(f"WARNING: --num_questions={args.num_questions} without --output_dir will "
              f"OVERWRITE the real datasets in {OUTPUT_DIR} with a truncated version.")

    num_nq = args.num_questions or NUM_NQ_QUESTIONS
    num_tqa = args.num_questions or NUM_TRIVIAQA_QUESTIONS
    num_hotpot = args.num_questions or NUM_HOTPOTQA_QUESTIONS
    if args.num_distractors is not None:
        NUM_DISTRACTORS = args.num_distractors

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("="*60)
    print("BASELINE v2 DATA PREPARATION — Real Wikipedia Passages (FAST)")
    print(f"Questions/dataset: {num_nq} | Corpus size: {NUM_DISTRACTORS + 1}")
    print("="*60)

    # Load pool + build index
    wiki_pool = load_wiki_pool(os.path.join(CACHE_DIR, "wiki_passage_pool.json"))
    word_index = build_word_index(wiki_pool)

    # Load QA datasets
    print("\nLoading NQ-Open...")
    nq_ds = load_dataset("google-research-datasets/nq_open", split="validation")
    print(f"  {len(nq_ds)} questions")

    print("Loading TriviaQA...")
    tqa_ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    print(f"  {len(tqa_ds)} questions")

    print("Loading HotpotQA...")
    hotpot_ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation")
    print(f"  {len(hotpot_ds)} questions")

    # Prepare
    nq_data, nq_stats = prepare_dataset("nq", nq_ds, wiki_pool, word_index, num_nq, RANDOM_SEED)
    tqa_data, tqa_stats = prepare_dataset("triviaqa", tqa_ds, wiki_pool, word_index, num_tqa, RANDOM_SEED + 1)
    hotpot_data, hotpot_stats = prepare_dataset("hotpotqa", hotpot_ds, wiki_pool, word_index, num_hotpot, RANDOM_SEED + 2)

    # Save
    nq_path = os.path.join(out_dir, "nq_prepared.json")
    tqa_path = os.path.join(out_dir, "triviaqa_prepared.json")

    with open(nq_path, "w", encoding="utf-8") as f:
        json.dump(nq_data, f, indent=2, ensure_ascii=False)
    print(f"\n>>> Saved: {nq_path} ({len(nq_data)} questions)")

    with open(tqa_path, "w", encoding="utf-8") as f:
        json.dump(tqa_data, f, indent=2, ensure_ascii=False)
    print(f">>> Saved: {tqa_path} ({len(tqa_data)} questions)")

    hotpot_path = os.path.join(out_dir, "hotpotqa_prepared.json")
    with open(hotpot_path, "w", encoding="utf-8") as f:
        json.dump(hotpot_data, f, indent=2, ensure_ascii=False)
    print(f">>> Saved: {hotpot_path} ({len(hotpot_data)} questions)")

    # Corpus census. Persisted rather than only printed, because these are numbers the
    # paper has to cite -- the share of "hard negatives" that actually contain the answer,
    # and the share that were topically mined rather than random filler. Both contradict
    # what the draft currently says about the corpus, and a number that exists only in a
    # console scrollback cannot be cited or re-checked.
    census = {}
    for name, data, st in (("nq", nq_data, nq_stats), ("triviaqa", tqa_data, tqa_stats),
                           ("hotpotqa", hotpot_data, hotpot_stats)):
        n = max(1, len(data))
        neg = max(1, st["total_negatives"])
        census[name] = {
            "n_queries": len(data),
            "real_gold": st["real_gold"], "constructed_gold": st["constructed"],
            "constructed_share": st["constructed"] / n,
            "total_negatives": st["total_negatives"],
            "hard_negatives": st["hard_negatives"],
            "hard_negative_share": st["hard_negatives"] / neg,
            "answer_bearing_distractors": st["answer_bearing_distractors"],
            "answer_bearing_share": st["answer_bearing_distractors"] / neg,
            "mean_answer_bearing_per_query": st["answer_bearing_distractors"] / n,
            "queries_with_contaminated_negatives": st["queries_with_contaminated_negatives"],
            "contaminated_query_share": st["queries_with_contaminated_negatives"] / n,
        }
    census["_config"] = {
        "wiki_pool_size": len(wiki_pool), "num_distractors": NUM_DISTRACTORS,
        "negative_strategy": NEGATIVE_STRATEGY,
        "hard_negative_fraction": HARD_NEGATIVE_FRACTION,
        "answer_bearing_test": "normalised substring, matching find_gold_fast's exclusion "
                               "criterion -- measures exactly what slipped past its "
                               "3-match cap",
    }
    # results/, not out_dir. .gitattributes routes experiments/baseline_v2/data/*.json to
    # Git LFS, because the three *_prepared.json files are ~577MB each and deliberately do
    # not travel with the repo. A 2KB census has no business in a quota-limited store, and
    # it is a report -- it belongs beside the other reports, where it is committed normally
    # and can be diffed between runs.
    census_dir = "results" if out_dir == OUTPUT_DIR else out_dir
    os.makedirs(census_dir, exist_ok=True)
    census_path = os.path.join(census_dir, "corpus_census.json")
    with open(census_path, "w", encoding="utf-8") as f:
        json.dump(census, f, indent=2)
    print(f">>> Saved: {census_path}")

    # Sample
    if nq_data:
        s = nq_data[0]
        print(f"\nSample — Q: {s['question']}")
        print(f"  Answers: {s['answers']}")
        print(f"  Gold type: {s['gold_type']}")
        print(f"  Gold (200 chars): {s['gold_passage'][:200]}...")

    print(f"\n>>> DONE. Next: python experiments/baseline_v2/02_run_baselines.py")


if __name__ == "__main__":
    main()
