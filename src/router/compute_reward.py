

# Module-level cost penalties (imported by train_reinforce.py, train_grpo.py)
COST_PENALTIES = {
    0: 0.00,   # BM25
    1: 0.02,   # Dense
    2: 0.03,   # Hybrid
    'BM25': 0.00,
    'Dense': 0.02,
    'Hybrid': 0.03,
    'No-retrieval': 0.00,
}


def compute_reward(retrieved_docs, gold_answer, action, config=None):
    """Compute reward as retrieval_quality - cost_penalty.

    retrieved_docs: list of passages (strings) ordered by rank (0 = top-1)
    gold_answer: string to search for in retrieved passages
    action: one of 'BM25', 'Dense', 'Hybrid', 'No-retrieval'
    config: unused optional dict
    """

    # determine retrieval quality by Hit@1/3/5
    retrieval_quality = 0.0
    try:
        # find index of first passage that contains the gold answer as substring
        idx = next(i for i, doc in enumerate(retrieved_docs) if gold_answer in doc)
    except StopIteration:
        idx = None

    if idx == 0:
        retrieval_quality = 1.0
    elif idx is not None and 1 <= idx <= 2:  # top-3 but not top-1
        retrieval_quality = 0.5
    elif idx is not None and 3 <= idx <= 4:  # top-5 but not top-3
        retrieval_quality = 0.2
    else:
        retrieval_quality = 0.0

    # cost penalties for different retrieval actions
    action = str(action)
    cost_penalty = COST_PENALTIES.get(action, 0.0)

    return retrieval_quality - cost_penalty


def compute_reward_from_metrics(metrics, action, reward_type="mrr"):
    """Compute reward directly from pre-computed metrics dict.

    metrics: dict with keys like 'hit_at_5', 'mrr', 'f1', 'recall', 'em'
    action: integer action (0=BM25, 1=Dense, 2=Hybrid)
    reward_type: which metric to use as base reward
    """
    base_reward = float(metrics.get(reward_type, metrics.get("mrr", 0.0)))
    cost = COST_PENALTIES.get(action, 0.0)
    return base_reward - cost

        
        
    


        
        
    
    
    

    

