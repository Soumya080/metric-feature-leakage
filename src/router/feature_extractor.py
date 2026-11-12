import csv
import math
import numpy as np
from typing import Dict, Any, List, Optional

# Lazy imports — spacy and sentence_transformers are only needed when
# QueryFeatureExtractor is instantiated, not at module import time.
spacy = None
SentenceTransformer = None

def _ensure_spacy():
    global spacy
    if spacy is None:
        import spacy as _spacy
        spacy = _spacy
    return spacy

def _ensure_sentence_transformers():
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _ST
        SentenceTransformer = _ST
    return SentenceTransformer

# Static standard NLP reference corpora
STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", 
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "did", 
    "do", "does", "doing", "don't", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its", 
    "itself", "just", "me", "more", "most", "myself", "no", "nor", "not", "now", 
    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "out", "over", 
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", 
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", 
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", 
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", 
    "with", "won't", "would", "you", "your", "yours", "yourself", "yourselves"
}

TARGET_ENTITY_LABELS = ["PERSON", "ORG", "GPE", "LOC", "DATE", "CARDINAL"]

class QueryFeatureExtractor:
    """
    Extracts a completely flat feature vector from text and retrieval metrics.
    Outputs data structures ready for direct tabular storage (CSV/DataFrames).
    """

    def __init__(self, include_embeddings: bool = False):
        self.stopwords = STANDARD_STOPWORDS
        self.target_entities = TARGET_ENTITY_LABELS
        self.question_words = ["who", "what", "when", "where", "why", "how"]
        self.include_embeddings = include_embeddings
        
        # Load pipelines once during initialization to optimize throughput
        _ensure_spacy()
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        
        if self.include_embeddings:
            _ensure_sentence_transformers()
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_query_features(
        self, 
        query_id: str, 
        query: str, 
        avg_bm25: float = 0.0, 
        avg_dense: float = 0.0, 
        bm25_variance: float = 0.0,
        oracle_best_retriever: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts structural text metrics and builds a flat 1D data dictionary row.
        """
        # 1. Text Parsing Passes
        spacy_doc = self.nlp(query)
        words = query.strip().split() if query.strip() else []
        query_length = len(words)
        
        # 2. Lexical Logic Calculations
        total_word_len = sum(len(word) for word in words)
        avg_word_length = total_word_len / query_length if query_length > 0 else 0.0
        
        stopword_count = sum(1 for word in words if word.lower() in self.stopwords)
        stopword_ratio = stopword_count / query_length if query_length > 0 else 0.0
        
        unique_word_count = len(set(word.lower() for word in words))
        unique_word_ratio = unique_word_count / query_length if query_length > 0 else 0.0
        has_quotes = float('"' in query or "'" in query)

        # 3. Entity Presence Mapping
        num_entities = len(spacy_doc.ents)
        entity_matrix = {f"entity_present_{label.lower()}": 0.0 for label in self.target_entities}
        for ent in spacy_doc.ents:
            if ent.label_ in self.target_entities:
                entity_matrix[f"entity_present_{ent.label_.lower()}"] = 1.0

        # 4. Structural Structural One-Hot Generation
        cleaned_query = query.lower().strip()
        question_type_features = {f"q_type_{word}": 0.0 for word in self.question_words}
        question_type_features["q_type_other"] = 1.0
        
        for word in self.question_words:
            if cleaned_query.startswith(word):
                question_type_features[f"q_type_{word}"] = 1.0
                question_type_features["q_type_other"] = 0.0
                break 

        contains_numbers = 1.0 if any(char.isdigit() for char in cleaned_query) else 0.0
        contains_proper_nouns = 1.0 if any(token.pos_ == "PROPN" for token in spacy_doc) else 0.0

        # 5. Core Retrieval Calculation Metrics
        score_gap = avg_dense - avg_bm25

        # 6. Base Record Construction
        flat_record = {
            "query_id": query_id,
            "query_length": int(query_length),
            "log_query_length": float(math.log1p(query_length)),
            "avg_word_length": float(avg_word_length),
            "stopword_ratio": float(stopword_ratio),
            "unique_word_ratio": float(unique_word_ratio),
            "has_quotes": float(has_quotes),
            "num_entities": int(num_entities),
            **entity_matrix,
            **question_type_features,
            "contains_numbers": contains_numbers,
            "contains_proper_nouns": contains_proper_nouns,
            "avg_bm25": float(avg_bm25),
            "avg_dense": float(avg_dense), 
            "score_gap": float(score_gap),
            "bm25_score_variance": float(bm25_variance)
        }

        # 7. Semantic Ablation Integration (Ablation testing flag)
        if self.include_embeddings:
            raw_emb = self.embedding_model.encode(query, convert_to_numpy=True)
            # Store array directly as string representations to safeguard standard CSV parser boundaries
            flat_record["query_embedding"] = str(raw_emb.tolist())

        # Ground-truth categorization target
        if oracle_best_retriever is not None:
            flat_record["oracle_best_retriever"] = oracle_best_retriever

        return flat_record

    @staticmethod
    def get_feature_order() -> List[str]:
        """Returns the canonical ordered list of numeric feature names.
        
        This defines the exact feature vector layout fed to the policy network.
        Must remain fixed across training and evaluation for reproducibility.
        """
        return [
            # Lexical features (6)
            "query_length", "log_query_length", "avg_word_length",
            "stopword_ratio", "unique_word_ratio", "has_quotes",
            # Entity features (7)
            "num_entities",
            "entity_present_person", "entity_present_org", "entity_present_gpe",
            "entity_present_loc", "entity_present_date", "entity_present_cardinal",
            # Structural features (9)
            "q_type_who", "q_type_what", "q_type_when", "q_type_where",
            "q_type_why", "q_type_how", "q_type_other",
            "contains_numbers", "contains_proper_nouns",
            # Retrieval score features (4)
            "avg_bm25", "avg_dense", "score_gap", "bm25_score_variance",
            # Cross-retriever agreement features (3) — how much BM25's and Dense's result
            # lists overlap. Absolute confidence signals (above) say "is this retriever
            # sure"; these say "do the retrievers disagree", which is where per-query
            # routing has anything to gain. Label-free: computed by comparing the two
            # ranked lists to each other, never to the gold answer.
            "cross_ret_overlap_5", "cross_ret_overlap_10", "cross_ret_top1_agree",
        ]

    @staticmethod
    def to_feature_vector(flat_record: Dict[str, Any]) -> List[float]:
        """Convert an extracted feature dict into an ordered numeric list.
        
        This is the bridge between feature_extractor output and the 
        policy network input. The ordering is fixed by get_feature_order()
        to ensure reproducibility across train/eval/ablation runs.
        
        Args:
            flat_record: Output of extract_query_features().
            
        Returns:
            A 1D list of floats with length = len(get_feature_order()) = 29.
        """
        feature_order = QueryFeatureExtractor.get_feature_order()
        return [float(flat_record.get(key, 0.0)) for key in feature_order]


# ==========================================
# SYSTEM EXECUTION RUNNER (DATASET GENERATOR)
# ==========================================
if __name__ == "__main__":
    # Initialize the automated tool runner
    # Toggle 'include_embeddings=True' if executing a feature ablation study chapter
    extractor = QueryFeatureExtractor(include_embeddings=False)

    # Simulated batch corpus input logs from experimental pipelines
    mock_raw_dataset = [
        {
            "query_id": "Q-001",
            "query": "Who is the primary actor in the film Inception?",
            "avg_bm25": 14.82, "avg_dense": 12.11, "bm25_variance": 4.12,
            "oracle_best_retriever": "BM25"
        },
        {
            "query_id": "Q-002",
            "query": "How do deep neural network architectures handle vanishing gradient adjustments over long sequences?",
            "avg_bm25": 4.11, "avg_dense": 18.94, "bm25_variance": 0.24,
            "oracle_best_retriever": "Dense"
        },
        {
            "query_id": "Q-003",
            "query": "What are the latest 'financial constraints' regarding trade policies in the US and Germany written on 2026-05-12?",
            "avg_bm25": 9.54, "avg_dense": 10.15, "bm25_variance": 1.85,
            "oracle_best_retriever": "Dense"
        }
    ]

    processed_rows: List[Dict[str, Any]] = []

    print("Beginning empirical feature extraction pass...")
    for entry in mock_raw_dataset:
        features_flat = extractor.extract_query_features(
            query_id=entry["query_id"],
            query=entry["query"],
            avg_bm25=entry["avg_bm25"],
            avg_dense=entry["avg_dense"],
            bm25_variance=entry["bm25_variance"],
            oracle_best_retriever=entry["oracle_best_retriever"]
        )
        processed_rows.append(features_flat)

    # Dynamic target export configuration setup
    output_csv_filename = "router_dataset.csv"

    if processed_rows:
        # Pull table header configurations safely from parsed entry keys
        csv_headers = list(processed_rows[0].keys())

        with open(output_csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(processed_rows)

        print(f"Extraction sequence complete. Target dataset saved to: '{output_csv_filename}'")
