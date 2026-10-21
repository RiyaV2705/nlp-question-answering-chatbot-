"""
Information Retrieval Module
============================
Ranks and retrieves relevant passages from the text corpus using
TF-IDF vectorization with cosine similarity and optional keyword matching.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import preprocess_text


class BaseRetriever:
    """Abstract base retriever interface."""

    def index_corpus(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        raise NotImplementedError


class TFIDFRetriever(BaseRetriever):
    """
    TF-IDF based passage retriever using Scikit-Learn.
    Computes unigram + bigram TF-IDF representations and cosine similarity.
    """

    def __init__(self, ngram_range: tuple = (1, 2), sublinear_tf: bool = True):
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            stop_words="english",
            lowercase=True
        )
        self.corpus_df: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None
        self.is_indexed = False

    def index_corpus(self, df: pd.DataFrame) -> None:
        """
        Indexes the provided DataFrame corpus.
        
        Args:
            df: DataFrame containing ['id', 'title', 'category', 'passage']
        """
        self.corpus_df = df.copy().reset_index(drop=True)
        # Combine title and passage text for richer representation
        combined_text = (
            self.corpus_df["title"].fillna("") + " " +
            self.corpus_df["passage"].fillna("")
        ).tolist()

        self.tfidf_matrix = self.vectorizer.fit_transform(combined_text)
        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Ranks all indexed passages for a given query and returns top_k results.
        
        Args:
            query: The user query string.
            top_k: Number of passages to retrieve.
            
        Returns:
            List of passage dicts with relevance scores and ranks.
        """
        if not self.is_indexed or self.corpus_df is None or self.tfidf_matrix is None:
            return []

        cleaned_query = preprocess_text(query, lowercase=True, remove_punctuation=False)
        if not cleaned_query:
            return []

        query_vec = self.vectorizer.transform([cleaned_query])
        # Compute cosine similarity
        cosine_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Use pure cosine similarity scores
        scores = cosine_scores

        # Rank indices descending
        k = min(top_k, len(self.corpus_df))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for rank_idx, idx in enumerate(top_indices, start=1):
            row = self.corpus_df.iloc[idx]
            score_val = float(scores[idx])
            results.append({
                "rank": rank_idx,
                "id": str(row["id"]),
                "title": str(row["title"]),
                "category": str(row["category"]),
                "passage": str(row["passage"]),
                "retrieval_score": round(score_val, 4),
                "retriever_type": "TF-IDF (Cosine Sim)"
            })

        return results

    def evaluate_retrieval(self, eval_set: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, float]:
        """
        Computes standard IR metrics across evaluation questions:
        - Precision@1
        - Precision@K
        - Recall@K
        - Mean Reciprocal Rank (MRR)
        """
        if not self.is_indexed:
            return {"precision@1": 0.0, f"precision@{top_k}": 0.0, f"recall@{top_k}": 0.0, "mrr": 0.0}

        p1_list = []
        pk_list = []
        rk_list = []
        rr_list = []

        valid_queries = [q for q in eval_set if not q.get("is_impossible", False) and q.get("ground_truth_passage_id")]
        if not valid_queries:
            return {"precision@1": 0.0, f"precision@{top_k}": 0.0, f"recall@{top_k}": 0.0, "mrr": 0.0}

        for item in valid_queries:
            q_text = item["question"]
            gt_id = str(item["ground_truth_passage_id"]).strip()

            retrieved = self.retrieve(q_text, top_k=top_k)
            retrieved_ids = [r["id"] for r in retrieved]

            # Precision@1: Is the top-1 retrieved passage relevant?
            p1 = 1.0 if retrieved_ids and retrieved_ids[0] == gt_id else 0.0
            p1_list.append(p1)

            # Precision@K: Proportion of retrieved passages that are relevant (for single relevant passage: 1/K or 0)
            hits = 1 if gt_id in retrieved_ids else 0
            pk = hits / max(len(retrieved_ids), 1)
            pk_list.append(pk)

            # Recall@K: Did top-K contain the relevant passage? (1 or 0)
            rk = 1.0 if hits > 0 else 0.0
            rk_list.append(rk)

            # Reciprocal Rank: 1 / (rank of first relevant document)
            rr = 0.0
            for rank_idx, pid in enumerate(retrieved_ids, start=1):
                if pid == gt_id:
                    rr = 1.0 / rank_idx
                    break
            rr_list.append(rr)

        return {
            "precision@1": round(float(np.mean(p1_list)), 4),
            f"precision@{top_k}": round(float(np.mean(pk_list)), 4),
            f"recall@{top_k}": round(float(np.mean(rk_list)), 4),
            "mrr": round(float(np.mean(rr_list)), 4),
            "total_evaluated": len(valid_queries)
        }
