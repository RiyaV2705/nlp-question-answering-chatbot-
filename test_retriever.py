"""
Unit tests for Information Retrieval (TF-IDF) module.
"""

import pytest
import pandas as pd
from src.retriever import TFIDFRetriever
from src.data_loader import load_corpus


@pytest.fixture
def retriever():
    df = load_corpus()
    r = TFIDFRetriever()
    r.index_corpus(df)
    return r


def test_retriever_indexing(retriever):
    assert retriever.is_indexed
    assert retriever.tfidf_matrix is not None
    assert retriever.tfidf_matrix.shape[0] == len(retriever.corpus_df)


def test_retriever_top_k(retriever):
    results = retriever.retrieve("Alan Turing Bletchley Park Enigma", top_k=3)
    assert len(results) == 3
    # Top result should be Alan Turing passage (P01)
    assert results[0]["id"] == "P01"
    assert results[0]["rank"] == 1
    assert results[0]["retrieval_score"] > 0.0


def test_retriever_mars_query(retriever):
    results = retriever.retrieve("moons of Mars Phobos Deimos", top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "P06"


def test_retriever_empty_query(retriever):
    results = retriever.retrieve("", top_k=3)
    assert results == []


def test_retriever_ir_metrics(retriever):
    eval_sample = [
        {
            "id": "T1",
            "question": "Who cracked the Enigma machine?",
            "ground_truth_passage_id": "P01",
            "is_impossible": False
        },
        {
            "id": "T2",
            "question": "What is the elevation of Mount Everest?",
            "ground_truth_passage_id": "P09",
            "is_impossible": False
        }
    ]
    metrics = retriever.evaluate_retrieval(eval_sample, top_k=3)
    assert "precision@1" in metrics
    assert "precision@3" in metrics
    assert "recall@3" in metrics
    assert "mrr" in metrics
    assert metrics["precision@1"] >= 0.5
    assert metrics["recall@3"] >= 0.5
