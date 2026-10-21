"""
Unit tests for Extractive QA Engine and confidence thresholding.
"""

import pytest
from src.qa_engine import ExtractiveQAEngine


def test_qa_engine_initialization():
    engine = ExtractiveQAEngine(model_name="distilbert-base-cased-distilled-squad")
    assert engine.model_name == "distilbert-base-cased-distilled-squad"


def test_qa_engine_empty_inputs():
    engine = ExtractiveQAEngine()
    # Test with empty question
    res = engine.answer_single_context(question="", context="Some context text here.")
    assert res["confidence"] == 0.0
    assert not res["is_confident"]

    # Test with empty context
    res2 = engine.answer_single_context(question="Who is this?", context="")
    assert res2["confidence"] == 0.0
    assert not res2["is_confident"]


def test_qa_engine_multi_passage_threshold():
    engine = ExtractiveQAEngine()
    passages = [
        {"id": "P01", "title": "Alan Turing", "passage": "Alan Turing was an English mathematician.", "retrieval_score": 0.8},
        {"id": "P02", "title": "Transformer", "passage": "The Transformer architecture was introduced in 2017.", "retrieval_score": 0.3}
    ]

    # Test with empty passages list
    res_empty = engine.answer_from_passages("Who was Turing?", passages=[], confidence_threshold=0.15)
    assert res_empty["status"] == "NO_PASSAGES"
    assert res_empty["confidence"] == 0.0
