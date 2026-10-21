"""
Unit tests for Data Loader module.
"""

import os
import pytest
import pandas as pd
from src.data_loader import preprocess_text, tokenize_words, load_corpus, load_eval_dataset, get_passage_by_id


def test_preprocess_text():
    raw = "  Hello, WORLD! This is a test...   "
    clean = preprocess_text(raw, lowercase=True, remove_punctuation=False)
    assert clean == "hello, world! this is a test..."

    no_punct = preprocess_text(raw, lowercase=True, remove_punctuation=True)
    assert no_punct == "hello world this is a test"


def test_tokenize_words():
    raw = "Natural Language Processing (NLP) is exciting!"
    tokens = tokenize_words(raw)
    assert "natural" in tokens
    assert "language" in tokens
    assert "nlp" in tokens
    assert len(tokens) == 6


def test_tokenize_words_nltk():
    from src.data_loader import tokenize_words_nltk
    raw = "Artificial Intelligence transforms Question Answering."
    tokens = tokenize_words_nltk(raw)
    assert len(tokens) >= 4
    assert any("intelligence" in t.lower() for t in tokens)


def test_split_sentences_nltk():
    from src.data_loader import split_sentences_nltk
    raw = "Alan Turing was born in London. He cracked the Enigma code!"
    sents = split_sentences_nltk(raw)
    assert len(sents) == 2
    assert "Alan Turing" in sents[0]


def test_load_corpus():
    df = load_corpus()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "id" in df.columns
    assert "passage" in df.columns
    assert "title" in df.columns
    assert len(df) >= 3


def test_load_eval_dataset():
    eval_data = load_eval_dataset()
    assert isinstance(eval_data, list)
    assert len(eval_data) > 0
    first = eval_data[0]
    assert "question" in first
    assert "ground_truth_answers" in first


def test_get_passage_by_id():
    df = load_corpus()
    first_id = df.iloc[0]["id"]
    p = get_passage_by_id(df, first_id)
    assert p is not None
    assert p["id"] == first_id

    missing = get_passage_by_id(df, "NON_EXISTENT_999")
    assert missing is None
