"""
Unit tests for Knowledge Base (SQLite) module.
"""

import os
import pytest
from src.knowledge_base import KnowledgeBase


@pytest.fixture
def temp_kb(tmp_path):
    """Fixture providing a temporary SQLite knowledge base instance."""
    db_file = os.path.join(tmp_path, "test_kb.db")
    kb = KnowledgeBase(db_path=db_file)
    return kb


def test_kb_initialization(temp_kb):
    entities = temp_kb.get_all_entities()
    assert len(entities) > 0
    assert "Alan Turing" in entities
    assert "Python" in entities
    assert "Mount Everest" in entities


def test_kb_lookup_fact(temp_kb):
    facts = temp_kb.lookup_fact("France", "capital")
    assert len(facts) > 0
    assert facts[0]["value"] == "Paris"

    python_creator = temp_kb.lookup_fact("Python", "creator")
    assert len(python_creator) > 0
    assert "Guido van Rossum" in python_creator[0]["value"]


def test_kb_answer_factoid_query(temp_kb):
    res1 = temp_kb.answer_factoid_query("What is the capital of France?")
    assert res1 is not None
    assert res1["answer"] == "Paris"
    assert res1["confidence"] == 1.0

    res2 = temp_kb.answer_factoid_query("Who created Python?")
    assert res2 is not None
    assert "Guido van Rossum" in res2["answer"]

    res3 = temp_kb.answer_factoid_query("What is the height of Mount Everest?")
    assert res3 is not None
    assert "8,848.86 metres" in res3["answer"]


def test_kb_unmatched_query(temp_kb):
    res = temp_kb.answer_factoid_query("What is the quantum state of tachyons?")
    assert res is None


def test_kb_add_fact(temp_kb):
    temp_kb.add_fact("Nikola Tesla", "Person", "invented", "Alternating Current (AC)", "Historical Record")
    facts = temp_kb.lookup_fact("Nikola Tesla", "invented")
    assert len(facts) > 0
    assert "Alternating Current" in facts[0]["value"]


def test_kb_execute_custom_query(temp_kb):
    df = temp_kb.execute_custom_query("SELECT COUNT(*) as cnt FROM facts")
    assert not df.empty
    assert df.iloc[0]["cnt"] > 0
