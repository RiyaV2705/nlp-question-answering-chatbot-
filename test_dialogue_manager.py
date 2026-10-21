"""
Unit tests for Dialogue Manager and Frame Tracking.
"""

import pytest
from src.dialogue_manager import DialogueManager, IntentClassifier, DialogueFrame
from src.knowledge_base import KnowledgeBase
from src.retriever import TFIDFRetriever
from src.data_loader import load_corpus


@pytest.fixture
def dialogue_system():
    kb = KnowledgeBase()
    retriever = TFIDFRetriever()
    retriever.index_corpus(load_corpus())
    dm = DialogueManager(kb=kb, retriever=retriever, qa_engine=None)
    return dm


def test_intent_classification_greetings():
    frame = DialogueFrame()
    intent, _ = IntentClassifier.classify("Hello there!", frame)
    assert intent == "GREETING"

    intent, _ = IntentClassifier.classify("good morning", frame)
    assert intent == "GREETING"


def test_intent_classification_gratitude_and_goodbye():
    frame = DialogueFrame()
    intent, _ = IntentClassifier.classify("Thank you very much!", frame)
    assert intent == "GRATITUDE"

    intent, _ = IntentClassifier.classify("Goodbye, see you later", frame)
    assert intent == "GOODBYE"


def test_intent_classification_help():
    frame = DialogueFrame()
    intent, _ = IntentClassifier.classify("What can you do?", frame)
    assert intent == "HELP"


def test_kb_intent_and_execution(dialogue_system):
    res = dialogue_system.process_turn("What is the capital of Japan?")
    assert res["response_type"] == "KNOWLEDGE_BASE"
    assert "Tokyo" in res["answer"]
    assert dialogue_system.frame.current_entity == "Japan"
    assert dialogue_system.frame.active_state == "KB_ANSWERED"


def test_dialogue_frame_followup(dialogue_system):
    # Turn 1: Ask about France
    res1 = dialogue_system.process_turn("What is the capital of France?")
    assert "Paris" in res1["answer"]
    assert dialogue_system.frame.current_entity == "France"

    # Turn 2: Follow-up question with pronoun "its currency"
    # Even without pronoun, if user asks "What is its currency?", follow-up resolution triggers
    resolved = dialogue_system.resolve_followup_query("What is its currency?", "France")
    assert "France" in resolved


def test_clear_chat(dialogue_system):
    dialogue_system.process_turn("What is the capital of France?")
    assert dialogue_system.frame.turn_count == 1
    assert dialogue_system.frame.current_entity == "France"

    res = dialogue_system.process_turn("clear chat")
    assert res["response_type"] == "CHITCHAT"
    assert dialogue_system.frame.turn_count == 0
    assert dialogue_system.frame.current_entity is None
