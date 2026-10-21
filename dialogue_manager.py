"""
Dialogue Manager Module
=======================
Handles intent recognition, dialogue state tracking, frame-based context
management (entity carryover, pronoun resolution for follow-ups), and response generation.
"""

import re
import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class DialogueFrame:
    """Maintains active dialogue state, recognized entities, and context slots across turns."""
    active_state: str = "INITIAL"
    last_intent: Optional[str] = None
    current_entity: Optional[str] = None
    last_topic: Optional[str] = None
    last_passage_id: Optional[str] = None
    last_answer: Optional[str] = None
    turn_count: int = 0
    history: List[Dict[str, str]] = field(default_factory=list)

    def reset(self):
        """Resets the dialogue frame to default state."""
        self.active_state = "INITIAL"
        self.last_intent = None
        self.current_entity = None
        self.last_topic = None
        self.last_passage_id = None
        self.last_answer = None
        self.turn_count = 0
        self.history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Returns dictionary snapshot of frame for UI debugging."""
        return {
            "active_state": self.active_state,
            "last_intent": self.last_intent,
            "current_entity": self.current_entity,
            "last_topic": self.last_topic,
            "last_passage_id": self.last_passage_id,
            "turn_count": self.turn_count,
            "history_length": len(self.history)
        }


class IntentClassifier:
    """Rule- and pattern-based intent recognizer."""

    GREETING_PATTERNS = [
        r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening|day))\b",
        r"^what's\s+up\b",
        r"^yo\b"
    ]

    GRATITUDE_PATTERNS = [
        r"\b(thank\s+you|thanks|thx|much\s+appreciated|great\s+help|awesome\s+thanks)\b"
    ]

    GOODBYE_PATTERNS = [
        r"\b(bye|goodbye|see\s+you|farewell|cya|take\s+care|exit|quit)\b"
    ]

    HELP_PATTERNS = [
        r"\b(help|what\s+can\s+you\s+do|how\s+to\s+use|features|capabilities|commands)\b"
    ]

    CLEAR_PATTERNS = [
        r"\b(clear\s+chat|reset\s+chat|clear\s+history|start\s+over|restart)\b"
    ]

    FOLLOW_UP_PATTERNS = [
        r"\b(who\s+is\s+(he|she|they))\b",
        r"\b(what\s+did\s+(he|she|they|it)\s+do)\b",
        r"\b(where\s+is\s+(it|that))\b",
        r"\b(how\s+(tall|old|fast|high)\s+is\s+(it|he|she))\b",
        r"\b(when\s+was\s+(he|she|it)\s+born)\b",
        r"\b(tell\s+me\s+more(\s+about\s+(it|him|her))?)\b",
        r"\b(what\s+about\s+(him|her|it|them))\b",
        r"\b(what\s+is\s+(its|his|her|their)\s+[a-zA-Z\s]+)\b",
        r"\b(what\s+else\b)"
    ]

    KB_KEYWORDS = [
        "capital of", "currency of", "creator of", "founder of",
        "elevation of", "height of", "moons of", "chemical formula of",
        "speed of light", "born in", "official language of"
    ]

    @classmethod
    def classify(cls, user_input: str, frame: DialogueFrame) -> Tuple[str, Dict[str, Any]]:
        """
        Classifies incoming user query into an intent and extracts relevant entity/slot cues.
        
        Returns:
            Tuple of (intent_name, metadata_dict)
        """
        text = user_input.strip().lower()

        # 1. Greetings
        for p in cls.GREETING_PATTERNS:
            if re.search(p, text):
                return "GREETING", {}

        # 2. Gratitude
        for p in cls.GRATITUDE_PATTERNS:
            if re.search(p, text):
                return "GRATITUDE", {}

        # 3. Goodbye
        for p in cls.GOODBYE_PATTERNS:
            if re.search(p, text):
                return "GOODBYE", {}

        # 4. Help
        for p in cls.HELP_PATTERNS:
            if re.search(p, text):
                return "HELP", {}

        # 5. Clear
        for p in cls.CLEAR_PATTERNS:
            if re.search(p, text):
                return "CLEAR", {}

        # 6. Follow-up pronoun resolution (if frame has an active entity)
        for p in cls.FOLLOW_UP_PATTERNS:
            if re.search(p, text) and frame.current_entity:
                return "FOLLOW_UP", {"resolved_entity": frame.current_entity}

        # 7. Knowledge Base check
        for kw in cls.KB_KEYWORDS:
            if kw in text:
                return "KB_QUERY", {"matched_keyword": kw}

        # 8. Default: Reading comprehension / Extractive QA
        return "PASSAGE_QA", {}


class DialogueManager:
    """
    Manages conversational turn-taking, dialogue state machine transitions,
    and routing between chit-chat, SQLite KB, and Extractive QA.
    """

    GREETING_RESPONSES = [
        "Hello! I am your NLP Question Answering Assistant. How can I assist you with facts or text reading comprehension today?",
        "Hi there! Ask me a factual question (e.g. 'What is the capital of France?') or any question based on our knowledge corpus.",
        "Greetings! Ready to answer your questions using structured SQLite facts or DistilBERT extractive comprehension."
    ]

    GRATITUDE_RESPONSES = [
        "You're very welcome! Let me know if you have any other questions.",
        "Glad I could help! Feel free to ask more.",
        "My pleasure! Happy to explore more questions with you."
    ]

    GOODBYE_RESPONSES = [
        "Goodbye! Have a productive day ahead.",
        "See you later! Feel free to return whenever you need factual answers.",
        "Farewell! Chat history will remain available until you clear it."
    ]

    HELP_RESPONSE = (
        "**Here is what I can do:**\n\n"
        "1. **Structured Knowledge Base QA**: Query verified facts from SQLite (e.g. *'What is the capital of Japan?'*, *'Who created Python?'*, *'What is the elevation of Mount Everest?'*).\n"
        "2. **Information Retrieval & Extractive QA**: Ask reading comprehension questions across AI, Astronomy, History, and Science (e.g. *'Who introduced the Transformer architecture?'*, *'What are the two moons of Mars?'*).\n"
        "3. **Contextual Follow-ups**: Ask follow-up questions like *'When was he born?'* or *'Tell me more about it'* after asking about an entity.\n"
        "4. **Dataset & Evaluation**: View the corpus and run real benchmark evaluations (Precision@K, Exact Match, Token F1) via the sidebar navigation!"
    )

    FALLBACK_RESPONSE = (
        "I'm sorry, I couldn't find sufficient evidence in the knowledge base or corpus to answer that with confidence. "
        "Try rephrasing your question or check the **Dataset Explorer** to see the available topics."
    )

    def __init__(self, kb=None, retriever=None, qa_engine=None):
        self.kb = kb
        self.retriever = retriever
        self.qa_engine = qa_engine
        self.frame = DialogueFrame()

    def reset_chat(self) -> None:
        """Clears frame and history."""
        self.frame.reset()

    def resolve_followup_query(self, query: str, entity: str) -> str:
        """Substitutes pronouns (he/she/it/its/his/their) with the active entity name."""
        q = query
        # Handle possessive pronouns first
        q = re.sub(r'\b(its|his|her|their)\b', f"{entity}'s", q, flags=re.IGNORECASE)
        # Handle subject/object pronouns
        q = re.sub(r'\b(he|she|it|they|him|them)\b', entity, q, flags=re.IGNORECASE)
        if "tell me more" in q.lower():
            q = f"Tell me about {entity}"
        return q

    def process_turn(
        self,
        user_message: str,
        retrieval_top_k: int = 3,
        confidence_threshold: float = 0.15,
        use_kb: bool = True
    ) -> Dict[str, Any]:
        """
        Executes a complete dialogue turn:
        1. Classifies intent.
        2. Routes to appropriate sub-system (Chit-chat / KB / IR-QA).
        3. Updates dialogue state & frame.
        4. Returns structured response with provenance metadata.
        """
        self._current_user_msg = user_message
        self.frame.turn_count += 1
        intent, meta = IntentClassifier.classify(user_message, self.frame)
        self.frame.last_intent = intent

        # Turn record for history
        turn_meta: Dict[str, Any] = {
            "turn": self.frame.turn_count,
            "user_message": user_message,
            "intent": intent,
            "state_before": self.frame.active_state
        }

        # 1. GREETING
        if intent == "GREETING":
            self.frame.active_state = "GREETING"
            response_text = random.choice(self.GREETING_RESPONSES)
            return self._build_response(
                text=response_text,
                response_type="CHITCHAT",
                confidence=1.0,
                provenance="Dialogue Manager (Greeting Intent)"
            )

        # 2. GRATITUDE
        if intent == "GRATITUDE":
            self.frame.active_state = "GRATITUDE"
            response_text = random.choice(self.GRATITUDE_RESPONSES)
            return self._build_response(
                text=response_text,
                response_type="CHITCHAT",
                confidence=1.0,
                provenance="Dialogue Manager (Gratitude Intent)"
            )

        # 3. GOODBYE
        if intent == "GOODBYE":
            self.frame.active_state = "GOODBYE"
            response_text = random.choice(self.GOODBYE_RESPONSES)
            return self._build_response(
                text=response_text,
                response_type="CHITCHAT",
                confidence=1.0,
                provenance="Dialogue Manager (Goodbye Intent)"
            )

        # 4. HELP
        if intent == "HELP":
            self.frame.active_state = "HELP"
            return self._build_response(
                text=self.HELP_RESPONSE,
                response_type="CHITCHAT",
                confidence=1.0,
                provenance="Dialogue Manager (Help Intent)"
            )

        # 5. CLEAR
        if intent == "CLEAR":
            self.reset_chat()
            return self._build_response(
                text="Conversation history and active context have been cleared.",
                response_type="CHITCHAT",
                confidence=1.0,
                provenance="Dialogue Manager (Clear Chat Intent)"
            )

        # 6. FOLLOW-UP QUERY RESOLUTION
        actual_query = user_message
        if intent == "FOLLOW_UP" and self.frame.current_entity:
            self.frame.active_state = "FOLLOW_UP"
            actual_query = self.resolve_followup_query(user_message, self.frame.current_entity)

        # 7. KNOWLEDGE BASE ROUTING
        # Check SQLite Knowledge Base first if enabled
        if use_kb and self.kb is not None:
            kb_res = self.kb.answer_factoid_query(actual_query)
            if kb_res is not None:
                self.frame.active_state = "KB_ANSWERED"
                self.frame.current_entity = kb_res.get("entity")
                self.frame.last_topic = kb_res.get("attribute")
                self.frame.last_answer = kb_res.get("answer")

                answer_text = f"**{kb_res['answer']}**"
                return self._build_response(
                    text=answer_text,
                    response_type="KNOWLEDGE_BASE",
                    confidence=kb_res["confidence"],
                    provenance=kb_res["provenance"],
                    kb_data=kb_res,
                    resolved_query=actual_query if actual_query != user_message else None
                )

        # 8. INFORMATION RETRIEVAL + EXTRACTIVE QA ROUTING
        if self.retriever is not None and self.qa_engine is not None:
            self.frame.active_state = "IR_EXTRACTIVE_QA"

            # Retrieve top-k passages
            retrieved_passages = self.retriever.retrieve(actual_query, top_k=retrieval_top_k)

            # Run extractive DistilBERT QA
            qa_res = self.qa_engine.answer_from_passages(
                question=actual_query,
                passages=retrieved_passages,
                confidence_threshold=confidence_threshold
            )

            # Update frame with topic/passage info
            if qa_res.get("source_passage"):
                src = qa_res["source_passage"]
                self.frame.last_passage_id = src.get("passage_id")
                self.frame.last_topic = src.get("passage_title")
                self.frame.current_entity = src.get("passage_title")
                self.frame.last_answer = qa_res.get("answer")

            # Check if answer is confident or fallback
            if qa_res.get("is_confident", False):
                return self._build_response(
                    text=qa_res["answer"],
                    response_type="EXTRACTIVE_QA",
                    confidence=qa_res.get("confidence", 0.0),
                    combined_score=qa_res.get("combined_score", 0.0),
                    provenance=f"Retrieved Passage: {qa_res['source_passage'].get('passage_title')} ({qa_res['source_passage'].get('passage_id')})",
                    source_passage=qa_res.get("source_passage"),
                    candidates=qa_res.get("candidates", []),
                    retrieved_passages=retrieved_passages,
                    resolved_query=actual_query if actual_query != user_message else None
                )
            else:
                self.frame.active_state = "FALLBACK"
                return self._build_response(
                    text=qa_res["answer"],
                    response_type="FALLBACK",
                    confidence=qa_res.get("confidence", 0.0),
                    provenance="Confidence below threshold (insufficient evidence in corpus)",
                    source_passage=qa_res.get("source_passage"),
                    candidates=qa_res.get("candidates", []),
                    retrieved_passages=retrieved_passages,
                    resolved_query=actual_query if actual_query != user_message else None
                )

        # 9. GENERAL FALLBACK
        self.frame.active_state = "FALLBACK"
        return self._build_response(
            text=self.FALLBACK_RESPONSE,
            response_type="FALLBACK",
            confidence=0.0,
            provenance="Dialogue Manager (Fallback)"
        )

    def _build_response(
        self,
        text: str,
        response_type: str,
        confidence: float,
        provenance: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Constructs standardized turn response object."""
        self.frame.history.append({
            "turn": self.frame.turn_count,
            "user": getattr(self, "_current_user_msg", ""),
            "response": text,
            "response_type": response_type,
            "intent": self.frame.last_intent
        })
        resp = {
            "answer": text,
            "response_type": response_type,
            "confidence": round(confidence, 4),
            "provenance": provenance,
            "dialogue_frame": self.frame.to_dict(),
        }
        resp.update(kwargs)
        return resp
