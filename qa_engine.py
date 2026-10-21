"""
Extractive Question Answering Engine
====================================
Uses Hugging Face Transformers pretrained QA pipeline (DistilBERT fine-tuned on SQuAD)
to extract exact answer spans from retrieved passages, with multi-passage aggregation,
confidence scoring, and threshold-based fallback against hallucinations.
"""

import os
from typing import List, Dict, Any, Optional

DEFAULT_MODEL_NAME = "distilbert-base-cased-distilled-squad"


class ExtractiveQAEngine:
    """Hugging Face pipeline-based Extractive Question Answering engine."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: int = -1):
        self.model_name = model_name
        self.device = device
        self._pipeline = None
        self._load_error = None

    def _get_pipeline(self):
        """Lazy loads the Hugging Face question-answering pipeline."""
        if self._pipeline is None and self._load_error is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "question-answering",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=self.device
                )
            except Exception as e:
                self._load_error = str(e)
                print(f"Warning: Failed to load QA model '{self.model_name}': {e}")
        return self._pipeline

    def is_available(self) -> bool:
        """Returns True if the pipeline is successfully loaded."""
        return self._get_pipeline() is not None

    def answer_single_context(
        self,
        question: str,
        context: str,
        passage_id: str = "P00",
        passage_title: str = "Context",
        retrieval_score: float = 1.0
    ) -> Dict[str, Any]:
        """
        Extracts an answer span from a single passage context.
        
        Args:
            question: The user question.
            context: Passage text.
            passage_id: ID of the context passage.
            passage_title: Title of the context passage.
            retrieval_score: Relevance score from the retriever.
            
        Returns:
            Dict containing answer, score, start, end, and context metadata.
        """
        qa_pipe = self._get_pipeline()
        if qa_pipe is None:
            return {
                "answer": "QA model unavailable. (Error: " + str(self._load_error) + ")",
                "confidence": 0.0,
                "combined_score": 0.0,
                "start": 0,
                "end": 0,
                "passage_id": passage_id,
                "passage_title": passage_title,
                "passage_text": context,
                "is_confident": False
            }

        q_clean = question.strip()
        ctx_clean = context.strip()

        if not q_clean or not ctx_clean:
            return {
                "answer": "No answer could be determined from empty input.",
                "confidence": 0.0,
                "combined_score": 0.0,
                "start": 0,
                "end": 0,
                "passage_id": passage_id,
                "passage_title": passage_title,
                "passage_text": context,
                "is_confident": False
            }

        try:
            res = qa_pipe(question=q_clean, context=ctx_clean)
            conf = float(res.get("score", 0.0))
            ans = str(res.get("answer", "")).strip()
            start = int(res.get("start", 0))
            end = int(res.get("end", 0))

            # Combined score: weighted average of extractive model confidence and IR score
            combined = round(0.70 * conf + 0.30 * min(max(retrieval_score, 0.0), 1.0), 4)

            return {
                "answer": ans,
                "confidence": round(conf, 4),
                "combined_score": combined,
                "start": start,
                "end": end,
                "passage_id": passage_id,
                "passage_title": passage_title,
                "passage_text": context,
                "is_confident": conf >= 0.15
            }
        except Exception as e:
            return {
                "answer": f"Inference error: {e}",
                "confidence": 0.0,
                "combined_score": 0.0,
                "start": 0,
                "end": 0,
                "passage_id": passage_id,
                "passage_title": passage_title,
                "passage_text": context,
                "is_confident": False
            }

    def answer_from_passages(
        self,
        question: str,
        passages: List[Dict[str, Any]],
        confidence_threshold: float = 0.15
    ) -> Dict[str, Any]:
        """
        Runs extractive QA across multiple retrieved passages, scores and aggregates candidates,
        and enforces confidence thresholding to prevent fabricated/hallucinated answers.
        
        Args:
            question: User question.
            passages: List of passage dicts from the retriever.
            confidence_threshold: Minimum required confidence score.
            
        Returns:
            Dict with best answer, source passage, confidence, candidate rankings, and status.
        """
        if not passages:
            return {
                "answer": "I could not find any relevant passages in the knowledge corpus to answer your question.",
                "best_guess": "",
                "confidence": 0.0,
                "is_confident": False,
                "status": "NO_PASSAGES",
                "source_passage": None,
                "candidates": []
            }

        max_retrieval_score = max((p.get("retrieval_score", 0.0) for p in passages), default=0.0)
        if max_retrieval_score < 0.05:
            return {
                "answer": "I could not find sufficient evidence in the retrieved passages to answer this question with high confidence.",
                "best_guess": "",
                "confidence": 0.0,
                "is_confident": False,
                "status": "LOW_CONFIDENCE",
                "source_passage": passages[0],
                "candidates": []
            }

        candidates = []
        for p in passages:
            p_id = p.get("id", "P00")
            p_title = p.get("title", "Untitled")
            p_text = p.get("passage", "")
            r_score = p.get("retrieval_score", 0.0)

            cand = self.answer_single_context(
                question=question,
                context=p_text,
                passage_id=p_id,
                passage_title=p_title,
                retrieval_score=r_score
            )
            candidates.append(cand)

        # Sort candidates primarily by model confidence and combined score
        candidates.sort(key=lambda x: (x["confidence"], x["combined_score"]), reverse=True)
        best = candidates[0]

        # Check threshold
        if best["confidence"] < confidence_threshold or not best["answer"]:
            return {
                "answer": "I could not find sufficient evidence in the retrieved passages to answer this question with high confidence.",
                "best_guess": best["answer"],
                "confidence": best["confidence"],
                "is_confident": False,
                "status": "LOW_CONFIDENCE",
                "source_passage": best,
                "candidates": candidates
            }

        return {
            "answer": best["answer"],
            "best_guess": best["answer"],
            "confidence": best["confidence"],
            "combined_score": best["combined_score"],
            "is_confident": True,
            "status": "SUCCESS",
            "source_passage": best,
            "candidates": candidates
        }
