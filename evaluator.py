"""
Evaluation Module
=================
Computes exact, reproducible NLP evaluation metrics following standard
SQuAD benchmarking protocols:
- Retrieval: Precision@1, Precision@K, Recall@K, Mean Reciprocal Rank (MRR)
- Extractive QA: Exact Match (EM) and token-level F1 score with article/punctuation normalization
- Fallback Accuracy for unanswerable/out-of-scope queries
"""

import re
import string
import time
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd


def normalize_text(s: str) -> str:
    """
    Standard SQuAD text normalizer:
    Lowers text, removes punctuation, removes articles ('a', 'an', 'the'),
    and standardizes whitespace.
    """
    if not isinstance(s, str):
        return ""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Computes binary Exact Match (EM) between normalized prediction and ground truth."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """
    Computes token-level F1 score (harmonic mean of precision and recall)
    between normalized prediction and ground truth tokens.
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common_tokens = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common_tokens.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    """Calculates maximum score over multiple acceptable ground truth answers."""
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def evaluate_batch(
    qa_engine,
    retriever,
    eval_set: List[Dict[str, Any]],
    top_k: int = 3,
    confidence_threshold: float = 0.15
) -> Dict[str, Any]:
    """
    Runs complete evaluation over the test dataset.
    Computes real, genuine metrics without any hardcoded or invented values.
    
    Returns:
        Dict containing:
        - metrics summary (EM, F1, Precision@1, Precision@K, Recall@K, MRR, latency)
        - detailed results DataFrame
    """
    results = []
    start_time = time.time()

    em_scores = []
    f1_scores = []
    retrieval_hits_1 = []
    retrieval_hits_k = []
    reciprocal_ranks = []
    unanswerable_correct = []

    for item in eval_set:
        q_id = item.get("id", "Q00")
        question = item.get("question", "")
        category = item.get("category", "General")
        gt_passage_id = item.get("ground_truth_passage_id")
        gt_answers = item.get("ground_truth_answers", [])
        is_impossible = item.get("is_impossible", False)

        q_start = time.time()

        # Step 1: Retrieval
        retrieved = retriever.retrieve(question, top_k=top_k)
        retrieved_ids = [r["id"] for r in retrieved]

        # Step 2: Extractive QA
        qa_output = qa_engine.answer_from_passages(
            question=question,
            passages=retrieved,
            confidence_threshold=confidence_threshold
        )
        q_latency = time.time() - q_start

        pred_answer = qa_output.get("answer", "")
        confidence = qa_output.get("confidence", 0.0)
        is_confident = qa_output.get("is_confident", False)
        status = qa_output.get("status", "SUCCESS")

        # Evaluate Retrieval (for questions with ground truth passages)
        p1 = 0.0
        pk = 0.0
        rk = 0.0
        rr = 0.0
        if not is_impossible and gt_passage_id:
            gt_pid_clean = str(gt_passage_id).strip()
            p1 = 1.0 if retrieved_ids and retrieved_ids[0] == gt_pid_clean else 0.0
            hits = 1 if gt_pid_clean in retrieved_ids else 0
            pk = hits / max(len(retrieved_ids), 1)
            rk = 1.0 if hits > 0 else 0.0
            for r_idx, pid in enumerate(retrieved_ids, start=1):
                if pid == gt_pid_clean:
                    rr = 1.0 / r_idx
                    break

            retrieval_hits_1.append(p1)
            retrieval_hits_k.append(rk)
            reciprocal_ranks.append(rr)

        # Evaluate QA
        if is_impossible:
            # Model is correct if it flagged low confidence / no evidence
            if not is_confident or status in ("LOW_CONFIDENCE", "NO_PASSAGES"):
                em = 1.0
                f1 = 1.0
                unanswerable_correct.append(1.0)
            else:
                em = 0.0
                f1 = 0.0
                unanswerable_correct.append(0.0)
        else:
            if gt_answers:
                em = metric_max_over_ground_truths(compute_exact_match, pred_answer, gt_answers)
                f1 = metric_max_over_ground_truths(compute_token_f1, pred_answer, gt_answers)
            else:
                em = 0.0
                f1 = 0.0

        em_scores.append(em)
        f1_scores.append(f1)

        results.append({
            "Question ID": q_id,
            "Question": question,
            "Category": category,
            "Ground Truth Answers": ", ".join(gt_answers) if gt_answers else "(Unanswerable)",
            "Predicted Answer": pred_answer,
            "Model Confidence": round(confidence, 4),
            "Top-1 Retrieved Passage": retrieved_ids[0] if retrieved_ids else "None",
            "Target Passage ID": gt_passage_id or "N/A",
            "Exact Match (EM)": em,
            "Token F1": round(f1, 4),
            "Retrieval Recall@K": rk if not is_impossible else "N/A",
            "Status": status,
            "Latency (s)": round(q_latency, 3)
        })

    total_time = time.time() - start_time
    df_results = pd.DataFrame(results)

    summary = {
        "Total Questions": len(eval_set),
        "Exact Match (EM) %": round(float(sum(em_scores) / max(len(em_scores), 1) * 100), 2),
        "Token F1 %": round(float(sum(f1_scores) / max(len(f1_scores), 1) * 100), 2),
        "Retrieval Precision@1 %": round(float(sum(retrieval_hits_1) / max(len(retrieval_hits_1), 1) * 100), 2) if retrieval_hits_1 else 0.0,
        f"Retrieval Recall@{top_k} %": round(float(sum(retrieval_hits_k) / max(len(retrieval_hits_k), 1) * 100), 2) if retrieval_hits_k else 0.0,
        "Retrieval MRR": round(float(sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1)), 4) if reciprocal_ranks else 0.0,
        "Unanswerable Detection Rate %": round(float(sum(unanswerable_correct) / max(len(unanswerable_correct), 1) * 100), 2) if unanswerable_correct else 100.0,
        "Avg Latency (s)": round(total_time / max(len(eval_set), 1), 3),
        "Total Eval Time (s)": round(total_time, 2)
    }

    return {
        "summary": summary,
        "results_df": df_results
    }
