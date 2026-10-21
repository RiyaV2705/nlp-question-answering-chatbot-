"""
Kaggle Dataset Download and Importer Script
===========================================
This utility assists in downloading and converting full reading comprehension / QA datasets
from Kaggle into the format used by the NLP Question Answering System.

Recommended Kaggle Datasets:
1. Stanford Question Answering Dataset (SQuAD):
   URL: https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset
   Kaggle CLI: kaggle datasets download -d stanfordu/stanford-question-answering-dataset

2. SQuAD 2.0 (with unanswerable questions):
   URL: https://rajpurkar.github.io/SQuAD-explorer/
   Train JSON: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
   Dev JSON:   https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

Usage:
------
1. Using Kaggle API:
   - Ensure you have `kaggle` installed (`pip install kaggle`)
   - Place your `kaggle.json` API token in `~/.kaggle/kaggle.json` (or `C:\\Users\\<User>\\.kaggle\\kaggle.json` on Windows)
   - Run: python data/download_kaggle_dataset.py --source kaggle

2. Download direct SQuAD dev sample:
   - Run: python data/download_kaggle_dataset.py --source url --limit 100
"""

import os
import json
import argparse
import pandas as pd
import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_CSV = os.path.join(DATA_DIR, "passages_corpus.csv")
EVAL_JSON = os.path.join(DATA_DIR, "test_eval_set.json")


def convert_squad_json_to_corpus(squad_data, max_passages=150, max_eval_questions=30):
    """
    Parses SQuAD format (v1.1 or v2.0) and generates:
    1. Passages corpus DataFrame (id, title, category, passage)
    2. Evaluation questions list (id, question, category, ground_truth_passage_id, ground_truth_answers, is_impossible)
    """
    passages = []
    eval_questions = []

    p_idx = 1
    q_idx = 1

    for article in squad_data.get("data", []):
        title = article.get("title", "General")
        category = "Reading Comprehension / " + title

        for paragraph in article.get("paragraphs", []):
            if len(passages) >= max_passages:
                break

            passage_id = f"P{p_idx:03d}"
            context = paragraph.get("context", "").strip()
            if not context:
                continue

            passages.append({
                "id": passage_id,
                "title": title.replace("_", " "),
                "category": category,
                "passage": context
            })
            p_idx += 1

            for qas in paragraph.get("qas", []):
                if len(eval_questions) >= max_eval_questions:
                    continue

                question_text = qas.get("question", "").strip()
                is_impossible = qas.get("is_impossible", False)
                answers = [a.get("text", "").strip() for a in qas.get("answers", []) if a.get("text")]

                if is_impossible or not answers:
                    eval_questions.append({
                        "id": f"Q{q_idx:03d}",
                        "question": question_text,
                        "category": category,
                        "ground_truth_passage_id": None,
                        "ground_truth_answers": [],
                        "is_impossible": True
                    })
                else:
                    eval_questions.append({
                        "id": f"Q{q_idx:03d}",
                        "question": question_text,
                        "category": category,
                        "ground_truth_passage_id": passage_id,
                        "ground_truth_answers": answers,
                        "is_impossible": False
                    })
                q_idx += 1

        if len(passages) >= max_passages:
            break

    df_passages = pd.DataFrame(passages)
    return df_passages, eval_questions


def fetch_from_url(limit=100):
    """Downloads sample SQuAD dev JSON directly from official repository."""
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    print(f"Fetching official SQuAD sample from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        squad_json = response.json()
        df, eval_data = convert_squad_json_to_corpus(squad_json, max_passages=limit, max_eval_questions=30)

        out_csv = os.path.join(DATA_DIR, "passages_squad_imported.csv")
        out_eval = os.path.join(DATA_DIR, "test_eval_squad_imported.json")
        df.to_csv(out_csv, index=False)
        with open(out_eval, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2)

        print(f"Successfully imported {len(df)} passages to {out_csv}")
        print(f"Successfully imported {len(eval_data)} QA pairs to {out_eval}")
    except Exception as e:
        print(f"Error fetching SQuAD dataset: {e}")
        print("Note: The offline sample dataset in data/passages_corpus.csv is ready to use.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle / SQuAD QA Dataset Downloader")
    parser.add_argument("--source", choices=["url", "kaggle"], default="url")
    parser.add_argument("--limit", type=int, default=50, help="Max passages to extract")
    args = parser.parse_args()

    if args.source == "url":
        fetch_from_url(limit=args.limit)
    else:
        print("To download via Kaggle CLI:")
        print("  1. Ensure kaggle.json is present in ~/.kaggle/")
        print("  2. Run: kaggle datasets download -d stanfordu/stanford-question-answering-dataset")
        print("  3. Unzip and use convert_squad_json_to_corpus() with the extracted json file.")
