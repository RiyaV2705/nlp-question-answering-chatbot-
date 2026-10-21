"""
Data Loader Module
==================
Handles loading, caching, cleaning, and preprocessing of text corpora,
passage datasets, and labeled QA evaluation datasets.
"""

import os
import re
import json
import string
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

# Default paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_CORPUS_PATH = os.path.join(DATA_DIR, "passages_corpus.csv")
DEFAULT_EVAL_PATH = os.path.join(DATA_DIR, "test_eval_set.json")

# Fallback passages if file is missing
FALLBACK_PASSAGES = [
    {
        "id": "P01",
        "title": "Alan Turing",
        "category": "AI & Computer Science",
        "passage": "Alan Turing was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist. During World War II, Turing worked for the Government Code and Cypher School at Bletchley Park, cracking the German Enigma machine."
    },
    {
        "id": "P02",
        "title": "Transformer Architecture",
        "category": "AI & Computer Science",
        "passage": "The Transformer is a deep learning architecture introduced in the 2017 paper 'Attention Is All You Need' by Ashish Vaswani and Google Brain researchers. It relies entirely on self-attention mechanisms without recurrence or convolution."
    },
    {
        "id": "P04",
        "title": "Python Programming Language",
        "category": "AI & Computer Science",
        "passage": "Python is a high-level, general-purpose programming language conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands. It was officially released in February 1991."
    }
]


def preprocess_text(text: str, lowercase: bool = True, remove_punctuation: bool = False) -> str:
    """
    Cleans and standardizes raw text.
    
    Args:
        text: Input string.
        lowercase: Whether to convert text to lower case.
        remove_punctuation: Whether to strip punctuation.
        
    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    if lowercase:
        cleaned = cleaned.lower()
        
    if remove_punctuation:
        cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
        
    return cleaned


def tokenize_words(text: str) -> List[str]:
    """Simple whitespace + punctuation-aware word tokenizer."""
    cleaned = preprocess_text(text, lowercase=True, remove_punctuation=True)
    return [w for w in cleaned.split() if w]


def tokenize_words_nltk(text: str) -> List[str]:
    """Tokenizes text using NLTK word_tokenize with fallback to RegexpTokenizer or split."""
    if not isinstance(text, str):
        return []
    try:
        import nltk
        try:
            return nltk.word_tokenize(text)
        except LookupError:
            from nltk.tokenize import RegexpTokenizer
            tokenizer = RegexpTokenizer(r'\w+')
            return tokenizer.tokenize(text)
    except ImportError:
        return tokenize_words(text)


def split_sentences_nltk(text: str) -> List[str]:
    """Splits text into sentences using NLTK sent_tokenize with regex fallback."""
    if not isinstance(text, str):
        return []
    try:
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    except ImportError:
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


def load_corpus(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads passage corpus from CSV. If not found, falls back to built-in passages.
    
    Returns:
        DataFrame with columns: ['id', 'title', 'category', 'passage']
    """
    target_path = filepath or DEFAULT_CORPUS_PATH
    if os.path.exists(target_path):
        try:
            df = pd.read_csv(target_path, encoding="utf-8")
            required_cols = {"id", "title", "category", "passage"}
            if not required_cols.issubset(set(df.columns)):
                missing = required_cols - set(df.columns)
                raise ValueError(f"Corpus CSV missing required columns: {missing}")
            # Ensure text columns are string and stripped
            df["passage"] = df["passage"].fillna("").astype(str).str.strip()
            df["title"] = df["title"].fillna("Untitled").astype(str).str.strip()
            df["category"] = df["category"].fillna("General").astype(str).str.strip()
            df["id"] = df["id"].fillna("").astype(str).str.strip()
            return df
        except Exception as e:
            print(f"Warning: Failed to read {target_path} ({e}). Using fallback corpus.")
            return pd.DataFrame(FALLBACK_PASSAGES)
    else:
        print(f"Warning: File {target_path} not found. Using fallback corpus.")
        return pd.DataFrame(FALLBACK_PASSAGES)


def load_eval_dataset(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads labeled QA evaluation dataset from JSON.
    
    Returns:
        List of dicts with fields:
        id, question, category, ground_truth_passage_id, ground_truth_answers, is_impossible
    """
    target_path = filepath or DEFAULT_EVAL_PATH
    if os.path.exists(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Warning: Failed to read {target_path} ({e}). Returning empty list.")
            return []
    return []


def get_passage_by_id(df: pd.DataFrame, passage_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves passage dictionary by passage ID."""
    match = df[df["id"] == passage_id]
    if not match.empty:
        return match.iloc[0].to_dict()
    return None
