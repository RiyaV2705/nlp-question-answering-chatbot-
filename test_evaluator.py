"""
Unit tests for Evaluator module (Exact Match, Token F1, Normalization).
"""

import pytest
from src.evaluator import normalize_text, compute_exact_match, compute_token_f1, metric_max_over_ground_truths


def test_normalize_text():
    assert normalize_text("The Enigma Machine.") == "enigma machine"
    assert normalize_text("  a  test   case!  ") == "test case"
    assert normalize_text("An Apple, A Banana.") == "apple banana"


def test_exact_match():
    # Identical after normalization
    assert compute_exact_match("The Enigma Machine", "enigma machine") == 1.0
    assert compute_exact_match("June 23, 1912", "June 23 1912") == 1.0

    # Non-match
    assert compute_exact_match("Alan Turing", "Guido van Rossum") == 0.0


def test_token_f1():
    # Exact match -> F1 = 1.0
    f1 = compute_token_f1("Ashish Vaswani and colleagues", "Ashish Vaswani and colleagues")
    assert f1 == 1.0

    # Partial overlap
    # Pred: "Ashish Vaswani" (2 tokens: ashish, vaswani)
    # GT: "Ashish Vaswani and colleagues" (4 tokens: ashish, vaswani, and, colleagues)
    # common = 2, precision = 2/2 = 1.0, recall = 2/4 = 0.5 -> F1 = 2 * 1.0 * 0.5 / 1.5 = 0.6667
    f1_partial = compute_token_f1("Ashish Vaswani", "Ashish Vaswani and colleagues")
    assert round(f1_partial, 3) == 0.667

    # Disjoint tokens
    f1_zero = compute_token_f1("London", "Paris")
    assert f1_zero == 0.0

    # Both empty
    assert compute_token_f1("", "") == 1.0


def test_metric_max_over_ground_truths():
    gt_list = ["June 23, 1912", "June 23 1912", "23 June 1912"]
    pred = "June 23 1912"
    em = metric_max_over_ground_truths(compute_exact_match, pred, gt_list)
    assert em == 1.0

    f1 = metric_max_over_ground_truths(compute_token_f1, pred, gt_list)
    assert f1 == 1.0
