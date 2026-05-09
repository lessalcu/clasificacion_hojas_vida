from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


def evaluate_with_cross_validation(
    texts: list[str],
    labels: list[int],
    estimator: Any,
    vectorizer_params: dict,
    requested_folds: int = 5,
    random_state: int = 42,
) -> dict:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    if len(texts) < 4:
        return {
            "enabled": False,
            "reason": "No hay suficientes registros para validación cruzada",
        }

    label_counts = Counter(labels)

    if len(label_counts) < 2:
        return {
            "enabled": False,
            "reason": "Se requieren al menos dos clases para validación cruzada",
        }

    min_class_count = min(label_counts.values())
    n_splits = min(requested_folds, min_class_count)

    if n_splits < 2:
        return {
            "enabled": False,
            "reason": "La clase minoritaria tiene menos de 2 registros",
            "class_distribution": dict(label_counts),
        }

    pipeline = Pipeline(
        [
            ("vectorizer", TfidfVectorizer(**vectorizer_params)),
            ("estimator", estimator),
        ]
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    result = cross_validate(
        pipeline,
        texts,
        labels,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    return {
        "enabled": True,
        "folds": n_splits,
        "class_distribution": dict(label_counts),
        "accuracy_mean": float(result["test_accuracy"].mean()),
        "accuracy_std": float(result["test_accuracy"].std()),
        "precision_mean": float(result["test_precision"].mean()),
        "precision_std": float(result["test_precision"].std()),
        "recall_mean": float(result["test_recall"].mean()),
        "recall_std": float(result["test_recall"].std()),
        "f1_score_mean": float(result["test_f1"].mean()),
        "f1_score_std": float(result["test_f1"].std()),
    }
