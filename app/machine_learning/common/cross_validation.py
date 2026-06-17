from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


METRIC_KEYS = {
    "accuracy": "test_accuracy",
    "precision": "test_precision",
    "recall": "test_recall",
    "f1_score": "test_f1_score",
}


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
        return _disabled_result(
            reason="No hay suficientes registros para validación cruzada",
            labels=labels,
        )

    label_counts = Counter(labels)

    if len(label_counts) < 2:
        return _disabled_result(
            reason="Se requieren al menos dos clases para validación cruzada",
            labels=labels,
        )

    min_class_count = min(label_counts.values())
    n_splits = min(int(requested_folds or 5), min_class_count)

    if n_splits < 2:
        return _disabled_result(
            reason="La clase minoritaria tiene menos de 2 registros",
            labels=labels,
        )

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
        "f1_score": make_scorer(f1_score, zero_division=0),
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
        return_estimator=False,
        error_score="raise",
    )

    fold_scores = {
        metric_name: [float(value) for value in result[result_key].tolist()]
        for metric_name, result_key in METRIC_KEYS.items()
    }

    metric_summary = {
        metric_name: _summarize_metric(values)
        for metric_name, values in fold_scores.items()
    }

    return {
        "enabled": True,
        "method": "StratifiedKFold",
        "folds": n_splits,
        "requested_folds": int(requested_folds or 5),
        "random_state": int(random_state),
        "class_distribution": {str(key): int(value) for key, value in label_counts.items()},
        "fold_scores": fold_scores,
        "metric_summary": metric_summary,
        "accuracy_mean": metric_summary["accuracy"]["mean"],
        "accuracy_std": metric_summary["accuracy"]["std"],
        "precision_mean": metric_summary["precision"]["mean"],
        "precision_std": metric_summary["precision"]["std"],
        "recall_mean": metric_summary["recall"]["mean"],
        "recall_std": metric_summary["recall"]["std"],
        "f1_score_mean": metric_summary["f1_score"]["mean"],
        "f1_score_std": metric_summary["f1_score"]["std"],
    }


def _disabled_result(reason: str, labels: list[int]) -> dict:
    label_counts = Counter(labels)

    return {
        "enabled": False,
        "method": "StratifiedKFold",
        "reason": reason,
        "class_distribution": {str(key): int(value) for key, value in label_counts.items()},
    }


def _summarize_metric(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)

    return {
        "mean": round(float(mean_value), 4),
        "std": round(float(variance**0.5), 4),
        "min": round(float(min(values)), 4),
        "max": round(float(max(values)), 4),
    }
