from __future__ import annotations

from typing import Any

from flask import current_app

from app.errors.exceptions import NotFoundError, ValidationError
from app.machine_learning.common.cross_validation import evaluate_with_cross_validation
from app.machine_learning.common.utils import build_training_text
from app.machine_learning.decision_tree.trainer import DecisionTreeTrainer
from app.machine_learning.knn.trainer import KNNTrainer
from app.repositories.candidate_profile_repository import CandidateProfileRepository
from app.repositories.dataset_sample_repository import DatasetSampleRepository
from app.repositories.job_profile_repository import JobProfileRepository


class ModelValidationService:
    def __init__(self):
        self.dataset_sample_repository = DatasetSampleRepository()
        self.candidate_profile_repository = CandidateProfileRepository()
        self.job_profile_repository = JobProfileRepository()

    def validate_job_profile_with_cross_validation(
        self,
        job_profile_id: str,
        folds: int | None = None,
    ) -> dict[str, Any]:
        job_profile = self.job_profile_repository.get_by_id(job_profile_id)

        if not job_profile:
            raise NotFoundError(f"Job profile '{job_profile_id}' was not found")

        texts, labels = self._build_validation_dataset(
            job_profile_id=job_profile_id,
            job_profile=job_profile,
        )

        if len(texts) < current_app.config["MIN_TRAINING_ROWS"]:
            raise ValidationError(
                f"No hay suficientes registros para validar. Actual: {len(texts)}"
            )

        vectorizer_params = {
            "max_features": current_app.config["TFIDF_MAX_FEATURES"],
            "ngram_range": current_app.config["TFIDF_NGRAM_RANGE"],
            "lowercase": True,
            "strip_accents": "unicode",
        }

        requested_folds = folds or current_app.config["CROSS_VALIDATION_FOLDS"]
        random_state = current_app.config["TRAINING_RANDOM_STATE"]

        knn_estimator = KNNTrainer(
            n_neighbors=current_app.config["KNN_DEFAULT_NEIGHBORS"],
        ).build_estimator(sample_size=len(texts))

        tree_estimator = DecisionTreeTrainer(
            max_depth=current_app.config["TREE_MAX_DEPTH"],
            min_samples_leaf=current_app.config["TREE_MIN_SAMPLES_LEAF"],
            random_state=random_state,
        ).build_estimator()

        results = {
            "knn": evaluate_with_cross_validation(
                texts=texts,
                labels=labels,
                estimator=knn_estimator,
                vectorizer_params=vectorizer_params,
                requested_folds=requested_folds,
                random_state=random_state,
            ),
            "decision_tree": evaluate_with_cross_validation(
                texts=texts,
                labels=labels,
                estimator=tree_estimator,
                vectorizer_params=vectorizer_params,
                requested_folds=requested_folds,
                random_state=random_state,
            ),
        }

        selected_algorithm = self._select_best_algorithm(results)

        return {
            "job_profile_id": job_profile_id,
            "job_profile_title": job_profile.get("title"),
            "total_samples": len(texts),
            "requested_folds": requested_folds,
            "method": "StratifiedKFold",
            "selection_criteria": (
                "Mayor Recall promedio, luego mayor F1-Score promedio y menor "
                "desviación estándar del F1-Score."
            ),
            "results": results,
            "selected_algorithm_by_cross_validation": selected_algorithm,
        }

    def _build_validation_dataset(
        self,
        job_profile_id: str,
        job_profile: dict,
    ) -> tuple[list[str], list[int]]:
        samples = self.dataset_sample_repository.get_rows(
            job_profile_id=job_profile_id,
            ready_only=True,
            labeled_only=True,
        )

        if not samples:
            raise ValidationError("No existen registros listos en dataset_sample")

        candidate_profile_ids = [
            sample["candidate_profile_id"]
            for sample in samples
            if sample.get("candidate_profile_id")
        ]

        candidate_profiles = self.candidate_profile_repository.get_by_ids(candidate_profile_ids)
        candidate_profile_map = {
            profile["id"]: profile
            for profile in candidate_profiles
            if profile.get("id")
        }

        texts: list[str] = []
        labels: list[int] = []

        for sample in samples:
            candidate_profile = candidate_profile_map.get(sample.get("candidate_profile_id"))
            if not candidate_profile:
                continue

            training_text = build_training_text(job_profile, candidate_profile)
            if not training_text.strip():
                continue

            texts.append(training_text)
            labels.append(1 if sample.get("label") else 0)

        return texts, labels

    @staticmethod
    def _select_best_algorithm(results: dict) -> str | None:
        valid_results = {
            algorithm: metrics
            for algorithm, metrics in results.items()
            if metrics.get("enabled")
        }

        if not valid_results:
            return None

        return sorted(
            valid_results.items(),
            key=lambda item: (
                -item[1].get("recall_mean", 0),
                -item[1].get("f1_score_mean", 0),
                item[1].get("f1_score_std", 999),
            ),
        )[0][0]
