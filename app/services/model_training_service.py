from __future__ import annotations

import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import current_app
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from app.errors.exceptions import NotFoundError, ValidationError
from app.machine_learning.common.cross_validation import evaluate_with_cross_validation
from app.machine_learning.common.evaluation import evaluate_predictions
from app.machine_learning.common.schemas import ModelEvaluation, TrainingDatasetRow
from app.machine_learning.common.serialization import persist_model_artifacts
from app.machine_learning.common.utils import (
    build_candidate_text,
    build_job_profile_text,
    build_training_text,
)
from app.machine_learning.decision_tree.trainer import DecisionTreeTrainer
from app.machine_learning.knn.trainer import KNNTrainer
from app.repositories.candidate_profile_repository import CandidateProfileRepository
from app.repositories.dataset_sample_repository import DatasetSampleRepository
from app.repositories.job_profile_repository import JobProfileRepository
from app.repositories.training_run_repository import TrainingRunRepository
from app.services.dataset_builder_service import DatasetBuilderService
from app.services.model_versioning_service import ModelVersioningService
from app.services.storage_service import StorageService
from app.services.training_report_service import TrainingReportService


class ModelTrainingService:
    def __init__(self):
        self.dataset_sample_repository = DatasetSampleRepository()
        self.candidate_profile_repository = CandidateProfileRepository()
        self.job_profile_repository = JobProfileRepository()
        self.training_run_repository = TrainingRunRepository()
        self.dataset_builder_service = DatasetBuilderService()
        self.storage_service = StorageService()
        self.model_versioning_service = ModelVersioningService()
        self.training_report_service = TrainingReportService()

    def train_job_profile(
        self,
        job_profile_id: str,
        created_by: str | None = None,
        dataset_version: str = "api-training-run",
        persist_to_storage: bool = True,
        auto_build_dataset: bool = True,
        random_state: int | None = None,
        iteration_number: int | None = None,
        experiment_run_id: str | None = None,
        save_training_report: bool = True,
    ) -> dict[str, Any]:
        job_profile = self.job_profile_repository.get_by_id(job_profile_id)

        if not job_profile:
            raise NotFoundError(f"Job profile '{job_profile_id}' was not found")

        current_random_state = self._normalize_random_state(random_state)
        dataset_build_result = None

        if auto_build_dataset and current_app.config["AUTO_BUILD_DATASET_ON_TRAIN"]:
            dataset_build_result = self.dataset_builder_service.ensure_dataset_for_training(
                job_profile_id=job_profile_id,
                created_by=created_by,
                dataset_version=f"{dataset_version}-auto-dataset",
            )

        dataset_rows = self._build_training_dataset(job_profile_id)

        min_rows = current_app.config["MIN_TRAINING_ROWS"]
        if len(dataset_rows) < min_rows:
            raise ValidationError(
                f"No hay suficientes registros listos en dataset_sample para entrenar. "
                f"Actual: {len(dataset_rows)}. Mínimo requerido: {min_rows}."
            )

        labels = [1 if row.label else 0 for row in dataset_rows]
        unique_labels = sorted(set(labels))

        if len(unique_labels) < 2:
            raise ValidationError(
                "El dataset solo tiene una clase. Para entrenar se necesitan ejemplos "
                "positivos y negativos."
            )

        texts = [row.training_text for row in dataset_rows]
        vectorizer_params = self._build_vectorizer_params()

        stratify = labels if self._can_stratify(labels) else None
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=current_app.config["TRAIN_TEST_SIZE"],
            random_state=current_random_state,
            stratify=stratify,
        )

        vectorizer = TfidfVectorizer(**vectorizer_params)
        X_train = vectorizer.fit_transform(X_train_texts)
        X_test = vectorizer.transform(X_test_texts)

        dataset_summary = self._build_dataset_summary(
            dataset_rows=dataset_rows,
            labels=labels,
            y_train=y_train,
            y_test=y_test,
            dataset_version=dataset_version,
            job_profile_id=job_profile_id,
            random_state=current_random_state,
        )

        knn_estimator = KNNTrainer(
            n_neighbors=current_app.config["KNN_DEFAULT_NEIGHBORS"],
        ).build_estimator(sample_size=len(y_train))

        tree_estimator = DecisionTreeTrainer(
            max_depth=current_app.config["TREE_MAX_DEPTH"],
            min_samples_leaf=current_app.config["TREE_MIN_SAMPLES_LEAF"],
            random_state=current_random_state,
        ).build_estimator()

        knn_cross_validation = self._run_cross_validation_if_enabled(
            texts=texts,
            labels=labels,
            estimator=knn_estimator,
            vectorizer_params=vectorizer_params,
            random_state=current_random_state,
        )

        tree_cross_validation = self._run_cross_validation_if_enabled(
            texts=texts,
            labels=labels,
            estimator=tree_estimator,
            vectorizer_params=vectorizer_params,
            random_state=current_random_state,
        )

        evaluations = [
            self._train_algorithm(
                algorithm="knn",
                estimator=knn_estimator,
                vectorizer=vectorizer,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                job_profile_id=job_profile_id,
                created_by=created_by,
                dataset_version=dataset_version,
                dataset_summary=dataset_summary,
                model_parameters=self._build_model_parameters("knn", current_random_state),
                cross_validation=knn_cross_validation,
                persist_to_storage=persist_to_storage,
            ),
            self._train_algorithm(
                algorithm="decision_tree",
                estimator=tree_estimator,
                vectorizer=vectorizer,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                job_profile_id=job_profile_id,
                created_by=created_by,
                dataset_version=dataset_version,
                dataset_summary=dataset_summary,
                model_parameters=self._build_model_parameters("decision_tree", current_random_state),
                cross_validation=tree_cross_validation,
                persist_to_storage=persist_to_storage,
            ),
        ]

        best_model = self._select_best_model(evaluations)
        selection_criteria = self._build_selection_criteria(evaluations)

        selection_reason = (
            f"Modelo seleccionado automáticamente. Criterio: {selection_criteria} "
            f"Algoritmo ganador: {best_model.algorithm}. "
            f"Recall={best_model.recall}, F1={best_model.f1_score}, "
            f"Duration={best_model.duration_ms}ms."
        )

        active_model_record = self.model_versioning_service.activate_model(
            model_version_id=best_model.model_version_id,
            created_by=created_by,
            reason=selection_reason,
        )

        result = {
            "job_profile_id": job_profile_id,
            "dataset_version": dataset_version,
            "experiment_run_id": experiment_run_id,
            "iteration_number": iteration_number,
            "random_state": current_random_state,
            "dataset_build_result": dataset_build_result,
            "dataset_summary": dataset_summary,
            "train_size": best_model.train_size,
            "test_size": best_model.test_size,
            "selection_criteria": selection_criteria,
            "selected_model": best_model.to_metrics_dict(),
            "active_model_record": active_model_record,
            "all_models": [evaluation.to_metrics_dict() for evaluation in evaluations],
            "message": (
                f"Modelo activo: {best_model.algorithm}. "
                f"Se eligió usando validación cruzada cuando está disponible; "
                f"si no aplica, se usa Recall, F1-Score y duración del holdout."
            ),
        }

        if save_training_report:
            result["training_report"] = self.training_report_service.save_iteration_report(
                training_result=result,
                created_by=created_by,
                persist_to_storage=persist_to_storage,
            )

        return result

    def train_job_profile_iterations(
        self,
        job_profile_id: str,
        iterations: int = 50,
        created_by: str | None = None,
        dataset_version_prefix: str = "experiment",
        persist_to_storage: bool = True,
        auto_build_dataset: bool = True,
        base_random_state: int | None = None,
        start_iteration: int = 1,
    ) -> dict[str, Any]:
        normalized_iterations = self._normalize_iterations(iterations)
        normalized_start = max(1, int(start_iteration or 1))
        base_state = self._normalize_random_state(base_random_state)
        experiment_run_id = f"exp-{job_profile_id[:8]}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"

        iteration_reports = []
        errors = []

        for offset in range(normalized_iterations):
            iteration_number = normalized_start + offset
            random_state = base_state + iteration_number
            dataset_version = f"{dataset_version_prefix}-{experiment_run_id}-it-{iteration_number:03d}"

            try:
                result = self.train_job_profile(
                    job_profile_id=job_profile_id,
                    created_by=created_by,
                    dataset_version=dataset_version,
                    persist_to_storage=persist_to_storage,
                    auto_build_dataset=auto_build_dataset,
                    random_state=random_state,
                    iteration_number=iteration_number,
                    experiment_run_id=experiment_run_id,
                    save_training_report=True,
                )
                iteration_reports.append(result.get("training_report"))
            except Exception as exc:
                errors.append(
                    {
                        "iteration_number": iteration_number,
                        "random_state": random_state,
                        "error": str(exc),
                    }
                )

        aggregate_report = self.training_report_service.build_all_executions_report(
            job_profile_id=job_profile_id,
            persist_to_storage=persist_to_storage,
        )

        return {
            "job_profile_id": job_profile_id,
            "experiment_run_id": experiment_run_id,
            "requested_iterations": normalized_iterations,
            "completed_iterations": len([item for item in iteration_reports if item]),
            "failed_iterations": len(errors),
            "errors": errors,
            "iteration_reports": iteration_reports,
            "aggregate_report": aggregate_report,
        }

    def _build_training_dataset(self, job_profile_id: str) -> list[TrainingDatasetRow]:
        samples = self.dataset_sample_repository.get_rows(
            job_profile_id=job_profile_id,
            ready_only=True,
            labeled_only=True,
        )

        if not samples:
            return []

        candidate_profile_ids = list(
            {
                sample["candidate_profile_id"]
                for sample in samples
                if sample.get("candidate_profile_id")
            }
        )

        candidate_profiles = self.candidate_profile_repository.get_by_ids(candidate_profile_ids)
        candidate_profile_map = {
            profile["id"]: profile
            for profile in candidate_profiles
            if profile.get("id")
        }

        job_profile = self.job_profile_repository.get_by_id(job_profile_id)
        if not job_profile:
            raise NotFoundError(f"Job profile '{job_profile_id}' was not found")

        rows: list[TrainingDatasetRow] = []
        for sample in samples:
            candidate_profile_id = sample.get("candidate_profile_id")
            candidate_profile = candidate_profile_map.get(candidate_profile_id)

            if not candidate_profile:
                continue

            candidate_text = build_candidate_text(candidate_profile)
            job_profile_text = build_job_profile_text(job_profile)
            training_text = build_training_text(job_profile, candidate_profile)

            if not training_text.strip():
                continue

            rows.append(
                TrainingDatasetRow(
                    dataset_sample_id=sample["id"],
                    candidate_profile_id=candidate_profile_id,
                    job_profile_id=job_profile_id,
                    label=bool(sample["label"]),
                    split_name=sample.get("split_name") or "train",
                    source_type=sample.get("source_type") or "unknown",
                    candidate_text=candidate_text,
                    job_profile_text=job_profile_text,
                    training_text=training_text,
                    candidate_profile=candidate_profile,
                    job_profile=job_profile,
                )
            )

        return rows

    def _train_algorithm(
        self,
        algorithm: str,
        estimator,
        vectorizer,
        X_train,
        X_test,
        y_train: list[int],
        y_test: list[int],
        job_profile_id: str,
        created_by: str | None,
        dataset_version: str,
        dataset_summary: dict,
        model_parameters: dict,
        cross_validation: dict,
        persist_to_storage: bool,
    ) -> ModelEvaluation:
        version_tag = self._build_version_tag(algorithm, job_profile_id)

        model_version = self.model_versioning_service.create_model_version(
            {
                "created_by": created_by,
                "job_profile_id": job_profile_id,
                "model_name": f"CV classifier {algorithm}",
                "algorithm": algorithm,
                "vectorizer": "tfidf",
                "version_tag": version_tag,
                "status": "training",
                "dataset_version": dataset_version,
                "train_size": len(y_train),
                "test_size": len(y_test),
                "model_parameters": model_parameters,
                "dataset_summary": dataset_summary,
            },
            created_by=created_by,
            metadata={
                "algorithm": algorithm,
                "dataset_version": dataset_version,
                "train_size": len(y_train),
                "test_size": len(y_test),
                "cross_validation": cross_validation,
            },
        )

        try:
            start = time.perf_counter()
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_test)
            duration_ms = int((time.perf_counter() - start) * 1000)

            metrics = evaluate_predictions(y_test, predictions)
            probabilities = self._extract_positive_probabilities(estimator, X_test, predictions)

            evaluation = ModelEvaluation(
                algorithm=algorithm,
                estimator=estimator,
                vectorizer=vectorizer,
                model_version_id=model_version["id"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                precision=metrics["precision"],
                accuracy=metrics["accuracy"],
                confusion_matrix=metrics["confusion_matrix"],
                labels=metrics["labels"],
                duration_ms=duration_ms,
                predictions=[int(prediction) for prediction in predictions],
                probabilities=[float(probability) for probability in probabilities],
                test_size=len(y_test),
                train_size=len(y_train),
                version_tag=version_tag,
                cross_validation=cross_validation,
            )

            local_artifact_dir = self._build_local_artifact_dir(model_version["id"])
            artifact_files = persist_model_artifacts(
                evaluation=evaluation,
                artifact_dir=local_artifact_dir,
                training_metadata={
                    "job_profile_id": job_profile_id,
                    "dataset_version": dataset_version,
                    "created_by": created_by,
                    "trained_at": datetime.now(UTC).isoformat(),
                    "algorithm": algorithm,
                    "model_parameters": model_parameters,
                    "dataset_summary": dataset_summary,
                    "cross_validation": cross_validation,
                },
            )

            artifact_bucket = None
            artifact_path = None

            if persist_to_storage:
                artifact_bucket = current_app.config["MODEL_ARTIFACT_BUCKET"]
                artifact_path = f"{model_version['id']}/bundle.joblib"

                self.storage_service.upload_file(
                    bucket_name=artifact_bucket,
                    remote_path=artifact_path,
                    local_file_path=artifact_files["bundle"],
                    content_type="application/octet-stream",
                    upsert=True,
                )
                self.storage_service.upload_file(
                    bucket_name=artifact_bucket,
                    remote_path=f"{model_version['id']}/metrics.json",
                    local_file_path=artifact_files["metrics"],
                    content_type="application/json",
                    upsert=True,
                )
                self.storage_service.upload_file(
                    bucket_name=artifact_bucket,
                    remote_path=f"{model_version['id']}/training_metadata.json",
                    local_file_path=artifact_files["metadata"],
                    content_type="application/json",
                    upsert=True,
                )

            evaluation.local_artifact_dir = local_artifact_dir
            evaluation.artifact_bucket = artifact_bucket
            evaluation.artifact_path = artifact_path
            metrics_payload = evaluation.to_metrics_dict()

            self.model_versioning_service.complete_model_version(
                model_version_id=model_version["id"],
                metrics=metrics_payload,
                artifact_bucket=artifact_bucket,
                artifact_path=artifact_path,
                metadata={
                    "algorithm": algorithm,
                    "duration_ms": duration_ms,
                    "metrics": metrics_payload,
                    "cross_validation": cross_validation,
                },
                created_by=created_by,
            )

            self.training_run_repository.create(
                {
                    "model_version_id": model_version["id"],
                    "job_profile_id": job_profile_id,
                    "created_by": created_by,
                    "dataset_version": dataset_version,
                    "train_size": len(y_train),
                    "test_size": len(y_test),
                    "recall": evaluation.recall,
                    "f1_score": evaluation.f1_score,
                    "precision": evaluation.precision,
                    "accuracy": evaluation.accuracy,
                    "duration_ms": duration_ms,
                    "confusion_matrix": evaluation.confusion_matrix,
                    "selected_algorithm": algorithm,
                    "status": "completed",
                    "notes": f"Training pipeline completed for {algorithm}",
                    "model_parameters": model_parameters,
                    "dataset_summary": dataset_summary,
                    "metrics": metrics_payload,
                    "is_selected": False,
                }
            )

            return evaluation

        except Exception as exc:
            self.model_versioning_service.fail_model_version(
                model_version_id=model_version["id"],
                error_message=str(exc),
                created_by=created_by,
            )
            raise

    def _run_cross_validation_if_enabled(
        self,
        texts: list[str],
        labels: list[int],
        estimator,
        vectorizer_params: dict,
        random_state: int,
    ) -> dict:
        if not current_app.config.get("CROSS_VALIDATION_ENABLED", True):
            return {"enabled": False, "reason": "Validación cruzada deshabilitada por configuración"}

        return evaluate_with_cross_validation(
            texts=texts,
            labels=labels,
            estimator=estimator,
            vectorizer_params=vectorizer_params,
            requested_folds=current_app.config["CROSS_VALIDATION_FOLDS"],
            random_state=random_state,
        )

    @staticmethod
    def _can_stratify(labels: list[int]) -> bool:
        counts = Counter(labels)
        return len(counts) >= 2 and all(count >= 2 for count in counts.values())

    @staticmethod
    def _extract_positive_probabilities(estimator, X_test, predictions):
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(X_test)
            if probabilities.shape[1] > 1:
                return probabilities[:, 1].tolist()
            return probabilities[:, 0].tolist()
        return [float(prediction) for prediction in predictions]

    def _select_best_model(self, evaluations: list[ModelEvaluation]) -> ModelEvaluation:
        return sorted(evaluations, key=self._selection_key)[0]

    @staticmethod
    def _selection_key(evaluation: ModelEvaluation):
        cv = evaluation.cross_validation or {}
        if cv.get("enabled"):
            return (
                -float(cv.get("recall_mean") or 0),
                -float(cv.get("f1_score_mean") or 0),
                float(cv.get("f1_score_std") or 999),
                evaluation.duration_ms,
            )
        return (-evaluation.recall, -evaluation.f1_score, evaluation.duration_ms)

    @staticmethod
    def _build_selection_criteria(evaluations: list[ModelEvaluation]) -> str:
        has_valid_cv = any((item.cross_validation or {}).get("enabled") for item in evaluations)
        if has_valid_cv:
            return (
                "Mayor Recall promedio en validación cruzada estratificada; "
                "luego mayor F1-Score promedio; luego menor desviación estándar "
                "del F1-Score; finalmente menor duración."
            )
        return "Mayor Recall en holdout; luego mayor F1-Score; finalmente menor duración."

    @staticmethod
    def _build_version_tag(algorithm: str, job_profile_id: str) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return f"{algorithm}-{job_profile_id[:8]}-{timestamp}"

    def _build_local_artifact_dir(self, model_version_id: str) -> str:
        artifact_root = Path(current_app.config["MODEL_ARTIFACTS_DIR"])
        target_dir = artifact_root / model_version_id
        target_dir.mkdir(parents=True, exist_ok=True)
        return str(target_dir)

    @staticmethod
    def _build_dataset_summary(
        dataset_rows: list[TrainingDatasetRow],
        labels: list[int],
        y_train: list[int],
        y_test: list[int],
        dataset_version: str,
        job_profile_id: str,
        random_state: int,
    ) -> dict:
        class_distribution = Counter(labels)
        train_distribution = Counter(y_train)
        test_distribution = Counter(y_test)

        return {
            "job_profile_id": job_profile_id,
            "dataset_version": dataset_version,
            "total_rows": len(dataset_rows),
            "train_size": len(y_train),
            "test_size": len(y_test),
            "random_state": random_state,
            "class_distribution": {str(key): int(value) for key, value in class_distribution.items()},
            "train_distribution": {str(key): int(value) for key, value in train_distribution.items()},
            "test_distribution": {str(key): int(value) for key, value in test_distribution.items()},
            "dataset_sample_ids": [row.dataset_sample_id for row in dataset_rows if row.dataset_sample_id],
            "generated_at": datetime.now(UTC).isoformat(),
        }

    @staticmethod
    def _build_vectorizer_params() -> dict:
        return {
            "max_features": current_app.config["TFIDF_MAX_FEATURES"],
            "ngram_range": current_app.config["TFIDF_NGRAM_RANGE"],
            "lowercase": True,
            "strip_accents": "unicode",
        }

    @staticmethod
    def _build_model_parameters(algorithm: str, random_state: int) -> dict:
        base_parameters = {
            "algorithm": algorithm,
            "vectorizer": "tfidf",
            "tfidf_max_features": current_app.config["TFIDF_MAX_FEATURES"],
            "tfidf_ngram_range": list(current_app.config["TFIDF_NGRAM_RANGE"]),
            "train_test_size": current_app.config["TRAIN_TEST_SIZE"],
            "random_state": random_state,
            "cross_validation_enabled": current_app.config.get("CROSS_VALIDATION_ENABLED", True),
            "cross_validation_method": "StratifiedKFold",
            "cross_validation_folds": current_app.config["CROSS_VALIDATION_FOLDS"],
        }

        if algorithm == "knn":
            base_parameters.update({"n_neighbors": current_app.config["KNN_DEFAULT_NEIGHBORS"]})

        if algorithm == "decision_tree":
            base_parameters.update(
                {
                    "criterion": "gini",
                    "max_depth": current_app.config["TREE_MAX_DEPTH"],
                    "min_samples_leaf": current_app.config["TREE_MIN_SAMPLES_LEAF"],
                }
            )

        return base_parameters

    @staticmethod
    def _normalize_random_state(random_state: int | None) -> int:
        if random_state is None:
            return int(current_app.config["TRAINING_RANDOM_STATE"])
        return int(random_state)

    @staticmethod
    def _normalize_iterations(iterations: int) -> int:
        try:
            normalized = int(iterations)
        except (TypeError, ValueError):
            raise ValidationError("iterations debe ser un número entero válido")

        if normalized <= 0:
            raise ValidationError("iterations debe ser mayor a 0")

        if normalized > 200:
            raise ValidationError("iterations no puede ser mayor a 200 por ejecución")

        return normalized
