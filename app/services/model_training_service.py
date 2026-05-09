from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import current_app
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from app.errors.exceptions import NotFoundError, ValidationError
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
from app.repositories.model_version_repository import ModelVersionRepository
from app.repositories.training_run_repository import TrainingRunRepository
from app.services.dataset_builder_service import DatasetBuilderService
from app.services.storage_service import StorageService


class ModelTrainingService:
    def __init__(self):
        self.dataset_sample_repository = DatasetSampleRepository()
        self.candidate_profile_repository = CandidateProfileRepository()
        self.job_profile_repository = JobProfileRepository()
        self.model_version_repository = ModelVersionRepository()
        self.training_run_repository = TrainingRunRepository()
        self.dataset_builder_service = DatasetBuilderService()
        self.storage_service = StorageService()

    def train_job_profile(
        self,
        job_profile_id: str,
        created_by: str | None = None,
        dataset_version: str = "api-training-run",
        persist_to_storage: bool = True,
        auto_build_dataset: bool = True,
    ) -> dict[str, Any]:
        job_profile = self.job_profile_repository.get_by_id(job_profile_id)

        if not job_profile:
            raise NotFoundError(f"Job profile '{job_profile_id}' was not found")

        dataset_build_result = None

        if auto_build_dataset and current_app.config["AUTO_BUILD_DATASET_ON_TRAIN"]:
            dataset_build_result = (
                self.dataset_builder_service.ensure_dataset_for_training(
                    job_profile_id=job_profile_id,
                    created_by=created_by,
                    dataset_version=f"{dataset_version}-auto-dataset",
                )
            )

        dataset_rows = self._build_training_dataset(job_profile_id)

        min_rows = current_app.config["MIN_TRAINING_ROWS"]
        if len(dataset_rows) < min_rows:
            raise ValidationError(
                f"No hay suficientes registros listos en dataset_sample para entrenar. "
                f"Actual: {len(dataset_rows)}. Mínimo requerido: {min_rows}. "
                f"Primero construye dataset_sample desde candidate_profile."
            )

        labels = [1 if row.label else 0 for row in dataset_rows]
        unique_labels = sorted(set(labels))

        if len(unique_labels) < 2:
            raise ValidationError(
                "El dataset solo tiene una clase. Para entrenar se necesitan ejemplos "
                "positivos y negativos. Revisa las etiquetas en dataset_sample."
            )

        texts = [row.training_text for row in dataset_rows]

        vectorizer = TfidfVectorizer(
            max_features=current_app.config["TFIDF_MAX_FEATURES"],
            ngram_range=current_app.config["TFIDF_NGRAM_RANGE"],
            lowercase=True,
            strip_accents="unicode",
        )

        stratify = labels if self._can_stratify(labels) else None

        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=current_app.config["TRAIN_TEST_SIZE"],
            random_state=current_app.config["TRAINING_RANDOM_STATE"],
            stratify=stratify,
        )

        X_train = vectorizer.fit_transform(X_train_texts)
        X_test = vectorizer.transform(X_test_texts)

        evaluations = [
            self._train_algorithm(
                algorithm="knn",
                estimator=KNNTrainer(
                    n_neighbors=current_app.config["KNN_DEFAULT_NEIGHBORS"]
                ).build_estimator(sample_size=len(y_train)),
                vectorizer=vectorizer,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                job_profile_id=job_profile_id,
                created_by=created_by,
                dataset_version=dataset_version,
                persist_to_storage=persist_to_storage,
            ),
            self._train_algorithm(
                algorithm="decision_tree",
                estimator=DecisionTreeTrainer(
                    max_depth=current_app.config["TREE_MAX_DEPTH"],
                    min_samples_leaf=current_app.config["TREE_MIN_SAMPLES_LEAF"],
                    random_state=current_app.config["TRAINING_RANDOM_STATE"],
                ).build_estimator(),
                vectorizer=vectorizer,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                job_profile_id=job_profile_id,
                created_by=created_by,
                dataset_version=dataset_version,
                persist_to_storage=persist_to_storage,
            ),
        ]

        best_model = self._select_best_model(evaluations)

        return {
            "job_profile_id": job_profile_id,
            "dataset_version": dataset_version,
            "dataset_build_result": dataset_build_result,
            "train_size": best_model.train_size,
            "test_size": best_model.test_size,
            "selected_model": best_model.to_metrics_dict(),
            "all_models": [evaluation.to_metrics_dict() for evaluation in evaluations],
            "message": (
                f"Modelo seleccionado: {best_model.algorithm}. "
                f"Se eligió priorizando Recall, luego F1-Score y finalmente duración."
            ),
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

        candidate_profiles = self.candidate_profile_repository.get_by_ids(
            candidate_profile_ids
        )
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
        persist_to_storage: bool,
    ) -> ModelEvaluation:
        version_tag = self._build_version_tag(algorithm, job_profile_id)

        model_version = self.model_version_repository.create(
            {
                "created_by": created_by,
                "model_name": f"CV classifier {algorithm}",
                "algorithm": algorithm,
                "version_tag": version_tag,
                "status": "training",
            }
        )

        if not model_version:
            raise ValidationError(f"Could not create model_version for {algorithm}")

        start = time.perf_counter()

        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)

        duration_ms = int((time.perf_counter() - start) * 1000)

        metrics = evaluate_predictions(y_test, predictions)
        probabilities = self._extract_positive_probabilities(
            estimator, X_test, predictions
        )

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

        self.model_version_repository.update(
            model_version["id"],
            {
                "status": "completed",
                "artifact_bucket": artifact_bucket,
                "artifact_path": artifact_path,
                "metrics": evaluation.to_metrics_dict(),
            },
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
            }
        )

        return evaluation

    @staticmethod
    def _can_stratify(labels: list[int]) -> bool:
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

        return len(counts) >= 2 and all(count >= 2 for count in counts.values())

    @staticmethod
    def _extract_positive_probabilities(estimator, X_test, predictions):
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(X_test)

            if probabilities.shape[1] > 1:
                return probabilities[:, 1].tolist()

            return probabilities[:, 0].tolist()

        return [float(prediction) for prediction in predictions]

    @staticmethod
    def _select_best_model(evaluations: list[ModelEvaluation]) -> ModelEvaluation:
        return sorted(
            evaluations,
            key=lambda item: (-item.recall, -item.f1_score, item.duration_ms),
        )[0]

    @staticmethod
    def _build_version_tag(algorithm: str, job_profile_id: str) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return f"{algorithm}-{job_profile_id[:8]}-{timestamp}"

    def _build_local_artifact_dir(self, model_version_id: str) -> str:
        artifact_root = Path(current_app.config["MODEL_ARTIFACTS_DIR"])
        target_dir = artifact_root / model_version_id
        target_dir.mkdir(parents=True, exist_ok=True)
        return str(target_dir)
