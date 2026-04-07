from __future__ import annotations

import time
from datetime import datetime, UTC
from pathlib import Path

from flask import current_app
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from app.machine_learning.common.evaluation import evaluate_predictions
from app.machine_learning.common.schemas import ModelEvaluation
from app.machine_learning.common.serialization import persist_model_artifacts
from app.machine_learning.decision_tree.trainer import DecisionTreeTrainer
from app.machine_learning.knn.trainer import KNNTrainer
from app.repositories.model_version_repository import ModelVersionRepository
from app.repositories.training_run_repository import TrainingRunRepository
from app.services.storage_service import StorageService
from app.services.training_dataset_service import TrainingDatasetService


class ModelTrainingService:
    def __init__(self):
        self.dataset_service = TrainingDatasetService()
        self.model_version_repository = ModelVersionRepository()
        self.training_run_repository = TrainingRunRepository()
        self.storage_service = StorageService()

    def train_job_profile(
        self,
        job_profile_id: str,
        created_by: str | None = None,
        dataset_version: str | None = None,
        persist_to_storage: bool = True,
    ) -> dict:
        dataframe = self.dataset_service.build_training_dataframe(job_profile_id=job_profile_id)

        if dataframe.empty:
            raise ValueError("No existen registros etiquetados en dataset_sample para entrenar")

        if len(dataframe) < current_app.config["MIN_TRAINING_ROWS"]:
            raise ValueError(
                "No hay suficientes registros etiquetados para entrenar. "
                f"Mínimo requerido: {current_app.config['MIN_TRAINING_ROWS']}"
            )

        label_values = dataframe["label"].astype(int).tolist()
        if len(set(label_values)) < 2:
            raise ValueError("Se requieren al menos dos clases para entrenar el modelo")

        X_train, X_test, y_train, y_test = self._split_dataset(dataframe)

        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=current_app.config["TFIDF_MAX_FEATURES"],
            ngram_range=current_app.config["TFIDF_NGRAM_RANGE"],
        )
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        evaluations: list[ModelEvaluation] = []

        knn_trainer = KNNTrainer(current_app.config["KNN_DEFAULT_NEIGHBORS"])
        knn_estimator = knn_trainer.build_estimator(sample_size=len(y_train))
        evaluations.append(
            self._train_single_model(
                algorithm="knn",
                estimator=knn_estimator,
                vectorizer=vectorizer,
                X_train=X_train_vectorized,
                X_test=X_test_vectorized,
                y_train=y_train,
                y_test=y_test,
                job_profile_id=job_profile_id,
                created_by=created_by,
                dataset_version=dataset_version,
                persist_to_storage=persist_to_storage,
            )
        )

        tree_trainer = DecisionTreeTrainer(
            max_depth=current_app.config["TREE_MAX_DEPTH"],
            min_samples_leaf=current_app.config["TREE_MIN_SAMPLES_LEAF"],
            random_state=current_app.config["TRAINING_RANDOM_STATE"],
        )
        tree_estimator = tree_trainer.build_estimator()
        evaluations.append(
            self._train_single_model(
                algorithm="decision_tree",
                estimator=tree_estimator,
                vectorizer=vectorizer,
                X_train=X_train_vectorized,
                X_test=X_test_vectorized,
                y_train=y_train,
                y_test=y_test,
                job_profile_id=job_profile_id,
                created_by=created_by,
                dataset_version=dataset_version,
                persist_to_storage=persist_to_storage,
            )
        )

        best_model = self._select_best_model(evaluations)

        for evaluation in evaluations:
            status = "active" if evaluation.model_version_id == best_model.model_version_id else "archived"
            self.model_version_repository.update(
                evaluation.model_version_id,
                {"status": status},
            )

        return {
            "success": True,
            "job_profile_id": job_profile_id,
            "dataset_summary": self.dataset_service.summarize_dataframe(dataframe),
            "best_model": best_model.to_metrics_dict(),
            "models": [evaluation.to_metrics_dict() for evaluation in evaluations],
        }

    def _split_dataset(self, dataframe):
        has_explicit_test = (dataframe["split_name"] == "test").any()
        has_explicit_train = (dataframe["split_name"] == "train").any()

        if has_explicit_train and has_explicit_test:
            train_df = dataframe[dataframe["split_name"] == "train"].copy()
            test_df = dataframe[dataframe["split_name"] == "test"].copy()

            return (
                train_df["training_text"].tolist(),
                test_df["training_text"].tolist(),
                train_df["label"].astype(int).tolist(),
                test_df["label"].astype(int).tolist(),
            )

        try:
            return train_test_split(
                dataframe["training_text"].tolist(),
                dataframe["label"].astype(int).tolist(),
                test_size=current_app.config["TRAIN_TEST_SIZE"],
                random_state=current_app.config["TRAINING_RANDOM_STATE"],
                stratify=dataframe["label"].astype(int).tolist(),
            )
        except ValueError:
            return train_test_split(
                dataframe["training_text"].tolist(),
                dataframe["label"].astype(int).tolist(),
                test_size=current_app.config["TRAIN_TEST_SIZE"],
                random_state=current_app.config["TRAINING_RANDOM_STATE"],
            )

    def _train_single_model(
        self,
        algorithm: str,
        estimator,
        vectorizer,
        X_train,
        X_test,
        y_train,
        y_test,
        job_profile_id: str,
        created_by: str | None,
        dataset_version: str | None,
        persist_to_storage: bool,
    ) -> ModelEvaluation:
        started_at = time.perf_counter()
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test).tolist()
        duration_ms = int((time.perf_counter() - started_at) * 1000)

        probabilities = self._extract_positive_probabilities(estimator, X_test, predictions)
        metrics = evaluate_predictions(y_test, predictions)
        version_tag = self._build_version_tag(algorithm, job_profile_id)

        model_version = self.model_version_repository.create(
            {
                "created_by": created_by,
                "model_name": f"{algorithm}-cv-classifier",
                "algorithm": algorithm,
                "vectorizer": "tfidf",
                "version_tag": version_tag,
                "status": "draft",
                "metrics": metrics,
            }
        )
        if not model_version:
            raise RuntimeError(f"No se pudo registrar la versión del modelo {algorithm}")

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
            predictions=predictions,
            probabilities=probabilities,
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

        self.model_version_repository.update(
            model_version["id"],
            {
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
                "duration_ms": duration_ms,
                "notes": f"Base training pipeline for {algorithm}",
            }
        )

        evaluation.local_artifact_dir = local_artifact_dir
        evaluation.artifact_bucket = artifact_bucket
        evaluation.artifact_path = artifact_path

        return evaluation

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
