from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.errors.exceptions import NotFoundError, ValidationError
from app.machine_learning.common.utils import build_training_text
from app.repositories.candidate_profile_repository import CandidateProfileRepository
from app.repositories.classification_result_repository import (
    ClassificationResultRepository,
)
from app.repositories.job_profile_repository import JobProfileRepository
from app.repositories.processing_run_repository import ProcessingRunRepository
from app.services.auto_training_service import AutoTrainingService
from app.services.model_artifact_loader_service import ModelArtifactLoaderService
from app.services.model_versioning_service import ModelVersioningService
from sklearn.metrics.pairwise import cosine_similarity

from app.machine_learning.common.utils import (
    build_candidate_text,
    build_job_profile_text,
    build_training_text,
)


class ModelInferenceService:
    def __init__(self):
        self.job_profile_repository = JobProfileRepository()
        self.candidate_profile_repository = CandidateProfileRepository()
        self.processing_run_repository = ProcessingRunRepository()
        self.classification_result_repository = ClassificationResultRepository()
        self.model_versioning_service = ModelVersioningService()
        self.model_artifact_loader_service = ModelArtifactLoaderService()
        self.auto_training_service = AutoTrainingService()

    def classify_candidate_profile(
        self,
        job_profile_id: str,
        candidate_profile_id: str,
        created_by: str | None = None,
        auto_retrain: bool = True,
        persist_result: bool = True,
    ) -> dict[str, Any]:
        result = self.classify_candidate_profiles(
            job_profile_id=job_profile_id,
            candidate_profile_ids=[candidate_profile_id],
            created_by=created_by,
            auto_retrain=auto_retrain,
            persist_result=persist_result,
        )

        classification = result["results"][0] if result["results"] else None

        return {
            **result,
            "classification": classification,
        }

    def classify_candidate_profiles(
        self,
        job_profile_id: str,
        candidate_profile_ids: list[str],
        created_by: str | None = None,
        auto_retrain: bool = True,
        persist_result: bool = True,
    ) -> dict[str, Any]:
        if not candidate_profile_ids:
            raise ValidationError("candidate_profile_ids is required")

        job_profile = self.job_profile_repository.get_by_id(job_profile_id)

        if not job_profile:
            raise NotFoundError(f"Job profile '{job_profile_id}' was not found")

        auto_training_result = self.auto_training_service.check_and_train_if_needed(
            job_profile_id=job_profile_id,
            created_by=created_by,
            enabled=auto_retrain,
            force=False,
        )

        active_model = self.model_versioning_service.get_active_model(job_profile_id)

        if not active_model:
            raise ValidationError(
                "No existe un modelo activo para este perfil de puesto"
            )

        bundle = self.model_artifact_loader_service.load_bundle(active_model)
        estimator = bundle["estimator"]
        vectorizer = bundle["vectorizer"]

        processing_run = None

        if persist_result:
            processing_run = self.processing_run_repository.create(
                {
                    "job_profile_id": job_profile_id,
                    "model_version_id": active_model["id"],
                    "created_by": created_by,
                    "input_type": "inference",
                    "status": "running",
                    "total_candidates": len(candidate_profile_ids),
                    "started_at": datetime.now(UTC).isoformat(),
                    "artifact_bucket": active_model.get("artifact_bucket"),
                    "artifact_path": active_model.get("artifact_path"),
                }
            )

        try:
            raw_results = []

            candidate_profiles = self.candidate_profile_repository.get_by_ids(
                candidate_profile_ids
            )

            candidate_profile_map = {
                profile["id"]: profile
                for profile in candidate_profiles
                if profile.get("id")
            }

            for candidate_profile_id in candidate_profile_ids:
                candidate_profile = candidate_profile_map.get(candidate_profile_id)

                if not candidate_profile:
                    raw_results.append(
                        {
                            "candidate_profile_id": candidate_profile_id,
                            "error": "candidate_profile not found",
                            "predicted_label": None,
                            "score_0_100": 0,
                        }
                    )
                    continue

                inference_text = build_training_text(job_profile, candidate_profile)

                if not inference_text.strip():
                    raw_results.append(
                        {
                            "candidate_profile_id": candidate_profile_id,
                            "error": "candidate profile has empty text",
                            "predicted_label": None,
                            "score_0_100": 0,
                        }
                    )
                    continue

                X = vectorizer.transform([inference_text])
                prediction = estimator.predict(X)[0]
                score_detail = self._calculate_score_detail(
                    estimator=estimator,
                    X=X,
                    prediction=prediction,
                    vectorizer=vectorizer,
                    job_profile=job_profile,
                    candidate_profile=candidate_profile,
                )

                score = score_detail["final_score_0_100"]

                predicted_label = bool(int(prediction))

                raw_results.append(
                    {
                        "candidate_profile_id": candidate_profile_id,
                        "predicted_label": predicted_label,
                        "score_0_100": score,
                        "score_detail": score_detail,
                        "model_version_id": active_model["id"],
                        "algorithm": active_model.get("algorithm"),
                        "version_tag": active_model.get("version_tag"),
                        "inference_text_length": len(inference_text),
                        "error": None,
                    }
                )

            sorted_results = sorted(
                raw_results,
                key=lambda item: item.get("score_0_100") or 0,
                reverse=True,
            )

            persisted_results = []

            for index, item in enumerate(sorted_results, start=1):
                item["rank_position"] = index

                if persist_result and processing_run and not item.get("error"):
                    persisted = self.classification_result_repository.create(
                        {
                            "processing_run_id": processing_run["id"],
                            "candidate_profile_id": item["candidate_profile_id"],
                            "model_version_id": active_model["id"],
                            "created_by": created_by,
                            "predicted_label": item["predicted_label"],
                            "score_0_100": item["score_0_100"],
                            "rank_position": index,
                            "match_summary": {
                                "job_profile_id": job_profile_id,
                                "job_profile_title": job_profile.get("title"),
                                "algorithm": active_model.get("algorithm"),
                                "version_tag": active_model.get("version_tag"),
                                "model_status": active_model.get("status"),
                                "inference_text_length": item.get(
                                    "inference_text_length"
                                ),
                                "auto_training_check": {
                                    "status": auto_training_result.get("status"),
                                    "new_profiles_count": auto_training_result.get(
                                        "new_profiles_count"
                                    ),
                                    "threshold": auto_training_result.get("threshold"),
                                    "training_executed": auto_training_result.get(
                                        "training_executed"
                                    ),
                                },
                                "score_detail": item.get("score_detail"),
                            },
                        }
                    )

                    item["classification_result_id"] = (
                        persisted.get("id") if persisted else None
                    )
                    persisted_results.append(persisted)

            if persist_result and processing_run:
                self.processing_run_repository.update(
                    processing_run["id"],
                    {
                        "status": "completed",
                        "finished_at": datetime.now(UTC).isoformat(),
                        "total_candidates": len(candidate_profile_ids),
                    },
                )

            return {
                "job_profile_id": job_profile_id,
                "job_profile_title": job_profile.get("title"),
                "active_model": {
                    "id": active_model.get("id"),
                    "algorithm": active_model.get("algorithm"),
                    "version_tag": active_model.get("version_tag"),
                    "status": active_model.get("status"),
                    "artifact_bucket": active_model.get("artifact_bucket"),
                    "artifact_path": active_model.get("artifact_path"),
                },
                "processing_run_id": (
                    processing_run.get("id") if processing_run else None
                ),
                "auto_training_result": auto_training_result,
                "total_candidates": len(candidate_profile_ids),
                "results": sorted_results,
                "persisted_results": persisted_results,
            }

        except Exception as exc:
            if persist_result and processing_run:
                self.processing_run_repository.update(
                    processing_run["id"],
                    {
                        "status": "failed",
                        "finished_at": datetime.now(UTC).isoformat(),
                        "error_message": str(exc),
                    },
                )
            raise

    @staticmethod
    def _calculate_score_detail(
        estimator,
        X,
        prediction,
        vectorizer,
        job_profile: dict,
        candidate_profile: dict,
    ) -> dict:
        model_score = 0.0

        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(X)[0]
            classes = list(estimator.classes_)

            positive_index = None

            for index, class_value in enumerate(classes):
                if int(class_value) == 1:
                    positive_index = index
                    break

            if positive_index is not None:
                model_score = round(float(probabilities[positive_index]) * 100, 2)
        else:
            model_score = 100.0 if bool(int(prediction)) else 0.0

        job_text = build_job_profile_text(job_profile)
        candidate_text = build_candidate_text(candidate_profile)

        text_similarity_score = 0.0

        if job_text.strip() and candidate_text.strip():
            job_vector = vectorizer.transform([job_text])
            candidate_vector = vectorizer.transform([candidate_text])

            similarity = cosine_similarity(job_vector, candidate_vector)[0][0]
            text_similarity_score = round(float(similarity) * 100, 2)

        final_score = round((model_score * 0.70) + (text_similarity_score * 0.30), 2)

        return {
            "model_score_0_100": model_score,
            "text_similarity_score_0_100": text_similarity_score,
            "final_score_0_100": final_score,
            "score_formula": "70% modelo + 30% similitud textual",
        }
