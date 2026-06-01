from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from app.errors.exceptions import NotFoundError, ValidationError
from app.machine_learning.common.utils import (
    build_candidate_text,
    build_job_profile_text,
    build_training_text,
)
from app.repositories.candidate_profile_repository import CandidateProfileRepository
from app.repositories.classification_result_repository import (
    ClassificationResultRepository,
)
from app.repositories.job_profile_repository import JobProfileRepository
from app.repositories.processing_run_repository import ProcessingRunRepository
from app.services.auto_training_service import AutoTrainingService
from app.services.model_artifact_loader_service import ModelArtifactLoaderService
from app.services.model_versioning_service import ModelVersioningService
from app.services.ranking_service import RankingService


class ModelInferenceService:
    def __init__(self):
        self.job_profile_repository = JobProfileRepository()
        self.candidate_profile_repository = CandidateProfileRepository()
        self.processing_run_repository = ProcessingRunRepository()
        self.classification_result_repository = ClassificationResultRepository()
        self.model_versioning_service = ModelVersioningService()
        self.model_artifact_loader_service = ModelArtifactLoaderService()
        self.auto_training_service = AutoTrainingService()
        self.ranking_service = RankingService()

    def classify_candidate_profile(
        self,
        job_profile_id: str,
        candidate_profile_id: str,
        created_by: str | None = None,
        auto_retrain: bool = True,
        persist_result: bool = True,
        top_k: int | str | None = None,
    ) -> dict[str, Any]:
        result = self.classify_candidate_profiles(
            job_profile_id=job_profile_id,
            candidate_profile_ids=[candidate_profile_id],
            created_by=created_by,
            auto_retrain=auto_retrain,
            persist_result=persist_result,
            top_k=top_k,
        )

        classification = result["results"][0] if result["results"] else None

        return {
            **result,
            "classification": classification,
        }

    def rank_all_candidate_profiles(
        self,
        job_profile_id: str,
        created_by: str | None = None,
        auto_retrain: bool = True,
        persist_result: bool = True,
        top_k: int | str | None = None,
        max_candidates: int | str | None = 1000,
        reuse_existing: bool = True,
        force_new_run: bool = False,
    ) -> dict[str, Any]:
        normalized_top_k = self.ranking_service.normalize_top_k(top_k)

        normalized_limit = self._normalize_candidate_limit(
            max_candidates=max_candidates,
            minimum_required=normalized_top_k,
        )

        job_profile = self.job_profile_repository.get_by_id(job_profile_id)

        if not job_profile:
            raise NotFoundError(f"Job profile '{job_profile_id}' was not found")

        active_model = self.model_versioning_service.get_active_model(job_profile_id)

        if not active_model:
            raise ValidationError(
                "No existe un modelo activo para este perfil de puesto"
            )

        candidate_profiles = self.candidate_profile_repository.list_valid_for_inference(
            limit=normalized_limit,
        )

        candidate_profile_ids = [
            profile["id"] for profile in candidate_profiles if profile.get("id")
        ]

        if not candidate_profile_ids:
            raise ValidationError("No existen candidatos válidos para generar ranking")

        candidate_profile_ids_hash = self._build_candidate_ids_hash(
            candidate_profile_ids
        )

        can_reuse = (
            reuse_existing is True
            and force_new_run is False
            and auto_retrain is False
            and persist_result is True
        )

        if can_reuse:
            reusable_run = self._find_reusable_rank_all_run(
                job_profile_id=job_profile_id,
                model_version_id=active_model["id"],
                candidate_profile_ids_hash=candidate_profile_ids_hash,
                total_candidates=len(candidate_profile_ids),
            )

            if reusable_run:
                results = self.classification_result_repository.list_by_processing_run(
                    processing_run_id=reusable_run["id"],
                    limit=normalized_top_k,
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
                    "processing_run_id": reusable_run["id"],
                    "reused_existing_run": True,
                    "candidate_source_mode": "all_valid_candidate_profiles",
                    "max_candidates_requested": normalized_limit,
                    "candidate_profiles_found": len(candidate_profile_ids),
                    "total_candidates": reusable_run.get("total_candidates"),
                    "total_ranked": reusable_run.get("total_candidates"),
                    "top_k": normalized_top_k,
                    "returned_results": len(results),
                    "results": results,
                    "message": "Ranking reutilizado desde una ejecución previa",
                }

        run_metadata = {
            "candidate_source_mode": "all_valid_candidate_profiles",
            "max_candidates_requested": normalized_limit,
            "candidate_profiles_found": len(candidate_profile_ids),
            "candidate_profile_ids_hash": candidate_profile_ids_hash,
        }

        result = self.classify_candidate_profiles(
            job_profile_id=job_profile_id,
            candidate_profile_ids=candidate_profile_ids,
            created_by=created_by,
            auto_retrain=auto_retrain,
            persist_result=persist_result,
            top_k=normalized_top_k,
            input_type="rank_all",
            run_metadata=run_metadata,
        )

        result["candidate_source_mode"] = "all_valid_candidate_profiles"
        result["max_candidates_requested"] = normalized_limit
        result["candidate_profiles_found"] = len(candidate_profile_ids)
        result["reused_existing_run"] = False

        return result

    def classify_candidate_profiles(
        self,
        job_profile_id: str,
        candidate_profile_ids: list[str],
        created_by: str | None = None,
        auto_retrain: bool = True,
        persist_result: bool = True,
        top_k: int | str | None = None,
        input_type: str = "inference",
        run_metadata: dict | None = None,
    ) -> dict[str, Any]:
        if not candidate_profile_ids:
            raise ValidationError("candidate_profile_ids is required")

        normalized_top_k = self.ranking_service.normalize_top_k(top_k)

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

        metadata = dict(run_metadata or {})

        initial_trace_summary = {
            "requested_top_k": normalized_top_k,
            "ranking_mode": "top_k" if normalized_top_k else "full",
            **metadata,
        }

        processing_run = None

        if persist_result:
            processing_run = self.processing_run_repository.create(
                {
                    "job_profile_id": job_profile_id,
                    "model_version_id": active_model["id"],
                    "created_by": created_by,
                    "input_type": input_type,
                    "status": "running",
                    "total_candidates": len(candidate_profile_ids),
                    "started_at": datetime.now(UTC).isoformat(),
                    "artifact_bucket": active_model.get("artifact_bucket"),
                    "artifact_path": active_model.get("artifact_path"),
                    "trace_summary": initial_trace_summary,
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

                raw_score = score_detail["final_score_0_100"]
                predicted_label = bool(int(prediction))

                raw_results.append(
                    {
                        "candidate_profile_id": candidate_profile_id,
                        "predicted_label": predicted_label,
                        "raw_score_0_100": raw_score,
                        "score_0_100": raw_score,
                        "score_detail": score_detail,
                        "model_version_id": active_model["id"],
                        "algorithm": active_model.get("algorithm"),
                        "version_tag": active_model.get("version_tag"),
                        "inference_text_length": len(inference_text),
                        "error": None,
                    }
                )

            ranked_results = self.ranking_service.generate_ranking(raw_results)

            top_results = self.ranking_service.limit_ranking(
                ranked_results=ranked_results,
                top_k=normalized_top_k,
            )

            persisted_results = []

            for item in ranked_results:
                if persist_result and processing_run and not item.get("error"):
                    persisted = self.classification_result_repository.create(
                        {
                            "processing_run_id": processing_run["id"],
                            "candidate_profile_id": item["candidate_profile_id"],
                            "model_version_id": active_model["id"],
                            "created_by": created_by,
                            "predicted_label": item["predicted_label"],
                            "score_0_100": item["score_0_100"],
                            "rank_position": item["rank_position"],
                            "match_summary": {
                                "job_profile_id": job_profile_id,
                                "job_profile_title": job_profile.get("title"),
                                "algorithm": active_model.get("algorithm"),
                                "version_tag": active_model.get("version_tag"),
                                "model_status": active_model.get("status"),
                                "inference_text_length": item.get(
                                    "inference_text_length"
                                ),
                                "requested_top_k": normalized_top_k,
                                "score_detail": item.get("score_detail"),
                                "ranking_detail": item.get("ranking_detail"),
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
                            },
                        }
                    )

                    item["classification_result_id"] = (
                        persisted.get("id") if persisted else None
                    )

                    if persisted:
                        persisted_results.append(persisted)

            successful_candidates = len(
                [item for item in ranked_results if not item.get("error")]
            )
            failed_candidates = len(
                [item for item in ranked_results if item.get("error")]
            )

            scores = [
                float(item.get("score_0_100") or 0)
                for item in ranked_results
                if not item.get("error")
            ]

            trace_summary = {
                "requested_top_k": normalized_top_k,
                "ranking_mode": "top_k" if normalized_top_k else "full",
                "total_candidates": len(candidate_profile_ids),
                "successful_candidates": successful_candidates,
                "failed_candidates": failed_candidates,
                "max_score_0_100": max(scores) if scores else 0,
                "min_score_0_100": min(scores) if scores else 0,
                "average_score_0_100": (
                    round(sum(scores) / len(scores), 2) if scores else 0
                ),
                "active_model": {
                    "id": active_model.get("id"),
                    "algorithm": active_model.get("algorithm"),
                    "version_tag": active_model.get("version_tag"),
                    "status": active_model.get("status"),
                },
                **metadata,
            }

            if persist_result and processing_run:
                self.processing_run_repository.update(
                    processing_run["id"],
                    {
                        "status": "completed",
                        "finished_at": datetime.now(UTC).isoformat(),
                        "total_candidates": len(candidate_profile_ids),
                        "trace_summary": trace_summary,
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
                "reused_existing_run": False,
                "auto_training_result": auto_training_result,
                "total_candidates": len(candidate_profile_ids),
                "total_ranked": len(ranked_results),
                "top_k": normalized_top_k,
                "returned_results": len(top_results),
                "results": top_results,
                "persisted_results_count": len(persisted_results),
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

    def get_ranking_by_processing_run(
        self,
        processing_run_id: str,
        limit: int | str | None = None,
    ) -> dict[str, Any]:
        normalized_limit = self.ranking_service.normalize_top_k(limit)

        processing_run = self.processing_run_repository.get_by_id(processing_run_id)

        if not processing_run:
            raise NotFoundError(f"Processing run '{processing_run_id}' was not found")

        results = self.classification_result_repository.list_by_processing_run(
            processing_run_id=processing_run_id,
            limit=normalized_limit,
        )

        return {
            "processing_run_id": processing_run_id,
            "job_profile_id": processing_run.get("job_profile_id"),
            "model_version_id": processing_run.get("model_version_id"),
            "status": processing_run.get("status"),
            "total_candidates": processing_run.get("total_candidates"),
            "limit": normalized_limit,
            "returned_results": len(results),
            "results": results,
        }

    def _find_reusable_rank_all_run(
        self,
        job_profile_id: str,
        model_version_id: str,
        candidate_profile_ids_hash: str,
        total_candidates: int,
    ) -> dict | None:
        runs = self.processing_run_repository.list_completed_rank_all(
            job_profile_id=job_profile_id,
            model_version_id=model_version_id,
            limit=20,
        )

        for run in runs:
            trace_summary = run.get("trace_summary") or {}

            same_candidates = (
                trace_summary.get("candidate_profile_ids_hash")
                == candidate_profile_ids_hash
            )

            same_total = int(run.get("total_candidates") or 0) == total_candidates

            if same_candidates and same_total:
                return run

        return None

    @staticmethod
    def _build_candidate_ids_hash(candidate_profile_ids: list[str]) -> str:
        normalized_ids = sorted(candidate_profile_ids)
        joined_ids = "|".join(normalized_ids)
        return hashlib.sha256(joined_ids.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_candidate_limit(
        max_candidates: int | str | None,
        minimum_required: int | None = None,
    ) -> int:
        try:
            normalized_limit = int(max_candidates or 1000)
        except (TypeError, ValueError):
            raise ValidationError("max_candidates debe ser un número entero válido")

        if normalized_limit <= 0:
            raise ValidationError("max_candidates debe ser mayor a 0")

        if minimum_required and normalized_limit < minimum_required:
            normalized_limit = minimum_required

        if normalized_limit > 5000:
            normalized_limit = 5000

        return normalized_limit

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
