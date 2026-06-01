from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from flask import current_app

from app.errors.exceptions import ValidationError
from app.repositories.auto_training_check_repository import AutoTrainingCheckRepository
from app.repositories.candidate_profile_repository import CandidateProfileRepository
from app.services.model_training_service import ModelTrainingService
from app.services.model_versioning_service import ModelVersioningService


class AutoTrainingService:
    def __init__(self):
        self.candidate_profile_repository = CandidateProfileRepository()
        self.model_versioning_service = ModelVersioningService()
        self.model_training_service = ModelTrainingService()
        self.auto_training_check_repository = AutoTrainingCheckRepository()

    def check_and_train_if_needed(
        self,
        job_profile_id: str,
        created_by: str | None = None,
        enabled: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        threshold = max(
            int(current_app.config["AUTO_RETRAIN_NEW_PROFILES_THRESHOLD"]),
            int(current_app.config["AUTO_RETRAIN_MIN_THRESHOLD"]),
        )

        active_model = self.model_versioning_service.get_active_model(job_profile_id)

        active_model_version_id = active_model.get("id") if active_model else None
        last_activation_date = None

        if active_model:
            last_activation_date = active_model.get("activated_at") or active_model.get(
                "created_at"
            )

        total_valid_profiles = (
            self.candidate_profile_repository.count_valid_profiles_created_after(
                created_after=None,
            )
        )

        new_profiles_count = (
            self.candidate_profile_repository.count_valid_profiles_created_after(
                created_after=last_activation_date,
            )
        )

        should_train = force or not active_model or new_profiles_count >= threshold

        if not enabled and not force:
            payload = {
                "job_profile_id": job_profile_id,
                "active_model_version_id": active_model_version_id,
                "created_by": created_by,
                "threshold": threshold,
                "new_profiles_count": new_profiles_count,
                "total_valid_profiles": total_valid_profiles,
                "should_train": False,
                "training_executed": False,
                "status": "skipped",
                "message": "Auto entrenamiento deshabilitado",
                "metadata": {
                    "enabled": enabled,
                    "force": force,
                },
            }
            saved_check = self.auto_training_check_repository.create(payload)

            return {
                **payload,
                "saved_auto_training_check_id": (
                    saved_check.get("id") if saved_check else None
                ),
                "training_result": None,
            }

        if not should_train:
            payload = {
                "job_profile_id": job_profile_id,
                "active_model_version_id": active_model_version_id,
                "created_by": created_by,
                "threshold": threshold,
                "new_profiles_count": new_profiles_count,
                "total_valid_profiles": total_valid_profiles,
                "should_train": False,
                "training_executed": False,
                "status": "checked",
                "message": "No se requiere reentrenamiento",
                "metadata": {
                    "last_activation_date": last_activation_date,
                    "enabled": enabled,
                    "force": force,
                },
            }
            saved_check = self.auto_training_check_repository.create(payload)

            return {
                **payload,
                "saved_auto_training_check_id": (
                    saved_check.get("id") if saved_check else None
                ),
                "training_result": None,
            }

        if total_valid_profiles < threshold:
            payload = {
                "job_profile_id": job_profile_id,
                "active_model_version_id": active_model_version_id,
                "created_by": created_by,
                "threshold": threshold,
                "new_profiles_count": new_profiles_count,
                "total_valid_profiles": total_valid_profiles,
                "should_train": True,
                "training_executed": False,
                "status": "failed",
                "message": "No hay suficientes perfiles válidos para reentrenar",
                "metadata": {
                    "last_activation_date": last_activation_date,
                    "enabled": enabled,
                    "force": force,
                },
            }

            saved_check = self.auto_training_check_repository.create(payload)

            if not active_model:
                raise ValidationError(
                    "No existe modelo activo y tampoco hay suficientes perfiles "
                    "válidos para entrenar uno nuevo."
                )

            return {
                **payload,
                "saved_auto_training_check_id": (
                    saved_check.get("id") if saved_check else None
                ),
                "training_result": None,
            }

        dataset_version = self._build_dataset_version(job_profile_id)

        try:
            training_result = self.model_training_service.train_job_profile(
                job_profile_id=job_profile_id,
                created_by=created_by,
                dataset_version=dataset_version,
                persist_to_storage=True,
                auto_build_dataset=True,
            )

            selected_model_version_id = (
                training_result.get("selected_model", {}).get("model_version_id")
                if training_result
                else None
            )

            payload = {
                "job_profile_id": job_profile_id,
                "active_model_version_id": active_model_version_id,
                "created_by": created_by,
                "threshold": threshold,
                "new_profiles_count": new_profiles_count,
                "total_valid_profiles": total_valid_profiles,
                "should_train": True,
                "training_executed": True,
                "selected_model_version_id": selected_model_version_id,
                "status": "training_completed",
                "message": "Reentrenamiento automático ejecutado correctamente",
                "metadata": {
                    "dataset_version": dataset_version,
                    "last_activation_date": last_activation_date,
                    "enabled": enabled,
                    "force": force,
                },
            }

            saved_check = self.auto_training_check_repository.create(payload)

            return {
                **payload,
                "saved_auto_training_check_id": (
                    saved_check.get("id") if saved_check else None
                ),
                "training_result": training_result,
            }

        except Exception as exc:
            payload = {
                "job_profile_id": job_profile_id,
                "active_model_version_id": active_model_version_id,
                "created_by": created_by,
                "threshold": threshold,
                "new_profiles_count": new_profiles_count,
                "total_valid_profiles": total_valid_profiles,
                "should_train": True,
                "training_executed": False,
                "status": "failed",
                "message": str(exc),
                "metadata": {
                    "dataset_version": dataset_version,
                    "last_activation_date": last_activation_date,
                    "enabled": enabled,
                    "force": force,
                },
            }

            saved_check = self.auto_training_check_repository.create(payload)

            return {
                **payload,
                "saved_auto_training_check_id": (
                    saved_check.get("id") if saved_check else None
                ),
                "training_result": None,
            }

    @staticmethod
    def _build_dataset_version(job_profile_id: str) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return f"auto-retrain-{job_profile_id[:8]}-{timestamp}"
