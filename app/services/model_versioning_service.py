from __future__ import annotations

from typing import Any

from app.errors.exceptions import NotFoundError, ValidationError
from app.repositories.model_version_event_repository import ModelVersionEventRepository
from app.repositories.model_version_repository import ModelVersionRepository
from app.repositories.training_run_repository import TrainingRunRepository


class ModelVersioningService:
    def __init__(self):
        self.model_version_repository = ModelVersionRepository()
        self.training_run_repository = TrainingRunRepository()
        self.model_version_event_repository = ModelVersionEventRepository()

    def create_model_version(
        self,
        payload: dict,
        created_by: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        model_version = self.model_version_repository.create(payload)

        if not model_version:
            raise ValidationError("No se pudo crear model_version")

        self.model_version_event_repository.create(
            {
                "model_version_id": model_version["id"],
                "job_profile_id": model_version.get("job_profile_id"),
                "created_by": created_by,
                "event_type": "created",
                "previous_status": None,
                "new_status": model_version.get("status"),
                "message": "Versión de modelo creada para entrenamiento",
                "metadata": metadata or {},
            }
        )

        return model_version

    def complete_model_version(
        self,
        model_version_id: str,
        metrics: dict,
        artifact_bucket: str | None,
        artifact_path: str | None,
        metadata: dict | None = None,
        created_by: str | None = None,
    ) -> dict:
        current = self.model_version_repository.get_by_id(model_version_id)

        if not current:
            raise NotFoundError(f"Model version '{model_version_id}' was not found")

        updated = self.model_version_repository.update(
            model_version_id,
            {
                "status": "completed",
                "metrics": metrics,
                "artifact_bucket": artifact_bucket,
                "artifact_path": artifact_path,
            },
        )

        self.model_version_event_repository.create(
            {
                "model_version_id": model_version_id,
                "job_profile_id": current.get("job_profile_id"),
                "created_by": created_by,
                "event_type": "training_completed",
                "previous_status": current.get("status"),
                "new_status": "completed",
                "message": "Entrenamiento completado y artefactos generados",
                "metadata": metadata or {},
            }
        )

        return updated

    def fail_model_version(
        self,
        model_version_id: str,
        error_message: str,
        created_by: str | None = None,
    ) -> dict | None:
        current = self.model_version_repository.get_by_id(model_version_id)

        if not current:
            return None

        updated = self.model_version_repository.update(
            model_version_id,
            {
                "status": "failed",
                "selection_reason": error_message,
            },
        )

        self.model_version_event_repository.create(
            {
                "model_version_id": model_version_id,
                "job_profile_id": current.get("job_profile_id"),
                "created_by": created_by,
                "event_type": "training_failed",
                "previous_status": current.get("status"),
                "new_status": "failed",
                "message": error_message,
                "metadata": {"error": error_message},
            }
        )

        return updated

    def activate_model(
        self,
        model_version_id: str,
        created_by: str | None = None,
        reason: str | None = None,
    ) -> dict:
        model_version = self.model_version_repository.get_by_id(model_version_id)

        if not model_version:
            raise NotFoundError(f"Model version '{model_version_id}' was not found")

        job_profile_id = model_version.get("job_profile_id")

        if not job_profile_id:
            raise ValidationError("El modelo no tiene job_profile_id asociado")

        if model_version.get("status") not in ["completed", "selected", "active"]:
            raise ValidationError(
                "Solo se puede activar un modelo con estado completed, selected o active"
            )

        archived_models = self.model_version_repository.archive_active_by_job_profile(
            job_profile_id=job_profile_id,
            exclude_model_version_id=model_version_id,
        )

        for archived_model in archived_models:
            self.model_version_event_repository.create(
                {
                    "model_version_id": archived_model["id"],
                    "job_profile_id": job_profile_id,
                    "created_by": created_by,
                    "event_type": "archived",
                    "previous_status": archived_model.get("status"),
                    "new_status": "archived",
                    "message": "Modelo archivado por activación de una nueva versión",
                    "metadata": {
                        "activated_model_version_id": model_version_id,
                    },
                }
            )

        active_model = self.model_version_repository.mark_as_active(
            model_version_id=model_version_id,
            selection_reason=reason,
        )

        self.training_run_repository.mark_selected_by_model_version(
            job_profile_id=job_profile_id,
            model_version_id=model_version_id,
        )

        self.model_version_event_repository.create(
            {
                "model_version_id": model_version_id,
                "job_profile_id": job_profile_id,
                "created_by": created_by,
                "event_type": "activated",
                "previous_status": model_version.get("status"),
                "new_status": "active",
                "message": reason or "Modelo marcado como activo",
                "metadata": {
                    "selection_reason": reason,
                },
            }
        )

        return active_model

    def get_active_model(self, job_profile_id: str) -> dict | None:
        return self.model_version_repository.get_active_by_job_profile(job_profile_id)

    def list_versions_by_job_profile(self, job_profile_id: str) -> list[dict]:
        return self.model_version_repository.list_by_job_profile(job_profile_id)

    def list_audit_events(self, model_version_id: str) -> list[dict]:
        return self.model_version_event_repository.list_by_model_version(
            model_version_id
        )
