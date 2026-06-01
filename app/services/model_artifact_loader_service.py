from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from flask import current_app

from app.errors.exceptions import ValidationError
from app.services.storage_service import StorageService


class ModelArtifactLoaderService:
    def __init__(self):
        self.storage_service = StorageService()

    def load_bundle(self, model_version: dict) -> dict[str, Any]:
        model_version_id = model_version.get("id")
        artifact_bucket = model_version.get("artifact_bucket")
        artifact_path = model_version.get("artifact_path")

        if not model_version_id:
            raise ValidationError("El modelo activo no tiene id")

        if not artifact_bucket or not artifact_path:
            raise ValidationError("El modelo activo no tiene artefacto configurado")

        local_path = self._build_local_cache_path(
            model_version_id=model_version_id,
            artifact_path=artifact_path,
        )

        if not Path(local_path).exists():
            self.storage_service.download_file(
                bucket_name=artifact_bucket,
                remote_path=artifact_path,
                local_file_path=local_path,
            )

        bundle = joblib.load(local_path)

        return self._normalize_bundle(bundle)

    def _build_local_cache_path(self, model_version_id: str, artifact_path: str) -> str:
        cache_root = Path(current_app.config["INFERENCE_MODEL_CACHE_DIR"])
        filename = Path(artifact_path).name or "bundle.joblib"

        target_path = cache_root / model_version_id / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        return str(target_path)

    @staticmethod
    def _normalize_bundle(bundle) -> dict[str, Any]:
        if isinstance(bundle, dict):
            estimator = (
                bundle.get("estimator")
                or bundle.get("model")
                or bundle.get("classifier")
            )

            vectorizer = bundle.get("vectorizer") or bundle.get("tfidf_vectorizer")

            metadata = bundle.get("metadata") or bundle.get("training_metadata") or {}

            if estimator is None or vectorizer is None:
                raise ValidationError(
                    "El bundle no contiene estimator/model y vectorizer"
                )

            return {
                "estimator": estimator,
                "vectorizer": vectorizer,
                "metadata": metadata,
            }

        estimator = getattr(bundle, "estimator", None)
        vectorizer = getattr(bundle, "vectorizer", None)

        if estimator is None or vectorizer is None:
            raise ValidationError(
                "No se pudo interpretar el bundle del modelo entrenado"
            )

        return {
            "estimator": estimator,
            "vectorizer": vectorizer,
            "metadata": {},
        }
