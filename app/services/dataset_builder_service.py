from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from flask import current_app

from app.errors.exceptions import NotFoundError, ValidationError
from app.machine_learning.common.utils import (
    build_candidate_text,
    build_job_profile_text,
    normalize_text_list,
)
from app.repositories.candidate_profile_repository import CandidateProfileRepository
from app.repositories.dataset_sample_repository import DatasetSampleRepository
from app.repositories.job_profile_repository import JobProfileRepository
from app.services.text_cleaning_service import TextCleaningService


class DatasetBuilderService:
    def __init__(self):
        self.job_profile_repository = JobProfileRepository()
        self.candidate_profile_repository = CandidateProfileRepository()
        self.dataset_sample_repository = DatasetSampleRepository()
        self.text_cleaning_service = TextCleaningService()

    def build_for_job_profile(
        self,
        job_profile_id: str,
        created_by: str | None = None,
        dataset_version: str = "auto-v1",
        source_type: str = "auto",
        limit: int = 500,
        overwrite_auto_labels: bool = True,
        label_strategy: str | None = None,
        positive_ratio: float | None = None,
        match_threshold: float | None = None,
    ) -> dict[str, Any]:
        job_profile = self.job_profile_repository.get_by_id(job_profile_id)

        if not job_profile:
            raise NotFoundError(f"Job profile '{job_profile_id}' was not found")

        min_text_length = current_app.config["DATASET_MIN_TEXT_LENGTH"]
        label_strategy = (
            label_strategy or current_app.config["DATASET_AUTO_LABEL_STRATEGY"]
        )
        positive_ratio = positive_ratio or current_app.config["DATASET_POSITIVE_RATIO"]
        match_threshold = (
            match_threshold or current_app.config["DATASET_MATCH_THRESHOLD"]
        )

        candidate_profiles = self.candidate_profile_repository.get_profiles_for_dataset(
            limit=limit
        )

        if not candidate_profiles:
            raise ValidationError(
                "No existen perfiles de candidatos válidos para construir dataset_sample"
            )

        scored_profiles: list[dict[str, Any]] = []

        for candidate_profile in candidate_profiles:
            candidate_profile_id = candidate_profile.get("id")
            if not candidate_profile_id:
                continue

            candidate_text = build_candidate_text(candidate_profile)
            candidate_text = self.text_cleaning_service.clean(candidate_text)

            if not self.text_cleaning_service.is_valid_for_training(
                candidate_text,
                min_length=min_text_length,
            ):
                continue

            job_text = build_job_profile_text(job_profile)
            score = self._calculate_match_score(
                job_profile, candidate_profile, candidate_text
            )

            scored_profiles.append(
                {
                    "candidate_profile": candidate_profile,
                    "candidate_profile_id": candidate_profile_id,
                    "candidate_text": candidate_text,
                    "job_text": job_text,
                    "match_score": score,
                }
            )

        if len(scored_profiles) < 2:
            raise ValidationError(
                "No hay suficientes perfiles válidos para crear dataset_sample. "
                "Se necesitan al menos 2 perfiles con texto limpio."
            )

        labeled_profiles = self._assign_labels(
            scored_profiles=scored_profiles,
            strategy=label_strategy,
            positive_ratio=positive_ratio,
            match_threshold=match_threshold,
        )

        created_count = 0
        updated_count = 0
        skipped_count = 0
        positive_count = 0
        negative_count = 0

        for item in labeled_profiles:
            label = bool(item["label"])
            positive_count += 1 if label else 0
            negative_count += 0 if label else 1

            split_name = self._build_split_name(
                candidate_profile_id=item["candidate_profile_id"],
                job_profile_id=job_profile_id,
            )

            payload = {
                "candidate_profile_id": item["candidate_profile_id"],
                "job_profile_id": job_profile_id,
                "created_by": created_by,
                "label": label,
                "split_name": split_name,
                "source_type": source_type,
                "match_score": round(float(item["match_score"]), 4),
                "label_strategy": label_strategy,
                "quality_status": "ready",
                "dataset_version": dataset_version,
                "auto_generated": True,
                "notes": (
                    "Registro generado automáticamente desde candidate_profile "
                    "para habilitar entrenamiento inicial."
                ),
                "updated_at": datetime.now(UTC).isoformat(),
            }

            existing = self.dataset_sample_repository.find_by_candidate_profile_and_job(
                candidate_profile_id=item["candidate_profile_id"],
                job_profile_id=job_profile_id,
            )

            if existing:
                existing_strategy = existing.get("label_strategy") or ""
                existing_auto = bool(existing.get("auto_generated"))

                if overwrite_auto_labels and (
                    existing_auto or existing_strategy.startswith("auto")
                ):
                    self.dataset_sample_repository.update(existing["id"], payload)
                    updated_count += 1
                else:
                    skipped_count += 1
            else:
                payload["created_at"] = datetime.now(UTC).isoformat()
                self.dataset_sample_repository.create(payload)
                created_count += 1

        return {
            "job_profile_id": job_profile_id,
            "dataset_version": dataset_version,
            "source_type": source_type,
            "label_strategy": label_strategy,
            "total_candidate_profiles_found": len(candidate_profiles),
            "valid_profiles_for_dataset": len(scored_profiles),
            "created": created_count,
            "updated": updated_count,
            "skipped": skipped_count,
            "positive_labels": positive_count,
            "negative_labels": negative_count,
            "message": "dataset_sample construido correctamente",
        }

    def ensure_dataset_for_training(
        self,
        job_profile_id: str,
        created_by: str | None = None,
        dataset_version: str = "auto-before-training",
    ) -> dict[str, Any]:
        min_rows = current_app.config["MIN_TRAINING_ROWS"]
        current_rows = self.dataset_sample_repository.count_ready_labeled_by_job(
            job_profile_id
        )

        if current_rows >= min_rows:
            return {
                "built": False,
                "reason": "dataset_sample ya tiene suficientes registros listos",
                "current_ready_labeled_rows": current_rows,
                "min_training_rows": min_rows,
            }

        result = self.build_for_job_profile(
            job_profile_id=job_profile_id,
            created_by=created_by,
            dataset_version=dataset_version,
            source_type="auto-before-training",
        )

        result["built"] = True
        result["previous_ready_labeled_rows"] = current_rows
        result["min_training_rows"] = min_rows

        return result

    def _assign_labels(
        self,
        scored_profiles: list[dict[str, Any]],
        strategy: str,
        positive_ratio: float,
        match_threshold: float,
    ) -> list[dict[str, Any]]:
        if strategy == "threshold":
            for item in scored_profiles:
                item["label"] = item["match_score"] >= match_threshold

            if self._has_both_classes(scored_profiles):
                return scored_profiles

        sorted_profiles = sorted(
            scored_profiles,
            key=lambda item: item["match_score"],
            reverse=True,
        )

        total = len(sorted_profiles)
        positive_count = round(total * positive_ratio)
        positive_count = max(1, positive_count)
        positive_count = min(positive_count, total - 1)

        positive_ids = {
            item["candidate_profile_id"] for item in sorted_profiles[:positive_count]
        }

        for item in sorted_profiles:
            item["label"] = item["candidate_profile_id"] in positive_ids

        return sorted_profiles

    @staticmethod
    def _has_both_classes(items: list[dict[str, Any]]) -> bool:
        labels = {bool(item.get("label")) for item in items}
        return labels == {True, False}

    def _calculate_match_score(
        self,
        job_profile: dict,
        candidate_profile: dict,
        candidate_text: str,
    ) -> float:
        required_skills = self._normalize_terms(job_profile.get("required_skills"))
        technologies = self._normalize_terms(job_profile.get("technologies"))
        languages = self._normalize_terms(job_profile.get("languages"))

        candidate_skills = self._normalize_terms(candidate_profile.get("skills"))
        candidate_technologies = self._normalize_terms(
            candidate_profile.get("technologies")
        )
        candidate_languages = self._normalize_terms(candidate_profile.get("languages"))

        candidate_tokens = self._tokenize(candidate_text)

        skill_score = self._overlap_score(
            required_skills, candidate_skills | candidate_tokens
        )
        technology_score = self._overlap_score(
            technologies,
            candidate_technologies | candidate_tokens,
        )
        language_score = self._overlap_score(
            languages, candidate_languages | candidate_tokens
        )

        title_terms = self._tokenize(str(job_profile.get("title") or ""))
        description_terms = self._tokenize(str(job_profile.get("description") or ""))
        job_general_terms = title_terms | description_terms
        general_text_score = self._overlap_score(job_general_terms, candidate_tokens)

        final_score = (
            skill_score * 0.45
            + technology_score * 0.35
            + language_score * 0.05
            + general_text_score * 0.15
        )

        return max(0.0, min(1.0, final_score))

    @staticmethod
    def _normalize_terms(value: Any) -> set[str]:
        terms = normalize_text_list(value)
        normalized_terms = set()

        for term in terms:
            normalized = term.strip().lower()
            if normalized:
                normalized_terms.add(normalized)

        return normalized_terms

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        text = text.lower()
        tokens = re.findall(r"[a-záéíóúñ0-9\+#\.]{2,}", text)
        return {token.strip() for token in tokens if token.strip()}

    @staticmethod
    def _overlap_score(expected_terms: set[str], candidate_terms: set[str]) -> float:
        if not expected_terms:
            return 0.0

        matched = 0

        for expected in expected_terms:
            expected_lower = expected.lower().strip()

            if expected_lower in candidate_terms:
                matched += 1
                continue

            for candidate in candidate_terms:
                if expected_lower in candidate or candidate in expected_lower:
                    matched += 1
                    break

        return matched / len(expected_terms)

    @staticmethod
    def _build_split_name(candidate_profile_id: str, job_profile_id: str) -> str:
        seed = f"{candidate_profile_id}-{job_profile_id}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) % 100

        return "test" if value < 20 else "train"
