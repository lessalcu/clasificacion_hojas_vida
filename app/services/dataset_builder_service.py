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
    TERM_ALIASES = {
        "python": {"python", "py", "flask", "fastapi", "fast api", "django"},
        "flask": {"flask"},
        "fastapi": {"fastapi", "fast api"},
        "django": {"django"},
        "apis rest": {"api", "apis", "rest", "api rest", "apis rest"},
        "api rest": {"api", "apis", "rest", "api rest", "apis rest"},
        "backend": {
            "backend",
            "back-end",
            "back end",
            "api",
            "rest",
            "microservicio",
            "microservicios",
        },
        "back-end": {"backend", "back-end", "back end", "api", "rest"},
        "postgresql": {"postgresql", "postgres", "sql"},
        "postgres": {"postgresql", "postgres", "sql"},
        "sql": {"sql", "postgresql", "postgres", "mysql", "sql server"},
        "supabase": {"supabase", "postgresql", "postgres"},
        "docker": {"docker", "contenedor", "contenedores", "containers"},
        "frontend": {"frontend", "front-end", "front end", "react", "angular", "vue"},
        "full stack": {"full stack", "fullstack", "backend", "frontend"},
        "java": {"java", "spring", "spring boot"},
        "spring boot": {"spring boot", "spring", "java"},
        "javascript": {"javascript", "js", "node", "nodejs", "node.js"},
        "typescript": {"typescript", "ts"},
        "react": {"react", "reactjs", "react.js"},
        "node": {"node", "nodejs", "node.js", "javascript"},
    }

    STOPWORDS = {
        "de",
        "del",
        "la",
        "el",
        "los",
        "las",
        "para",
        "por",
        "con",
        "en",
        "un",
        "una",
        "y",
        "o",
        "a",
        "al",
        "developer",
        "desarrollador",
        "desarrollo",
        "perfil",
        "puesto",
        "vacante",
    }

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
        positive_ratio = (
            positive_ratio
            if positive_ratio is not None
            else current_app.config["DATASET_POSITIVE_RATIO"]
        )
        match_threshold = (
            match_threshold
            if match_threshold is not None
            else current_app.config["DATASET_MATCH_THRESHOLD"]
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

            score_detail = self._calculate_match_score_detail(
                job_profile=job_profile,
                candidate_profile=candidate_profile,
                candidate_text=candidate_text,
            )

            scored_profiles.append(
                {
                    "candidate_profile": candidate_profile,
                    "candidate_profile_id": candidate_profile_id,
                    "candidate_text": candidate_text,
                    "job_text": build_job_profile_text(job_profile),
                    "match_score": score_detail["final_score"],
                    "rubric": score_detail,
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
                "label_reason": item.get("label_reason"),
                "split_name": split_name,
                "source_type": source_type,
                "match_score": round(float(item["match_score"]), 4),
                "label_strategy": label_strategy,
                "quality_status": "ready",
                "dataset_version": dataset_version,
                "auto_generated": True,
                "rubric": item.get("rubric") or {},
                "notes": None,
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
        strategy = (strategy or "hybrid").lower().strip()

        if strategy == "threshold":
            for item in scored_profiles:
                item["label"] = item["match_score"] >= match_threshold
                item["label_reason"] = (
                    "threshold_positive" if item["label"] else "threshold_negative"
                )

            if self._has_both_classes(scored_profiles):
                return scored_profiles

        if strategy in ("rule", "rules", "rule_based"):
            for item in scored_profiles:
                item["label"] = bool(item.get("rubric", {}).get("rule_positive"))
                item["label_reason"] = (
                    "rule_positive" if item["label"] else "rule_negative"
                )

            if self._has_both_classes(scored_profiles):
                return scored_profiles

        if strategy == "hybrid":
            for item in scored_profiles:
                rule_positive = bool(item.get("rubric", {}).get("rule_positive"))
                threshold_positive = item["match_score"] >= match_threshold

                item["label"] = rule_positive or threshold_positive
                item["label_reason"] = (
                    "hybrid_positive" if item["label"] else "hybrid_negative"
                )

            positive_rate = self._positive_rate(scored_profiles)

            if (
                self._has_both_classes(scored_profiles)
                and 0.15 <= positive_rate <= 0.65
            ):
                return scored_profiles

        return self._assign_relative_labels(
            scored_profiles=scored_profiles,
            positive_ratio=positive_ratio,
        )

    @staticmethod
    def _assign_relative_labels(
        scored_profiles: list[dict[str, Any]],
        positive_ratio: float,
    ) -> list[dict[str, Any]]:
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
            item["label_reason"] = (
                "relative_top_ratio" if item["label"] else "relative_outside_ratio"
            )

        return sorted_profiles

    @staticmethod
    def _positive_rate(items: list[dict[str, Any]]) -> float:
        if not items:
            return 0.0

        positives = sum(1 for item in items if bool(item.get("label")))
        return positives / len(items)

    @staticmethod
    def _has_both_classes(items: list[dict[str, Any]]) -> bool:
        labels = {bool(item.get("label")) for item in items}
        return labels == {True, False}

    def _calculate_match_score_detail(
        self,
        job_profile: dict,
        candidate_profile: dict,
        candidate_text: str,
    ) -> dict[str, Any]:
        required_skills = self._normalize_terms(job_profile.get("required_skills"))
        technologies = self._normalize_terms(job_profile.get("technologies"))
        languages = self._normalize_terms(job_profile.get("languages"))

        candidate_skills = self._normalize_terms(candidate_profile.get("skills"))
        candidate_technologies = self._normalize_terms(
            candidate_profile.get("technologies")
        )
        candidate_languages = self._normalize_terms(candidate_profile.get("languages"))
        candidate_keywords = self._normalize_terms(candidate_profile.get("keywords"))

        title_terms = self._filter_relevant_terms(
            self._tokenize(str(job_profile.get("title") or ""))
        )
        description_terms = self._filter_relevant_terms(
            self._tokenize(str(job_profile.get("description") or ""))
        )

        job_general_terms = title_terms | description_terms

        candidate_tokens = self._tokenize(candidate_text)

        candidate_terms = (
            candidate_skills
            | candidate_technologies
            | candidate_languages
            | candidate_keywords
            | candidate_tokens
        )

        candidate_terms = self._expand_terms(candidate_terms)

        skill_score = self._overlap_score(required_skills, candidate_terms)
        technology_score = self._overlap_score(technologies, candidate_terms)
        language_score = self._overlap_score(languages, candidate_terms)
        general_text_score = self._overlap_score(job_general_terms, candidate_terms)

        matched_skills = self._matched_terms(required_skills, candidate_terms)
        matched_technologies = self._matched_terms(technologies, candidate_terms)
        matched_languages = self._matched_terms(languages, candidate_terms)
        matched_general_terms = self._matched_terms(job_general_terms, candidate_terms)

        rule_score = (
            len(matched_skills) * 2.0
            + len(matched_technologies) * 1.5
            + len(matched_general_terms) * 0.5
            + len(matched_languages) * 0.2
        )

        expected_weight = (
            len(required_skills) * 2.0
            + len(technologies) * 1.5
            + len(job_general_terms) * 0.5
            + len(languages) * 0.2
        )

        rule_score_normalized = (
            rule_score / expected_weight if expected_weight > 0 else 0.0
        )

        final_score = (
            skill_score * 0.40
            + technology_score * 0.30
            + language_score * 0.05
            + general_text_score * 0.15
            + rule_score_normalized * 0.10
        )

        final_score = max(0.0, min(1.0, final_score))

        required_hits = len(matched_skills) + len(matched_technologies)

        rule_positive = (
            rule_score >= 4.0
            or (len(matched_skills) >= 1 and required_hits >= 2 and final_score >= 0.25)
            or final_score >= 0.45
        )

        return {
            "final_score": round(final_score, 4),
            "skill_score": round(skill_score, 4),
            "technology_score": round(technology_score, 4),
            "language_score": round(language_score, 4),
            "general_text_score": round(general_text_score, 4),
            "rule_score": round(rule_score, 4),
            "rule_score_normalized": round(rule_score_normalized, 4),
            "rule_positive": rule_positive,
            "matched_skills": sorted(matched_skills),
            "matched_technologies": sorted(matched_technologies),
            "matched_languages": sorted(matched_languages),
            "matched_general_terms": sorted(matched_general_terms),
            "required_hits": required_hits,
        }

    def _normalize_terms(self, value: Any) -> set[str]:
        terms = normalize_text_list(value)
        normalized_terms = set()

        for term in terms:
            normalized = term.strip().lower()
            if normalized:
                normalized_terms.add(normalized)

        return self._expand_terms(normalized_terms)

    def _expand_terms(self, terms: set[str]) -> set[str]:
        expanded_terms = set(terms)

        for term in list(terms):
            normalized = term.lower().strip()

            if normalized in self.TERM_ALIASES:
                expanded_terms.update(self.TERM_ALIASES[normalized])

            for alias_key, aliases in self.TERM_ALIASES.items():
                if normalized in aliases:
                    expanded_terms.add(alias_key)
                    expanded_terms.update(aliases)

        return {term.strip().lower() for term in expanded_terms if term.strip()}

    def _filter_relevant_terms(self, terms: set[str]) -> set[str]:
        return {
            term
            for term in terms
            if term not in self.STOPWORDS and len(term.strip()) >= 3
        }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        text = text.lower()
        tokens = re.findall(r"[a-záéíóúñ0-9\+#\.]{2,}", text)
        return {token.strip() for token in tokens if token.strip()}

    def _overlap_score(
        self,
        expected_terms: set[str],
        candidate_terms: set[str],
    ) -> float:
        if not expected_terms:
            return 0.0

        matched = self._matched_terms(expected_terms, candidate_terms)
        return len(matched) / len(expected_terms)

    def _matched_terms(
        self,
        expected_terms: set[str],
        candidate_terms: set[str],
    ) -> set[str]:
        matched_terms = set()

        expanded_candidate_terms = self._expand_terms(candidate_terms)

        for expected in expected_terms:
            expected_lower = expected.lower().strip()
            expected_aliases = self._expand_terms({expected_lower})

            if expected_aliases & expanded_candidate_terms:
                matched_terms.add(expected_lower)
                continue

            for candidate in expanded_candidate_terms:
                if expected_lower in candidate or candidate in expected_lower:
                    matched_terms.add(expected_lower)
                    break

        return matched_terms

    @staticmethod
    def _build_split_name(candidate_profile_id: str, job_profile_id: str) -> str:
        seed = f"{candidate_profile_id}-{job_profile_id}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) % 100

        return "test" if value < 20 else "train"
